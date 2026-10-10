"""
Audio length read from the containers soundfile can't open: MP4 (m4a, mov, 3gp), Matroska (webm, mkv) and ADTS AAC.

Every length a container states or implies counts, and the longest one wins: a provider bills the audio it decodes,
and a declared duration can be shorter than the samples a file carries. Each pass walks the upload once, so a
crafted file costs time in proportion to its size.
"""

import math
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from itertools import groupby, takewhile
from types import MappingProxyType
from typing import Final

_NANOSECONDS_PER_SECOND: Final = 1_000_000_000
_MP4_FIRST_BOXES: Final = frozenset((b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide"))
_MATROSKA_MAGIC: Final = b"\x1a\x45\xdf\xa3"
_ADTS_SAMPLE_RATES: Final = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350)
_AAC_SAMPLES_PER_BLOCK: Final = 1024


def container_duration_seconds(content: bytes) -> float | None:
    if content[4:8] in _MP4_FIRST_BOXES:
        return _mp4_seconds(content)
    if content.startswith(_MATROSKA_MAGIC):
        return _matroska_seconds(content)
    return _adts_seconds(content)


def _longest(readings: Iterable[float]) -> float | None:
    longest: Final = max((reading for reading in readings if math.isfinite(reading)), default=0.0)
    return longest if longest > 0 else None


def _uint(content: bytes, start: int, size: int) -> int:
    return int.from_bytes(content[start : start + size], "big")


@dataclass(frozen=True, slots=True)
class _Box:
    kind: bytes
    start: int
    end: int


def _boxes(content: bytes, start: int, end: int) -> Iterator[_Box]:
    offset = start  # rebind-ok: a cursor over consecutive boxes
    while offset + 8 <= end:
        declared = _uint(content, offset, 4)
        if declared == 1 and offset + 16 > end:
            return
        header = 16 if declared == 1 else 8
        size = _uint(content, offset + 8, 8) if declared == 1 else declared or end - offset
        if size < header or offset + size > end:
            return
        yield _Box(content[offset + 4 : offset + 8], offset + header, offset + size)
        offset += size


def _children(content: bytes, box: _Box, kind: bytes) -> Iterator[_Box]:
    return (child for child in _boxes(content, box.start, box.end) if child.kind == kind)


def _descendant(content: bytes, box: _Box | None, *path: bytes) -> _Box | None:
    if box is None or not path:
        return box
    return _descendant(content, next(_children(content, box, path[0]), None), *path[1:])


def _wide(content: bytes, box: _Box) -> bool:
    """Whether a full box is version 1, whose times and durations are 64-bit."""
    return box.end > box.start and content[box.start] == 1


def _clock(content: bytes, box: _Box | None) -> tuple[int, int]:
    """Timescale and duration of an mvhd or mdhd box. An unknown duration (all ones) reads as 0."""
    if box is None:
        return 0, 0
    wide: Final = _wide(content, box)
    if box.end - box.start < (32 if wide else 20):
        return 0, 0
    timescale: Final = _uint(content, box.start + (20 if wide else 12), 4)
    duration: Final = _uint(content, box.start + (24 if wide else 16), 8 if wide else 4)
    return timescale, 0 if duration == (1 << (64 if wide else 32)) - 1 else duration


def _seconds(ticks: int, timescale: int) -> float:
    return ticks / timescale if timescale > 0 else 0.0


@dataclass(frozen=True, slots=True)
class _Mp4Track:
    track_id: int
    timescale: int
    declared_ticks: int
    sample_ticks: int


def _mp4_track(content: bytes, trak: _Box) -> _Mp4Track | None:
    header: Final = _descendant(content, trak, b"tkhd")
    if header is None or header.end - header.start < 24:
        return None
    timescale, declared = _clock(content, _descendant(content, trak, b"mdia", b"mdhd"))
    stts: Final = _descendant(content, trak, b"mdia", b"minf", b"stbl", b"stts")
    return _Mp4Track(
        track_id=_uint(content, header.start + (20 if _wide(content, header) else 12), 4),
        timescale=timescale,
        declared_ticks=declared,
        sample_ticks=0 if stts is None else _stts_ticks(content, stts),
    )


def _stts_ticks(content: bytes, stts: _Box) -> int:
    entries: Final = min(_uint(content, stts.start + 4, 4), (stts.end - stts.start - 8) // 8)
    return sum(
        _uint(content, stts.start + 8 + 8 * index, 4) * _uint(content, stts.start + 12 + 8 * index, 4)
        for index in range(entries)
    )


def _trex_default_ticks(content: bytes, moov: _Box) -> Mapping[int, int]:
    mvex: Final = _descendant(content, moov, b"mvex")
    if mvex is None:
        return MappingProxyType({})
    return MappingProxyType(
        {
            _uint(content, trex.start + 4, 4): _uint(content, trex.start + 12, 4)
            for trex in _children(content, mvex, b"trex")
        }
    )


def _mehd_ticks(content: bytes, moov: _Box) -> int:
    mehd: Final = _descendant(content, moov, b"mvex", b"mehd")
    if mehd is None:
        return 0
    return _uint(content, mehd.start + 4, 8 if _wide(content, mehd) else 4)


def _trafs(content: bytes) -> Iterator[_Box]:
    for moof in _boxes(content, 0, len(content)):
        if moof.kind == b"moof":
            yield from _children(content, moof, b"traf")


@dataclass(frozen=True, slots=True)
class _Fragment:
    track_id: int
    start_ticks: int
    ticks: int


def _fragment(content: bytes, traf: _Box, trex_ticks: Mapping[int, int]) -> _Fragment | None:
    tfhd: Final = _descendant(content, traf, b"tfhd")
    if tfhd is None:
        return None
    flags: Final = _uint(content, tfhd.start + 1, 3)
    track_id: Final = _uint(content, tfhd.start + 4, 4)
    default_at: Final = tfhd.start + 8 + (8 if flags & 0x01 else 0) + (4 if flags & 0x02 else 0)
    default: Final = _uint(content, default_at, 4) if flags & 0x08 else trex_ticks.get(track_id, 0)
    tfdt: Final = _descendant(content, traf, b"tfdt")
    return _Fragment(
        track_id=track_id,
        start_ticks=0 if tfdt is None else _uint(content, tfdt.start + 4, 8 if _wide(content, tfdt) else 4),
        ticks=sum(_trun_ticks(content, trun, default) for trun in _children(content, traf, b"trun")),
    )


def _trun_ticks(content: bytes, trun: _Box, default: int) -> int:
    flags: Final = _uint(content, trun.start + 1, 3)
    declared_samples: Final = _uint(content, trun.start + 4, 4)
    if not flags & 0x100:
        return declared_samples * default
    first: Final = trun.start + 8 + (4 if flags & 0x01 else 0) + (4 if flags & 0x04 else 0)
    stride: Final = 4 * bin(flags & 0xF00).count("1")
    samples: Final = min(declared_samples, (trun.end - first) // stride)
    return sum(_uint(content, first + stride * index, 4) for index in range(samples))


def _fragments_ticks(fragments: tuple[_Fragment, ...]) -> int:
    """
    The longer of the summed sample durations and the span from the first fragment's start to the last one's end.
    Chromium's recorder states a short duration for each fragment's last sample, and the start times keep the timeline.
    """
    span: Final = max(f.start_ticks + f.ticks for f in fragments) - min(f.start_ticks for f in fragments)
    return max(sum(f.ticks for f in fragments), span)


def _track_of(fragment: _Fragment) -> int:
    return fragment.track_id


def _fragment_ticks_by_track(content: bytes, trex_ticks: Mapping[int, int]) -> Mapping[int, int]:
    fragments: Final = sorted(
        (fragment for traf in _trafs(content) if (fragment := _fragment(content, traf, trex_ticks)) is not None),
        key=_track_of,
    )
    return MappingProxyType(
        {track_id: _fragments_ticks(tuple(group)) for track_id, group in groupby(fragments, key=_track_of)}
    )


def _mp4_seconds(content: bytes) -> float | None:
    moov: Final = next((box for box in _boxes(content, 0, len(content)) if box.kind == b"moov"), None)
    if moov is None:
        return None
    movie_timescale, movie_ticks = _clock(content, _descendant(content, moov, b"mvhd"))
    tracks: Final = tuple(
        track for trak in _children(content, moov, b"trak") if (track := _mp4_track(content, trak)) is not None
    )
    fragments: Final = _fragment_ticks_by_track(content, _trex_default_ticks(content, moov))
    return _longest(
        (
            _seconds(movie_ticks, movie_timescale),
            _seconds(_mehd_ticks(content, moov), movie_timescale),
            *(_seconds(track.declared_ticks, track.timescale) for track in tracks),
            *(_seconds(track.sample_ticks, track.timescale) for track in tracks),
            *(_seconds(fragments.get(track.track_id, 0), track.timescale) for track in tracks),
        )
    )


class _Ebml(IntEnum):
    SEGMENT = 0x18538067
    INFO = 0x1549A966
    TRACKS = 0x1654AE6B
    CLUSTER = 0x1F43B675
    TIMESTAMP_SCALE = 0x2AD7B1
    DURATION = 0x4489
    TRACK_ENTRY = 0xAE
    TRACK_NUMBER = 0xD7
    CODEC_ID = 0x86
    DEFAULT_DURATION = 0x23E383
    CLUSTER_TIMESTAMP = 0xE7
    SIMPLE_BLOCK = 0xA3
    BLOCK_GROUP = 0xA0
    BLOCK = 0xA1
    BLOCK_DURATION = 0x9B


_ENTERED_MASTERS: Final = frozenset((_Ebml.SEGMENT, _Ebml.INFO, _Ebml.TRACKS, _Ebml.CLUSTER))
_DEFAULT_TIMESTAMP_SCALE_NS: Final = 1_000_000


def _varint_length(content: bytes, offset: int) -> int:
    """Bytes in the EBML variable-size integer at `offset`, or 0 when there is none."""
    if offset >= len(content) or content[offset] == 0:
        return 0
    length: Final = 9 - content[offset].bit_length()
    return length if offset + length <= len(content) else 0


def _varint_value(content: bytes, offset: int, length: int) -> int:
    return _uint(content, offset, length) & ((1 << 7 * length) - 1)


@dataclass(frozen=True, slots=True)
class _Element:
    element_id: int
    start: int
    end: int


def _elements(content: bytes, start: int, end: int) -> Iterator[_Element]:
    """
    Elements in file order. A master a length depends on is entered rather than skipped, so its children follow it
    even when its size is unknown, as in a live recording; its `end` is then its parent's.
    """
    offset = start  # rebind-ok: a cursor over consecutive elements
    while offset < end:
        id_length = _varint_length(content, offset)
        size_length = _varint_length(content, offset + id_length) if id_length else 0
        if not size_length:
            return
        element_id = _uint(content, offset, id_length)
        payload = offset + id_length + size_length
        if element_id in _ENTERED_MASTERS:
            yield _Element(element_id, payload, end)
            offset = payload
            continue
        size = _varint_value(content, offset + id_length, size_length)
        if size == (1 << 7 * size_length) - 1 or payload + size > end:
            return
        yield _Element(element_id, payload, payload + size)
        offset = payload + size


def _element_uint(content: bytes, element: _Element) -> int:
    return _uint(content, element.start, element.end - element.start)


def _element_float(content: bytes, element: _Element) -> float:
    unpacked: Final[tuple[float, ...]] = struct.unpack(
        ">d" if element.end - element.start == 8 else ">f", content[element.start : element.end]
    )
    return unpacked[0]


@dataclass(frozen=True, slots=True)
class _MatroskaTrack:
    default_duration_ns: int
    opus: bool


@dataclass(frozen=True, slots=True)
class _MatroskaHeader:
    timestamp_scale_ns: int
    declared_ticks: float
    tracks: Mapping[int, _MatroskaTrack]


def _track_entry(content: bytes, entry: _Element) -> tuple[int, _MatroskaTrack]:
    fields: Final = MappingProxyType({child.element_id: child for child in _elements(content, entry.start, entry.end)})
    number: Final = fields.get(_Ebml.TRACK_NUMBER)
    default_duration: Final = fields.get(_Ebml.DEFAULT_DURATION)
    codec: Final = fields.get(_Ebml.CODEC_ID)
    return 0 if number is None else _element_uint(content, number), _MatroskaTrack(
        default_duration_ns=0 if default_duration is None else _element_uint(content, default_duration),
        opus=codec is not None and content[codec.start : codec.end].rstrip(b"\x00") == b"A_OPUS",
    )


def _before_clusters(content: bytes, element_id: int) -> Iterator[_Element]:
    header: Final = takewhile(lambda element: element.element_id != _Ebml.CLUSTER, _elements(content, 0, len(content)))
    return (element for element in header if element.element_id == element_id)


def _matroska_header(content: bytes) -> _MatroskaHeader:
    scale: Final = next((_element_uint(content, e) for e in _before_clusters(content, _Ebml.TIMESTAMP_SCALE)), 0)
    declared: Final = next(
        (
            _element_float(content, element)
            for element in _before_clusters(content, _Ebml.DURATION)
            if element.end - element.start in (4, 8)
        ),
        0.0,
    )
    return _MatroskaHeader(
        timestamp_scale_ns=scale or _DEFAULT_TIMESTAMP_SCALE_NS,
        declared_ticks=declared,
        tracks=MappingProxyType(
            dict(_track_entry(content, entry) for entry in _before_clusters(content, _Ebml.TRACK_ENTRY))
        ),
    )


@dataclass(frozen=True, slots=True)
class _Block:
    cluster_timestamp: int
    element: _Element
    duration_ticks: int | None


def _group_blocks(content: bytes, cluster_timestamp: int, group: _Element) -> Iterator[_Block]:
    duration: Final = next(
        (
            _element_uint(content, child)
            for child in _elements(content, group.start, group.end)
            if child.element_id == _Ebml.BLOCK_DURATION
        ),
        None,
    )
    return (
        _Block(cluster_timestamp, child, duration)
        for child in _elements(content, group.start, group.end)
        if child.element_id == _Ebml.BLOCK
    )


def _blocks(content: bytes) -> Iterator[_Block]:
    cluster_timestamp = 0  # rebind-ok: the Timestamp of the Cluster the next blocks belong to
    for element in _elements(content, 0, len(content)):
        if element.element_id == _Ebml.SIMPLE_BLOCK:
            yield _Block(cluster_timestamp, element, None)
        elif element.element_id == _Ebml.CLUSTER_TIMESTAMP:
            cluster_timestamp = _element_uint(content, element)
        elif element.element_id == _Ebml.BLOCK_GROUP:
            yield from _group_blocks(content, cluster_timestamp, element)


def _opus_frame_us(configuration: int) -> int:
    if configuration < 12:
        return (10_000, 20_000, 40_000, 60_000)[configuration % 4]
    if configuration < 16:
        return (10_000, 20_000)[configuration % 2]
    return (2_500, 5_000, 10_000, 20_000)[configuration % 4]


def _opus_packet_ns(content: bytes, start: int, end: int) -> int:
    """RFC 6716 section 3.1: the TOC byte's configuration sets the frame size and its code the frame count."""
    if start >= end:
        return 0
    code: Final = content[start] & 0x03
    frames: Final = (1, 2, 2, content[start + 1] & 0x3F if start + 1 < end else 0)[code]
    return _opus_frame_us(content[start] >> 3) * 1000 * frames


def _carried_ns(
    content: bytes,
    header: _MatroskaHeader,
    block: _Block,
    track: _MatroskaTrack | None,
    frames: int,
    packet: int | None,
) -> int:
    """
    The audio a block carries: its BlockDuration, else its frames at the track's DefaultDuration, else the length its
    Opus packet states. `packet` is where an unlaced block's only frame starts.
    """
    if block.duration_ticks is not None:
        return block.duration_ticks * header.timestamp_scale_ns
    if track is not None and track.default_duration_ns > 0:
        return frames * track.default_duration_ns
    if track is not None and track.opus and packet is not None:
        return _opus_packet_ns(content, packet, block.element.end)
    return 0


def _block_span(content: bytes, header: _MatroskaHeader, block: _Block) -> tuple[int, int]:
    """When the block's audio ends and how much audio it carries, in nanoseconds."""
    element: Final = block.element
    track_length: Final = _varint_length(content, element.start)
    timing: Final = element.start + track_length
    if not track_length or timing + 3 > element.end:
        return 0, 0
    track: Final = header.tracks.get(_varint_value(content, element.start, track_length))
    laced: Final = content[timing + 2] & 0x06
    frames: Final = content[timing + 3] + 1 if laced and timing + 3 < element.end else 1
    carried_ns: Final = _carried_ns(content, header, block, track, frames, None if laced else timing + 3)
    relative: Final = int.from_bytes(content[timing : timing + 2], "big", signed=True)
    return (block.cluster_timestamp + relative) * header.timestamp_scale_ns + carried_ns, carried_ns


def _tally(tally: tuple[int, int], span: tuple[int, int]) -> tuple[int, int]:
    return max(tally[0], span[0]), tally[1] + span[1]


def _matroska_seconds(content: bytes) -> float | None:
    header: Final = _matroska_header(content)
    latest_ns, carried_ns = reduce(_tally, (_block_span(content, header, block) for block in _blocks(content)), (0, 0))
    return _longest(
        (
            header.declared_ticks * header.timestamp_scale_ns / _NANOSECONDS_PER_SECOND,
            latest_ns / _NANOSECONDS_PER_SECOND,
            carried_ns / _NANOSECONDS_PER_SECOND,
        )
    )


def _id3_size(content: bytes) -> int:
    if not content.startswith(b"ID3") or len(content) < 10:
        return 0
    synchsafe: Final = (content[6] & 0x7F) << 21 | (content[7] & 0x7F) << 14 | (content[8] & 0x7F) << 7
    footer: Final = 10 if content[5] & 0x10 else 0
    return 10 + (synchsafe | content[9] & 0x7F) + footer


def _adts_frames(content: bytes) -> Iterator[float]:
    """Seconds of each complete ADTS frame from the start of the stream, up to the first bytes that aren't one."""
    offset = _id3_size(content)  # rebind-ok: a cursor over consecutive frames
    while offset + 7 <= len(content) and content[offset] == 0xFF and content[offset + 1] & 0xF6 == 0xF0:
        rate_index = content[offset + 2] >> 2 & 0x0F
        length = (content[offset + 3] & 0x03) << 11 | content[offset + 4] << 3 | content[offset + 5] >> 5
        if rate_index >= len(_ADTS_SAMPLE_RATES) or length < 7 or offset + length > len(content):
            return
        yield ((content[offset + 6] & 0x03) + 1) * _AAC_SAMPLES_PER_BLOCK / _ADTS_SAMPLE_RATES[rate_index]
        offset += length


def _adts_seconds(content: bytes) -> float | None:
    return _longest((sum(_adts_frames(content)),))
