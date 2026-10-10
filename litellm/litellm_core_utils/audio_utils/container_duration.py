"""
Audio length read from an upload's own bytes, for what soundfile can't measure: MP4 (m4a, mov, 3gp), Matroska (webm,
mkv) and ADTS AAC, a WAV whose header was never finalized, and MP3, whose length soundfile takes from a header that
can understate it. A file soundfile can't open that holds a stream of ADTS or MPEG audio frames reads as the longer
of the two kinds, so a file holding both bills the longer, whichever a decoder takes it for.

The length is the audio a decoder plays: the samples a file carries and their timestamps, on its audio tracks, each
measured from its first sample. A duration a header declares counts only when the file carries no samples, since a
buggy or crafted header can state any length. Like a decoder, the readers skip bytes they can't parse. A Matroska file
with more elements, skipped stretches included, or an MP4 with more fragments than any recording has reads as
unmeasured, and a stream of frames resyncs only on a valid frame header, so stray bytes cost one search each. Each
pass walks the upload once, so a crafted file costs time in proportion to its size.
"""

import math
import re
import struct
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum
from itertools import islice
from types import MappingProxyType
from typing import Final, TypeAlias

_NANOSECONDS_PER_SECOND: Final = 1_000_000_000
_MP4_FIRST_BOXES: Final = frozenset((b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide"))
_MATROSKA_MAGIC: Final = b"\x1a\x45\xdf\xa3"
_SOUNDFILE_MAGICS: Final = (
    b"OggS",
    b"fLaC",
    b"FORM",
    b".snd",
    b"dns.",
    b"caff",
    b"RF64",
    b"BW64",
    b"RIFX",
    b"riff",
    b"NIST",
    b"Creative Voice File",
    b"\x64\xa3",
    b"\x00\x01\xa3\x64",
    b"\x00\x02\xa3\x64",
    b"\x00\x03\xa3\x64",
    b"\x00\x04\xa3\x64",
)


@dataclass(frozen=True, slots=True)
class _Lengths:
    played: tuple[float, ...]
    declared: tuple[float, ...] = ()


def container_duration_seconds(content: bytes) -> float | None:
    """The length of an upload soundfile can't read."""
    if content[4:8] in _MP4_FIRST_BOXES:
        return _length(_mp4_lengths(content))
    if content.startswith(_MATROSKA_MAGIC):
        return _length(_matroska_lengths(content))
    if content.startswith(b"RIFF") and content[8:12] == b"WAVE":
        return _streamed_wav_seconds(content)
    start: Final = _id3_end(content, 0)
    if content.startswith(_SOUNDFILE_MAGICS) or content.startswith(_SOUNDFILE_MAGICS, start):
        return None
    adts: Final = _first_frame(content, start, _adts_frame, _ADTS_SYNC) is not None
    if adts or _first_frame(content, start, _mpeg_frame, _MPEG_SYNC) is not None:
        return _frame_stream_seconds(content, start)
    return None


def mp3_duration_seconds(content: bytes, soundfile_seconds: float) -> float:
    """
    The length of an MP3 soundfile opened, by its MPEG frames from the end of its ID3 tag. soundfile takes the length
    from a Xing or Info header, which can state far fewer frames than the file carries, or without one estimates it
    from the first frames' bitrate, which can be far off for VBR. So the frames decide when frames that follow one
    another make up at least half the bytes. Fewer means they aren't the file's stream (as in a free-format MP3, whose
    frames this reader doesn't parse, so it finds only stray bytes that look like frames), and the longer of the frames
    and soundfile's reading counts.
    """
    start: Final = _id3_end(content, 0)
    tally: Final = _tally(_frames(content, start, _mpeg_frame, _MPEG_SYNC))
    after_tag: Final = len(content) - start
    if after_tag > 0 and 2 * tally.chained_bytes >= after_tag:
        return tally.seconds
    return max(tally.seconds, soundfile_seconds)


def _length(lengths: _Lengths) -> float | None:
    played: Final = _longest(lengths.played)
    return played if played is not None else _longest(lengths.declared)


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
    audio: bool
    declared_ticks: int
    sample_ticks: int


def _mp4_track(content: bytes, trak: _Box) -> _Mp4Track | None:
    header: Final = _descendant(content, trak, b"tkhd")
    if header is None or header.end - header.start < 24:
        return None
    timescale, declared = _clock(content, _descendant(content, trak, b"mdia", b"mdhd"))
    handler: Final = _descendant(content, trak, b"mdia", b"hdlr")
    stts: Final = _descendant(content, trak, b"mdia", b"minf", b"stbl", b"stts")
    return _Mp4Track(
        track_id=_uint(content, header.start + (20 if _wide(content, header) else 12), 4),
        timescale=timescale,
        audio=handler is not None and content[handler.start + 8 : handler.start + 12] == b"soun",
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


@dataclass(frozen=True, slots=True)
class _Timeline:
    ticks: int
    first_start: int
    last_end: int


_MAX_FRAGMENTS: Final = 1_000_000


def _fragment_timelines(
    content: bytes, track_ids: frozenset[int], trex_ticks: Mapping[int, int]
) -> Mapping[int, _Timeline] | None:
    """Per track, its fragments' summed sample durations and the span they cover; None past 1,000,000 fragments."""
    trafs: Final = _trafs(content)
    timelines: Final[dict[int, _Timeline]] = {}  # mutable-ok: running per-track totals keep the fragments out of memory
    for traf in islice(trafs, _MAX_FRAGMENTS):
        fragment = _fragment(content, traf, trex_ticks)
        if fragment is None or fragment.track_id not in track_ids:
            continue
        seen = timelines.get(fragment.track_id)
        end = fragment.start_ticks + fragment.ticks
        timelines[fragment.track_id] = (
            _Timeline(fragment.ticks, fragment.start_ticks, end)
            if seen is None
            else _Timeline(
                seen.ticks + fragment.ticks, min(seen.first_start, fragment.start_ticks), max(seen.last_end, end)
            )
        )
    return None if next(trafs, None) is not None else MappingProxyType(timelines)


def _played_ticks(track: _Mp4Track, timeline: _Timeline | None) -> int:
    """
    The samples in the sample table and in the fragments that follow it, or the span from the track's first sample to
    its last fragment's end when that is longer: Chromium's recorder states a short duration for each fragment's last
    sample, and the fragments' start times keep the timeline.
    """
    if timeline is None:
        return track.sample_ticks
    start: Final = 0 if track.sample_ticks else timeline.first_start
    return max(track.sample_ticks + timeline.ticks, timeline.last_end - start)


def _mp4_lengths(content: bytes) -> _Lengths:
    moov: Final = next((box for box in _boxes(content, 0, len(content)) if box.kind == b"moov"), None)
    if moov is None:
        return _Lengths(played=())
    tracks: Final = tuple(
        track for trak in _children(content, moov, b"trak") if (track := _mp4_track(content, trak)) is not None
    )
    audio: Final = tuple(track for track in tracks if track.audio) or tracks
    timelines: Final = _fragment_timelines(
        content, frozenset(track.track_id for track in audio), _trex_default_ticks(content, moov)
    )
    if timelines is None:
        return _Lengths(played=())
    movie_timescale, movie_ticks = _clock(content, _descendant(content, moov, b"mvhd"))
    return _Lengths(
        played=tuple(_seconds(_played_ticks(track, timelines.get(track.track_id)), track.timescale) for track in audio),
        declared=(
            *(_seconds(track.declared_ticks, track.timescale) for track in audio),
            _seconds(movie_ticks, movie_timescale),
            _seconds(_mehd_ticks(content, moov), movie_timescale),
        ),
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
    TRACK_TYPE = 0x83
    CODEC_ID = 0x86
    DEFAULT_DURATION = 0x23E383
    CLUSTER_TIMESTAMP = 0xE7
    SIMPLE_BLOCK = 0xA3
    BLOCK_GROUP = 0xA0
    BLOCK = 0xA1
    BLOCK_DURATION = 0x9B


_ENTERED_MASTERS: Final = frozenset((_Ebml.SEGMENT, _Ebml.INFO, _Ebml.TRACKS, _Ebml.CLUSTER))
_CLUSTER_ID: Final = b"\x1f\x43\xb6\x75"
# No element has ID 0: an EBML variable-size integer never starts with a zero byte
_SKIPPED_ID: Final = 0
_AUDIO_TRACK_TYPE: Final = 2
_DEFAULT_TIMESTAMP_SCALE_NS: Final = 1_000_000
_EBML_MAX_UINT_BYTES: Final = 8
_MAX_HEADER_ELEMENTS: Final = 65536
_MAX_CHILD_ELEMENTS: Final = 256
_MAX_GROUP_FIELDS: Final = 16
_MAX_TRACKS: Final = 64
_MAX_ELEMENTS: Final = 4_000_000


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


def _element_at(content: bytes, offset: int, end: int) -> tuple[_Element, int] | None:
    """The element at `offset` and where the next one starts, or None when the bytes there are not one."""
    id_length: Final = _varint_length(content, offset)
    size_length: Final = _varint_length(content, offset + id_length) if id_length else 0
    if not size_length:
        return None
    element_id: Final = _uint(content, offset, id_length)
    payload: Final = offset + id_length + size_length
    if element_id in _ENTERED_MASTERS:
        return _Element(element_id, payload, end), payload
    size: Final = _varint_value(content, offset + id_length, size_length)
    if size == (1 << 7 * size_length) - 1 or payload + size > end:
        return None
    return _Element(element_id, payload, payload + size), payload + size


def _elements(content: bytes, start: int, end: int, resync: bool) -> Iterator[_Element]:
    """
    Elements in file order. A master a length depends on is entered rather than skipped, so its children follow it
    even when its size is unknown, as in a live recording; its `end` is then its parent's. With `resync`, bytes that
    are not an element are skipped to the next Cluster, as a decoder does, and yielded as one element with ID 0, so a
    budget on elements bounds the skipping too.
    """
    offset = start  # rebind-ok: a cursor over consecutive elements
    while offset < end:
        found = _element_at(content, offset, end)
        if found is not None:
            yield found[0]
            offset = found[1]
            continue
        cluster = content.find(_CLUSTER_ID, offset + 1, end) if resync else -1
        if cluster < 0:
            return
        yield _Element(_SKIPPED_ID, offset, cluster)
        offset = cluster


def _stream_elements(content: bytes) -> Iterator[_Element]:
    """The file's elements, each BlockGroup followed by its children, so one budget bounds them all."""
    for element in _elements(content, 0, len(content), resync=True):
        yield element
        if element.element_id == _Ebml.BLOCK_GROUP:
            yield from _elements(content, element.start, element.end, resync=False)


def _children_of(content: bytes, master: _Element) -> Iterator[_Element]:
    return islice(_elements(content, master.start, master.end, resync=False), _MAX_CHILD_ELEMENTS)


def _element_uint(content: bytes, element: _Element) -> int:
    """An EBML unsigned integer, which is at most 8 bytes; a longer one reads as 0."""
    size: Final = element.end - element.start
    return 0 if size > _EBML_MAX_UINT_BYTES else _uint(content, element.start, size)


def _element_float(content: bytes, element: _Element) -> float:
    unpacked: Final[tuple[float, ...]] = struct.unpack(
        ">d" if element.end - element.start == 8 else ">f", content[element.start : element.end]
    )
    return unpacked[0]


@dataclass(frozen=True, slots=True)
class _MatroskaTrack:
    number: int
    audio: bool
    default_duration_ns: int
    opus: bool


@dataclass(frozen=True, slots=True)
class _MatroskaHeader:
    timestamp_scale_ns: int
    declared_ticks: float
    tracks: Mapping[int, _MatroskaTrack]


def _track_entry(content: bytes, entry: _Element) -> _MatroskaTrack:
    fields: Final = MappingProxyType({child.element_id: child for child in _children_of(content, entry)})
    number: Final = fields.get(_Ebml.TRACK_NUMBER)
    kind: Final = fields.get(_Ebml.TRACK_TYPE)
    default_duration: Final = fields.get(_Ebml.DEFAULT_DURATION)
    codec: Final = fields.get(_Ebml.CODEC_ID)
    return _MatroskaTrack(
        number=0 if number is None else _element_uint(content, number),
        audio=kind is not None and _element_uint(content, kind) == _AUDIO_TRACK_TYPE,
        default_duration_ns=0 if default_duration is None else _element_uint(content, default_duration),
        opus=codec is not None and content[codec.start : codec.end].rstrip(b"\x00") == b"A_OPUS",
    )


def _matroska_header(content: bytes) -> _MatroskaHeader:
    """TimestampScale, Duration and up to 64 tracks, read in one pass that ends at the first Cluster."""
    scale = 0  # rebind-ok: the first TimestampScale, once read
    declared = 0.0  # rebind-ok: the first Duration, once read
    tracks: tuple[_MatroskaTrack, ...] = ()  # rebind-ok: the TrackEntries read so far
    for element in islice(_elements(content, 0, len(content), resync=False), _MAX_HEADER_ELEMENTS):
        if element.element_id == _Ebml.CLUSTER:
            break
        if element.element_id == _Ebml.TIMESTAMP_SCALE and not scale:
            scale = _element_uint(content, element)
        elif element.element_id == _Ebml.DURATION and not declared and element.end - element.start in (4, 8):
            declared = _element_float(content, element)
        elif element.element_id == _Ebml.TRACK_ENTRY and len(tracks) < _MAX_TRACKS:
            tracks = (*tracks, _track_entry(content, element))
    return _MatroskaHeader(
        timestamp_scale_ns=scale or _DEFAULT_TIMESTAMP_SCALE_NS,
        declared_ticks=declared,
        tracks=MappingProxyType({track.number: track for track in tracks}),
    )


@dataclass(frozen=True, slots=True)
class _Block:
    cluster_timestamp: int
    element: _Element
    duration_ticks: int | None


@dataclass(frozen=True, slots=True)
class _Group:
    cluster_timestamp: int
    end: int
    duration_ticks: int | None


def _block_duration(content: bytes, group: _Element) -> int | None:
    """The BlockDuration among a BlockGroup's first 16 children, where muxers write it next to the Block."""
    return next(
        (
            _element_uint(content, child)
            for child in islice(_elements(content, group.start, group.end, resync=False), _MAX_GROUP_FIELDS)
            if child.element_id == _Ebml.BLOCK_DURATION
        ),
        None,
    )


def _blocks(content: bytes, elements: Iterable[_Element]) -> Iterator[_Block]:
    """Every SimpleBlock, and every Block among the children that follow a BlockGroup."""
    cluster_timestamp = 0  # rebind-ok: the Timestamp of the Cluster the next blocks belong to
    group: _Group | None = None  # rebind-ok: the BlockGroup whose children are being read
    for element in elements:
        if group is not None and element.start >= group.end:
            group = None
        if element.element_id == _Ebml.SIMPLE_BLOCK:
            yield _Block(cluster_timestamp, element, None)
        elif element.element_id == _Ebml.BLOCK and group is not None:
            yield _Block(group.cluster_timestamp, element, group.duration_ticks)
        elif element.element_id == _Ebml.CLUSTER_TIMESTAMP:
            cluster_timestamp = _element_uint(content, element)
        elif element.element_id == _Ebml.BLOCK_GROUP and group is None:
            group = _Group(cluster_timestamp, element.end, _block_duration(content, element))


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


@dataclass(frozen=True, slots=True)
class _Span:
    track: int
    start_ns: int
    end_ns: int
    carried_ns: int


def _block_span(content: bytes, header: _MatroskaHeader, block: _Block) -> _Span | None:
    """A block's track, when its audio starts and ends, and how much audio it carries, in nanoseconds."""
    element: Final = block.element
    track_length: Final = _varint_length(content, element.start)
    timing: Final = element.start + track_length
    if not track_length or timing + 3 > element.end:
        return None
    number: Final = _varint_value(content, element.start, track_length)
    laced: Final = content[timing + 2] & 0x06
    frames: Final = content[timing + 3] + 1 if laced and timing + 3 < element.end else 1
    carried: Final = _carried_ns(
        content, header, block, header.tracks.get(number), frames, None if laced else timing + 3
    )
    relative: Final = int.from_bytes(content[timing : timing + 2], "big", signed=True)
    start: Final = (block.cluster_timestamp + relative) * header.timestamp_scale_ns
    return _Span(number, start, start + carried, carried)


def _played_ns(content: bytes, header: _MatroskaHeader) -> tuple[int, ...] | None:
    """
    The audio each counted track plays: the span from its first block's start to its last block's end, or the audio
    its blocks carry when that is longer. The audio tracks count; with none named, every named track counts, and with
    none named, all blocks count as one. Empty when no block counts, and None past 4,000,000 elements.
    """
    audio: Final = frozenset(number for number, track in header.tracks.items() if track.audio)
    counted: Final = audio or frozenset(header.tracks)
    elements: Final = _stream_elements(content)
    spans: Final[dict[int, _Span]] = {}  # mutable-ok: running per-track totals keep the blocks out of memory
    for block in _blocks(content, islice(elements, _MAX_ELEMENTS)):
        span = _block_span(content, header, block)
        if span is None or (counted and span.track not in counted):
            continue
        track = span.track if counted else 0
        seen = spans.get(track, span)
        spans[track] = (
            span
            if seen is span
            else _Span(
                track,
                min(seen.start_ns, span.start_ns),
                max(seen.end_ns, span.end_ns),
                seen.carried_ns + span.carried_ns,
            )
        )
    if next(elements, None) is not None:
        return None
    return tuple(max(span.end_ns - span.start_ns, span.carried_ns) for span in spans.values())


def _matroska_lengths(content: bytes) -> _Lengths:
    """The audio the blocks play, or the Segment's Duration when no block counts at all."""
    header: Final = _matroska_header(content)
    played: Final = _played_ns(content, header)
    if played is None:
        return _Lengths(played=())
    if played:
        return _Lengths(played=tuple(nanoseconds / _NANOSECONDS_PER_SECOND for nanoseconds in played))
    declared: Final = header.declared_ticks * header.timestamp_scale_ns / _NANOSECONDS_PER_SECOND
    return _Lengths(played=(), declared=(declared,))


def _id3_end(content: bytes, offset: int) -> int:
    """Where an ID3v2 tag at `offset` ends, or `offset` when there is none."""
    if content[offset : offset + 3] != b"ID3" or offset + 10 > len(content):
        return offset
    high: Final = (content[offset + 6] & 0x7F) << 21 | (content[offset + 7] & 0x7F) << 14
    size: Final = high | (content[offset + 8] & 0x7F) << 7 | content[offset + 9] & 0x7F
    footer: Final = 10 if content[offset + 5] & 0x10 else 0
    return offset + 10 + size + footer


@dataclass(frozen=True, slots=True)
class _Frame:
    seconds: float
    length: int
    stream: int


def _byte_class(values: Iterable[int]) -> bytes:
    """A regular expression character class matching these byte values."""
    return b"[" + b"".join(re.escape(bytes((value,))) for value in values) + b"]"


_ADTS_SAMPLE_RATES: Final = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350)
# An ADTS header with a known sample rate and a frame length of at least its own 7 bytes: what `_adts_frame` reads
_ADTS_SYNC: Final = re.compile(
    rb"\xff(?=[\xf0\xf1\xf8\xf9]"
    + _byte_class(value for value in range(0x100) if value >> 2 & 0x0F < len(_ADTS_SAMPLE_RATES))
    + b"(?:"
    + _byte_class(value for value in range(0x100) if value & 0x03)
    + rb"..|.[^\x00].|..[\xe0-\xff]))",
    re.DOTALL,
)
_AAC_SAMPLES_PER_BLOCK: Final = 1024


def _adts_frame(content: bytes, offset: int) -> _Frame | None:
    """The complete ADTS frame at `offset`, or None when there is none."""
    if offset + 7 > len(content) or content[offset] != 0xFF or content[offset + 1] & 0xF6 != 0xF0:
        return None
    rate_index: Final = content[offset + 2] >> 2 & 0x0F
    length: Final = (content[offset + 3] & 0x03) << 11 | content[offset + 4] << 3 | content[offset + 5] >> 5
    if rate_index >= len(_ADTS_SAMPLE_RATES) or length < 7 or offset + length > len(content):
        return None
    blocks: Final = (content[offset + 6] & 0x03) + 1
    return _Frame(
        seconds=blocks * _AAC_SAMPLES_PER_BLOCK / _ADTS_SAMPLE_RATES[rate_index],
        length=length,
        stream=(content[offset + 1] & 0x08) << 16 | (content[offset + 2] & 0xFD) << 8 | content[offset + 3] & 0xC0,
    )


# An MPEG audio header with a known version, layer, bitrate and sample rate: what `_mpeg_frame` reads
_MPEG_SYNC: Final = re.compile(
    rb"\xff(?="
    + _byte_class(value for value in range(0xE0, 0x100) if value >> 3 & 0x03 != 1 and value >> 1 & 0x03 != 0)
    + _byte_class(value for value in range(0x100) if value >> 4 not in (0, 15) and value >> 2 & 0x03 != 3)
    + b")"
)
_MPEG_SAMPLE_RATES: Final = MappingProxyType(
    {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}
)
_MPEG1_KBPS: Final = MappingProxyType(
    {
        3: (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
        2: (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
        1: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    }
)
_MPEG2_KBPS: Final = MappingProxyType(
    {
        3: (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
        2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
        1: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    }
)


def _mpeg_frame(content: bytes, offset: int) -> _Frame | None:
    """The complete MPEG audio frame (layer I, II or III) at `offset`, or None when there is none."""
    if offset + 4 > len(content) or content[offset] != 0xFF or content[offset + 1] & 0xE0 != 0xE0:
        return None
    version: Final = content[offset + 1] >> 3 & 0x03
    layer: Final = content[offset + 1] >> 1 & 0x03
    bitrate_index: Final = content[offset + 2] >> 4
    rate_index: Final = content[offset + 2] >> 2 & 0x03
    if version == 1 or layer == 0 or bitrate_index in (0, 15) or rate_index == 3:
        return None
    bits_per_second: Final = (_MPEG1_KBPS if version == 3 else _MPEG2_KBPS)[layer][bitrate_index] * 1000
    rate: Final = _MPEG_SAMPLE_RATES[version][rate_index]
    padding: Final = content[offset + 2] >> 1 & 0x01
    samples: Final = 384 if layer == 3 else 1152 if layer == 2 or version == 3 else 576
    slot_bytes: Final = 4 if layer == 3 else 1
    length: Final = (samples // 8 * bits_per_second // rate // slot_bytes + padding) * slot_bytes
    if offset + length > len(content):
        return None
    return _Frame(seconds=samples / rate, length=length, stream=(content[offset + 1] & 0x1E) << 8 | rate_index)


_FrameReader: TypeAlias = Callable[[bytes, int], _Frame | None]  # mutable-ok: Callable params
_SEARCH_BYTES: Final = 65536
_STREAM_FRAMES: Final = 4


def _starts_a_stream(content: bytes, offset: int, frame_at: _FrameReader) -> bool:
    """Whether four frames of one stream follow each other from `offset`, or fewer run exactly to the end."""
    first: Final = frame_at(content, offset)
    if first is None:
        return False
    position = offset + first.length  # rebind-ok: a cursor over the following frames
    for _ in range(_STREAM_FRAMES - 1):
        if position == len(content):
            return True
        frame = frame_at(content, position)
        if frame is None or frame.stream != first.stream:
            return False
        position += frame.length
    return True


def _first_frame(content: bytes, start: int, frame_at: _FrameReader, sync: re.Pattern[bytes]) -> int | None:
    """
    Where the first frame that starts a stream of these frames lies within the first 64 KB: what tells such a stream
    from other bytes, as a decoder probes it. Its frames are then counted from the stream's start.
    """
    for found in sync.finditer(content, start, min(len(content), start + _SEARCH_BYTES)):
        if _starts_a_stream(content, found.start(), frame_at):
            return found.start()
    return None


@dataclass(frozen=True, slots=True)
class _Step:
    frame: _Frame
    follows: bool


def _frames(content: bytes, start: int, frame_at: _FrameReader, sync: re.Pattern[bytes]) -> Iterator[_Step]:
    """
    Each complete frame of a stream, past stray bytes, as a decoder reads it, and whether it starts where the frame
    before it ended. `sync` finds only a valid
    frame header, so the search past stray bytes lands on a frame, or within a frame's length of the end. An ID3 tag
    within the stream is stray bytes too, as to a decoder that reads no such tags, so frames inside one still count.
    """
    offset = start  # rebind-ok: a cursor over the stream
    follows = False  # rebind-ok: whether the frame at the cursor would start where the one before it ended
    while offset < len(content):
        frame = frame_at(content, offset)
        if frame is not None:
            yield _Step(frame, follows)
            offset += frame.length
            follows = True
            continue
        found = sync.search(content, offset + 1)
        if found is None:
            return
        offset = found.start()
        follows = False


@dataclass(frozen=True, slots=True)
class _Tally:
    seconds: float
    chained_bytes: int


def _tally(steps: Iterable[_Step]) -> _Tally:
    """The audio the frames carry, and the bytes of those that start where the frame before them ended, in one pass."""
    seconds = 0.0  # rebind-ok: running total of the frames' audio
    chained_bytes = 0  # rebind-ok: running total of the bytes of frames that follow another
    for step in steps:
        seconds += step.frame.seconds
        chained_bytes += step.frame.length if step.follows else 0
    return _Tally(seconds, chained_bytes)


def _frame_stream_seconds(content: bytes, start: int) -> float | None:
    """The longer of the ADTS frames and the MPEG audio frames from `start` on, or None when there are none."""
    adts: Final = _tally(_frames(content, start, _adts_frame, _ADTS_SYNC))
    mpeg: Final = _tally(_frames(content, start, _mpeg_frame, _MPEG_SYNC))
    return _longest((adts.seconds, mpeg.seconds))


def _streamed_wav_seconds(content: bytes) -> float | None:
    """
    The length of a WAV whose data size was never written (0 or all ones) or runs past the end of the file, as a
    recorder that streams its output leaves it: the data runs to the end. Any other WAV is soundfile's to read.
    """
    offset = 12  # rebind-ok: a cursor over the RIFF chunks
    byte_rate = 0  # rebind-ok: the format chunk's average bytes per second, once read
    while offset + 8 <= len(content):
        kind = content[offset : offset + 4]
        size = int.from_bytes(content[offset + 4 : offset + 8], "little")
        if kind == b"fmt " and size >= 16:
            byte_rate = int.from_bytes(content[offset + 16 : offset + 20], "little")
        if kind == b"data":
            available = len(content) - offset - 8
            finalized = size not in (0, 0xFFFFFFFF) and size <= available
            return None if finalized or byte_rate <= 0 else _longest((available / byte_rate,))
        offset += 8 + size + size % 2
    return None
