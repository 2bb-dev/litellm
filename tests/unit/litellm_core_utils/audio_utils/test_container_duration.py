import itertools
import struct
from pathlib import Path
from typing import Final

import pytest

from litellm.litellm_core_utils.audio_utils import container_duration
from litellm.litellm_core_utils.audio_utils.container_duration import (
    _ADTS_SYNC,
    _MPEG_SYNC,
    _adts_frame,
    _mpeg_frame,
    container_duration_seconds,
    mp3_duration_seconds,
)

RECORDINGS: Final = Path(__file__).resolve().parent / "recordings"
# The tone-1.5s recordings hold 1.5 s of audio. AAC adds 1024 samples of priming and pads its last frame (64 ms each
# at 16 kHz); Opus pads its last 20 ms frame; MP3 adds an Info frame and pads its last 36 ms frame.
ENCODED_SECONDS: Final = 1.5
AAC_FRAME_SECONDS: Final = 1024 / 16000


def _recording(name: str) -> bytes:
    return (RECORDINGS / name).read_bytes()


@pytest.mark.parametrize(
    "name",
    (
        "tone-1.5s.m4a",
        "tone-1.5s.mp4",
        "tone-1.5s.mov",
        "tone-1.5s-fragmented.mp4",
        "tone-1.5s-fragmented-moov.mp4",
        "tone-1.5s.aac",
        "tone-1.5s.webm",
        "tone-1.5s-live.webm",
        "tone-1.5s-segment.webm",
        "tone-1.5s.mkv",
        "tone-1.5s.mp3",
        "tone-1.5s-video-3s.mp4",
        "tone-1.5s-video-3s.webm",
    ),
)
def test_a_recording_reads_its_audio_and_at_most_its_encoder_padding_more(name: str):
    seconds: Final = container_duration_seconds(_recording(name))

    assert seconds is not None
    assert ENCODED_SECONDS <= seconds <= ENCODED_SECONDS + 2 * AAC_FRAME_SECONDS


def test_a_browser_webm_without_sizes_or_duration_reads_to_the_end_of_its_last_block():
    # Chromium's MediaRecorder: unknown-size Segment and Clusters, no Duration, ten 60 ms Opus packets whose
    # timestamps start late, the last at 1.307 s
    assert container_duration_seconds(_recording("chromium-recorder.webm")) == pytest.approx(1.367)


def test_a_browser_mp4_reads_the_span_its_fragments_cover():
    # Chromium's MediaRecorder: fragmented MP4 with empty sample tables. Each fragment states 21 ms for its last 60 ms
    # packet, so the samples sum to 0.924 s, while the fragments' start times run to 1.041 s
    assert container_duration_seconds(_recording("chromium-recorder.mp4")) == pytest.approx(1.0409167, abs=1e-6)


def _patched(content: bytes, box: bytes, offset_in_payload: int, value: bytes) -> bytes:
    at: Final = content.index(box) + 4 + offset_in_payload
    return content[:at] + value + content[at + len(value) :]


def _with_header_durations(content: bytes, duration: bytes) -> bytes:
    return _patched(_patched(content, b"mvhd", 16, duration), b"mdhd", 16, duration)


def test_an_mp4_reads_its_samples_whatever_its_headers_declare():
    m4a: Final = _recording("tone-1.5s.m4a")

    for declared in (bytes(4), b"\xff" * 4, b"\x7f\xff\xff\xfe"):
        assert container_duration_seconds(_with_header_durations(m4a, declared)) == container_duration_seconds(m4a)


def test_an_mp4_without_samples_reads_the_duration_its_headers_declare():
    m4a: Final = _recording("tone-1.5s.m4a")
    without_samples: Final = _patched(m4a, b"stts", 4, bytes(4))

    assert container_duration_seconds(without_samples) == pytest.approx(
        container_duration_seconds(m4a), abs=AAC_FRAME_SECONDS
    )
    assert container_duration_seconds(_with_header_durations(without_samples, bytes(4))) is None


def _id3_holding(payload: bytes) -> bytes:
    """An ID3v2 tag whose payload (such as embedded art) holds bytes that look like a frame."""
    size: Final = bytes((len(payload) >> 21 & 0x7F, len(payload) >> 14 & 0x7F, len(payload) >> 7 & 0x7F))
    return b"ID3\x04\x00\x00" + size + bytes((len(payload) & 0x7F,)) + payload


def _split_frames(content: bytes, start: int, frame_at) -> tuple[bytes, ...]:
    """The frames from `start` on, each as long as its header states, up to the first byte that starts none."""
    frame: Final = frame_at(content, start)
    if frame is None:
        return ()
    return (content[start : start + frame.length], *_split_frames(content, start + frame.length, frame_at))


def _adts_frames(aac: bytes) -> tuple[bytes, ...]:
    return _split_frames(aac, 0, _adts_frame)


def test_adts_reads_on_past_a_leading_id3_tag_and_stray_bytes_and_drops_an_incomplete_last_frame():
    aac: Final = _recording("tone-1.5s.aac")
    whole: Final = container_duration_seconds(aac)
    id3: Final = b"ID3\x04\x00\x00\x00\x00\x00\x05" + bytes(5)
    first_frame: Final = len(_adts_frames(aac)[0])

    assert whole is not None
    assert container_duration_seconds(id3 + aac + b"not a frame") == whole
    assert container_duration_seconds(aac[:first_frame] + b"\x00" + aac[first_frame:]) == whole
    assert container_duration_seconds(aac[:-1]) == pytest.approx(whole - AAC_FRAME_SECONDS)


def test_a_frame_inside_an_id3_tag_within_the_stream_counts_as_a_decoder_that_reads_no_tags_plays_it():
    aac: Final = _recording("tone-1.5s.aac")
    whole: Final = container_duration_seconds(aac)

    assert whole is not None
    assert container_duration_seconds(aac + _id3_holding(_adts_frames(aac)[0]) + aac) == pytest.approx(
        2 * whole + AAC_FRAME_SECONDS
    )


def test_adts_frames_count_past_any_number_of_stray_bytes():
    frames: Final = _adts_frames(_recording("tone-1.5s.aac")) * 200
    split: Final = b"".join(frames[:4]) + b"".join(frame + b"\x00" for frame in frames[4:])

    assert container_duration_seconds(split) == pytest.approx(len(frames) * AAC_FRAME_SECONDS)


def test_a_stream_of_frames_needs_four_of_them_in_a_row():
    tiny_adts_frame: Final = b"\xff\xf1\x60\x40\x00\xff\xfc"

    assert container_duration_seconds(bytes(1000) + tiny_adts_frame * 2 + bytes(1000)) is None
    assert container_duration_seconds(bytes(1000) + tiny_adts_frame * 4 + bytes(1000)) == pytest.approx(
        4 * 1024 / 16000
    )


def _mp3_frames() -> tuple[bytes, tuple[bytes, ...]]:
    """The recording's ID3 tag, and its frames: an Info frame, then the audio."""
    mp3: Final = _recording("tone-1.5s.mp3")
    first: Final = mp3.index(b"\xff\xf3")
    return mp3[:first], _split_frames(mp3, first, _mpeg_frame)


MP3_FRAME_SECONDS: Final = 576 / 16000


def test_mp3_frames_count_past_any_number_of_stray_bytes_without_four_in_a_row():
    tag, frames = _mp3_frames()
    many: Final = (frames[0], *frames[1:] * 120)
    split: Final = tag + b"".join(frame + b"\x00" for frame in many)

    assert mp3_duration_seconds(tag + b"".join(frames)) == pytest.approx(len(frames) * MP3_FRAME_SECONDS)
    assert mp3_duration_seconds(split) == pytest.approx(len(many) * MP3_FRAME_SECONDS)
    assert mp3_duration_seconds(b"ID3" + bytes(7) + bytes(200)) is None


def test_a_file_holding_both_kinds_of_frames_reads_the_longer_whichever_starts_a_stream():
    adts: Final = _adts_frames(_recording("tone-1.5s.aac"))
    _, mpeg = _mp3_frames()
    # A stray byte after every third AAC frame leaves no four in a row, so only the MPEG frames start a stream
    split_adts: Final = b"".join(b"".join(adts[group : group + 3]) + b"\x00" for group in range(0, len(adts), 3))

    assert container_duration_seconds(b"".join(mpeg[1:5]) + split_adts) == pytest.approx(len(adts) * AAC_FRAME_SECONDS)
    assert mp3_duration_seconds(b"".join(adts * 3) + b"".join(mpeg)) == pytest.approx(3 * len(adts) * AAC_FRAME_SECONDS)


@pytest.mark.parametrize(
    ("sync", "frame_at", "headers"),
    (
        (_MPEG_SYNC, _mpeg_frame, tuple(b"\xff" + bytes(pair) for pair in itertools.product(range(256), repeat=2))),
        (
            _ADTS_SYNC,
            _adts_frame,
            tuple(
                b"\xff" + bytes(fields)
                for fields in itertools.product(
                    (0x00, 0xE1, 0xF0, 0xF1, 0xF2, 0xF6, 0xF8, 0xF9),
                    range(256),
                    (0x00, 0x01, 0x02, 0x03, 0xFC),
                    (0x00, 0x01, 0xFF),
                    (0x00, 0xDF, 0xE0, 0xFF),
                )
            ),
        ),
    ),
)
def test_a_sync_pattern_finds_exactly_the_headers_its_frame_reader_reads(sync, frame_at, headers):
    room: Final = bytes(8192)
    mismatched: Final = tuple(
        header for header in headers if bool(sync.match(header + room)) != (frame_at(header + room, 0) is not None)
    )

    assert mismatched == ()


def test_mp3_reads_its_frames_whatever_its_info_header_says_and_after_padding():
    mp3: Final = _recording("tone-1.5s.mp3")
    info_frames: Final = mp3.index(b"Info") + 8
    understated: Final = mp3[:info_frames] + (1).to_bytes(4, "big") + mp3[info_frames + 4 :]

    assert container_duration_seconds(understated) == container_duration_seconds(mp3)
    assert container_duration_seconds(bytes(300) + mp3) == container_duration_seconds(mp3)


def _wav(data_size: int, frames: int) -> bytes:
    fmt: Final = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    return (
        b"RIFF"
        + bytes(4)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + fmt
        + b"data"
        + struct.pack("<I", data_size)
        + bytes(2 * frames)
    )


def test_a_wav_whose_sizes_were_never_written_reads_its_data_to_the_end():
    assert container_duration_seconds(_wav(0, 8000)) == pytest.approx(0.5)
    assert container_duration_seconds(_wav(0xFFFFFFFF, 8000)) == pytest.approx(0.5)
    assert container_duration_seconds(_wav(64000, 8000)) == pytest.approx(0.5)
    assert container_duration_seconds(_wav(16000, 8000)) is None


@pytest.mark.parametrize(
    "content",
    (
        b"",
        b"not audio at all",
        b"\x00\x00\x00\x18ftypM4A \x00\x00\x02\x00isomM4A ",
        b"\x1a\x45\xdf\xa3" + bytes(16),
        b"OggS" + bytes(60),
    ),
)
def test_content_without_a_length_reads_none(content: bytes):
    assert container_duration_seconds(content) is None


def _element(element_id: int, payload: bytes) -> bytes:
    size: Final = (1 << 56 | len(payload)).to_bytes(8, "big")
    return element_id.to_bytes((element_id.bit_length() + 7) // 8, "big") + size + payload


UNKNOWN_SIZE: Final = b"\x01\xff\xff\xff\xff\xff\xff\xff"
EBML_HEADER: Final = _element(0x1A45DFA3, _element(0x4282, b"webm"))
SEGMENT: Final = b"\x18\x53\x80\x67" + UNKNOWN_SIZE


def _track(number: int, codec: bytes = b"A_OPUS", kind: int | None = None, default_duration_ns: int = 0) -> bytes:
    fields: Final = _element(0xD7, bytes((number,))) + _element(0x86, codec)
    typed: Final = _element(0x83, bytes((kind,))) if kind is not None else b""
    default: Final = _element(0x23E383, default_duration_ns.to_bytes(4, "big")) if default_duration_ns else b""
    return _element(0xAE, fields + typed + default)


def _block(timestamp: int, frames: bytes, flags: int = 0x80, track: int = 1) -> bytes:
    return _element(0xA3, bytes((0x80 | track,)) + timestamp.to_bytes(2, "big", signed=True) + bytes((flags,)) + frames)


def _cluster(timestamp: bytes, *blocks: bytes) -> bytes:
    return b"\x1f\x43\xb6\x75" + UNKNOWN_SIZE + _element(0xE7, timestamp) + b"".join(blocks)


def _live_webm(*blocks: bytes, tracks: bytes = _track(1), info: bytes = b"", cluster_timestamp: int = 0) -> bytes:
    return (
        EBML_HEADER
        + SEGMENT
        + info
        + _element(0x1654AE6B, tracks)
        + _cluster(cluster_timestamp.to_bytes(2, "big"), *blocks)
    )


OPUS_20_MS: Final = b"\x08\xaa"
OPUS_60_MS: Final = b"\x18\xaa"


def test_opus_packets_count_even_when_their_timestamps_do_not_move():
    blocks: Final = tuple(_block(0, OPUS_20_MS) for _ in range(50))

    assert container_duration_seconds(_live_webm(*blocks)) == pytest.approx(1.0)


def test_the_latest_block_ends_after_the_packet_it_carries():
    assert container_duration_seconds(_live_webm(_block(0, OPUS_20_MS), _block(1000, OPUS_60_MS))) == pytest.approx(
        1.06
    )


def test_a_track_is_measured_from_its_first_block_in_timestamp_scale_units():
    info: Final = _element(0x1549A966, _element(0x2AD7B1, (10_000_000).to_bytes(4, "big")))
    segment: Final = _live_webm(_block(0, OPUS_20_MS), _block(50, OPUS_20_MS), info=info, cluster_timestamp=18_000)

    assert container_duration_seconds(segment) == pytest.approx(0.52)


def test_an_integer_longer_than_eight_bytes_reads_as_zero():
    oversized: Final = b"\x01" + bytes(4096)
    info: Final = _element(0x1549A966, _element(0x2AD7B1, oversized))
    tracks: Final = _element(0x1654AE6B, _track(1))
    clusters: Final = _cluster(bytes(2), _block(0, OPUS_20_MS)) + _cluster(oversized, _block(1000, OPUS_20_MS))

    assert container_duration_seconds(EBML_HEADER + SEGMENT + info + tracks + clusters) == pytest.approx(1.02)


GROUPED_OPUS_20_MS: Final = _element(0xA1, b"\x81" + bytes(2) + b"\x80" + OPUS_20_MS)
VOID: Final = _element(0xEC, b"")


def test_a_block_group_lasts_its_block_duration():
    group: Final = _element(0xA0, GROUPED_OPUS_20_MS + _element(0x9B, (2500).to_bytes(2, "big")))

    assert container_duration_seconds(_live_webm(group)) == pytest.approx(2.5)


def test_consecutive_block_groups_each_last_their_own_block_duration():
    first: Final = _element(0xA0, GROUPED_OPUS_20_MS + _element(0x9B, (500).to_bytes(2, "big")))
    second_block: Final = _element(0xA1, b"\x81" + (1000).to_bytes(2, "big") + b"\x80" + OPUS_20_MS)
    second: Final = _element(0xA0, second_block + _element(0x9B, (2500).to_bytes(2, "big")))

    assert container_duration_seconds(_live_webm(first, second)) == pytest.approx(3.5)


def test_a_block_counts_wherever_it_lies_in_its_group_and_a_block_duration_among_the_first_16_fields():
    block_duration: Final = _element(0x9B, (2500).to_bytes(2, "big"))

    assert container_duration_seconds(_live_webm(_element(0xA0, VOID * 50 + GROUPED_OPUS_20_MS))) == pytest.approx(0.02)
    assert container_duration_seconds(
        _live_webm(_element(0xA0, VOID * 14 + GROUPED_OPUS_20_MS + block_duration))
    ) == pytest.approx(2.5)
    assert container_duration_seconds(
        _live_webm(_element(0xA0, VOID * 15 + GROUPED_OPUS_20_MS + block_duration))
    ) == pytest.approx(0.02)


def test_laced_frames_last_the_track_default_duration_each():
    xiph_laced: Final = _block(0, b"\x04" + bytes((10, 10, 10, 10)) + bytes(50), flags=0x82)

    assert container_duration_seconds(
        _live_webm(xiph_laced, tracks=_track(1, default_duration_ns=20_000_000))
    ) == pytest.approx(0.1)


def test_only_audio_tracks_count_and_each_one_on_its_own():
    tracks: Final = _track(1, kind=2) + _track(2, kind=2) + _track(3, codec=b"V_VP8", kind=1)
    audio: Final = tuple(_block(0, OPUS_20_MS, track=1 + index % 2) for index in range(100))
    video: Final = _block(5000, b"\x00" * 10, track=3)

    assert container_duration_seconds(_live_webm(*audio, video, tracks=tracks)) == pytest.approx(1.0)


def test_a_declared_duration_counts_only_when_no_block_does():
    info: Final = _element(0x1549A966, _element(0x4489, struct.pack(">d", 2500.0)))

    assert container_duration_seconds(_live_webm(_block(0, OPUS_20_MS), info=info)) == pytest.approx(0.02)
    assert container_duration_seconds(_live_webm(info=info)) == pytest.approx(2.5)
    # A block whose length nothing states (not Opus, no DefaultDuration) still counts, so the header can't stand in
    assert container_duration_seconds(_live_webm(_block(0, bytes(4)), tracks=_track(1, b"A_AAC"), info=info)) is None


def test_bytes_that_are_not_an_element_are_skipped_to_the_next_cluster():
    first: Final = _live_webm(_block(0, OPUS_20_MS))
    second: Final = _cluster((1000).to_bytes(2, "big"), _block(0, OPUS_20_MS))

    assert container_duration_seconds(first + b"\x00" + second) == pytest.approx(1.02)


def test_any_number_of_stray_stretches_are_skipped_to_the_next_cluster():
    clusters: Final = b"".join(
        b"\x00" + _cluster((20 * index).to_bytes(4, "big"), _block(0, OPUS_20_MS)) for index in range(1, 5000)
    )

    assert container_duration_seconds(_live_webm(_block(0, OPUS_20_MS)) + clusters) == pytest.approx(100.0)


def test_skipped_stretches_and_block_group_fields_count_toward_the_element_budget(monkeypatch):
    monkeypatch.setattr(container_duration, "_MAX_ELEMENTS", 40)
    stray_clusters: Final = b"\x00" + b"\x1f\x43\xb6\x75\x00"

    assert container_duration_seconds(_live_webm(_block(0, OPUS_20_MS)) + stray_clusters * 10) == pytest.approx(0.02)
    assert container_duration_seconds(_live_webm(_block(0, OPUS_20_MS)) + stray_clusters * 50) is None
    assert container_duration_seconds(_live_webm(_element(0xA0, VOID * 20 + GROUPED_OPUS_20_MS))) == pytest.approx(0.02)
    assert container_duration_seconds(_live_webm(_element(0xA0, VOID * 50 + GROUPED_OPUS_20_MS))) is None


def test_a_truncated_block_ends_the_walk_with_the_blocks_before_it():
    whole: Final = _live_webm(_block(0, OPUS_20_MS), _block(500, OPUS_20_MS))

    assert container_duration_seconds(whole[:-1]) == pytest.approx(0.02)
