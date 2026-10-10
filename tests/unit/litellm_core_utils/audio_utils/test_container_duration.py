import struct
from pathlib import Path
from typing import Final

import pytest

from litellm.litellm_core_utils.audio_utils.container_duration import container_duration_seconds

RECORDINGS: Final = Path(__file__).resolve().parent / "recordings"
# The tone-1.5s recordings hold 1.5 s of audio. AAC adds 1024 samples of priming and pads its last frame (64 ms each
# at 16 kHz); Opus pads its last 20 ms frame.
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
        "tone-1.5s.aac",
        "tone-1.5s.webm",
        "tone-1.5s-live.webm",
        "tone-1.5s.mkv",
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


def _without_header_durations(content: bytes) -> bytes:
    return _patched(_patched(content, b"mvhd", 16, bytes(4)), b"mdhd", 16, bytes(4))


def test_an_mp4_whose_headers_state_no_duration_reads_its_sample_table():
    m4a: Final = _recording("tone-1.5s.m4a")

    assert container_duration_seconds(_without_header_durations(m4a)) == container_duration_seconds(m4a)


def test_an_mp4_whose_sample_table_is_short_reads_the_longer_duration_its_headers_state():
    m4a: Final = _recording("tone-1.5s.m4a")
    without_samples: Final = _patched(m4a, b"stts", 4, bytes(4))

    assert container_duration_seconds(without_samples) == pytest.approx(
        container_duration_seconds(m4a), abs=AAC_FRAME_SECONDS
    )
    assert container_duration_seconds(_without_header_durations(without_samples)) is None


def test_an_mp4_unknown_duration_is_not_read_as_a_length():
    m4a: Final = _recording("tone-1.5s.m4a")
    unknown: Final = _patched(_patched(m4a, b"mvhd", 16, b"\xff" * 4), b"mdhd", 16, b"\xff" * 4)

    assert container_duration_seconds(unknown) == container_duration_seconds(m4a)


def test_adts_counts_complete_frames_after_an_id3_tag_and_before_trailing_bytes():
    aac: Final = _recording("tone-1.5s.aac")
    id3: Final = b"ID3\x04\x00\x00\x00\x00\x00\x05" + bytes(5)

    assert container_duration_seconds(id3 + aac + b"not a frame") == container_duration_seconds(aac)
    assert container_duration_seconds(aac[:-1]) == pytest.approx(container_duration_seconds(aac) - AAC_FRAME_SECONDS)


@pytest.mark.parametrize(
    "content",
    (
        b"",
        b"not audio at all",
        b"\x00\x00\x00\x18ftypM4A \x00\x00\x02\x00isomM4A ",
        b"\x1a\x45\xdf\xa3" + bytes(16),
    ),
)
def test_content_without_a_length_reads_none(content: bytes):
    assert container_duration_seconds(content) is None


def _element(element_id: int, payload: bytes) -> bytes:
    size: Final = (1 << 56 | len(payload)).to_bytes(8, "big")
    return element_id.to_bytes((element_id.bit_length() + 7) // 8, "big") + size + payload


UNKNOWN_SIZE: Final = b"\x01\xff\xff\xff\xff\xff\xff\xff"
EBML_HEADER: Final = _element(0x1A45DFA3, _element(0x4282, b"webm"))


def _opus_track(default_duration_ns: int = 0) -> bytes:
    fields: Final = _element(0xD7, b"\x01") + _element(0x86, b"A_OPUS")
    default: Final = _element(0x23E383, default_duration_ns.to_bytes(4, "big")) if default_duration_ns else b""
    return _element(0x1654AE6B, _element(0xAE, fields + default))


def _block(timestamp: int, frames: bytes, flags: int = 0x80) -> bytes:
    return b"\x81" + timestamp.to_bytes(2, "big", signed=True) + bytes((flags,)) + frames


def _live_webm(*blocks: bytes, track: bytes = _opus_track(), info: bytes = b"", cluster_timestamp: int = 0) -> bytes:
    cluster: Final = b"\x1f\x43\xb6\x75" + UNKNOWN_SIZE + _element(0xE7, cluster_timestamp.to_bytes(2, "big"))
    return EBML_HEADER + b"\x18\x53\x80\x67" + UNKNOWN_SIZE + info + track + cluster + b"".join(blocks)


OPUS_20_MS: Final = b"\x08\xaa"
OPUS_60_MS: Final = b"\x18\xaa"


def test_opus_packets_count_even_when_their_timestamps_do_not_move():
    blocks: Final = tuple(_element(0xA3, _block(0, OPUS_20_MS)) for _ in range(50))

    assert container_duration_seconds(_live_webm(*blocks)) == pytest.approx(1.0)


def test_the_latest_block_ends_after_the_packet_it_carries():
    blocks: Final = (_element(0xA3, _block(0, OPUS_20_MS)), _element(0xA3, _block(1000, OPUS_60_MS)))

    assert container_duration_seconds(_live_webm(*blocks)) == pytest.approx(1.06)


def test_a_cluster_timestamp_and_timestamp_scale_place_the_blocks():
    info: Final = _element(0x1549A966, _element(0x2AD7B1, (10_000_000).to_bytes(4, "big")))
    blocks: Final = (_element(0xA3, _block(50, OPUS_20_MS)),)

    assert container_duration_seconds(_live_webm(*blocks, info=info, cluster_timestamp=100)) == pytest.approx(1.52)


def test_a_block_group_lasts_its_block_duration():
    group: Final = _element(0xA0, _element(0xA1, _block(0, OPUS_20_MS)) + _element(0x9B, (2500).to_bytes(2, "big")))

    assert container_duration_seconds(_live_webm(group)) == pytest.approx(2.5)


def test_laced_frames_last_the_track_default_duration_each():
    xiph_laced: Final = _block(0, b"\x04" + bytes((10, 10, 10, 10)) + bytes(50), flags=0x82)

    assert container_duration_seconds(
        _live_webm(_element(0xA3, xiph_laced), track=_opus_track(default_duration_ns=20_000_000))
    ) == pytest.approx(0.1)


def test_a_declared_duration_longer_than_the_blocks_wins():
    info: Final = _element(0x1549A966, _element(0x4489, struct.pack(">d", 2500.0)))

    assert container_duration_seconds(_live_webm(_element(0xA3, _block(0, OPUS_20_MS)), info=info)) == pytest.approx(
        2.5
    )


def test_a_truncated_block_ends_the_walk_with_the_blocks_before_it():
    whole: Final = _live_webm(_element(0xA3, _block(0, OPUS_20_MS)), _element(0xA3, _block(500, OPUS_20_MS)))

    assert container_duration_seconds(whole[:-1]) == pytest.approx(0.02)
