"""
Test the audio utils functionality in litellm_core_utils/audio_utils/utils.py
"""

import io
import math
import os
import tempfile
import wave
from pathlib import Path
from typing import Final
from unittest.mock import mock_open, patch

import pytest

from litellm.litellm_core_utils.audio_utils.utils import (
    ProcessedAudioFile,
    calculate_request_duration,
    get_audio_file_content_hash,
    get_audio_file_for_health_check,
    get_audio_file_name,
    longest_audio_seconds,
    priced_by_audio_length,
    process_audio_file,
)

RECORDINGS: Final = Path(__file__).resolve().parent / "audio_utils" / "recordings"


class TestProcessAudioFile:
    """Test the process_audio_file function with various input types"""

    def test_process_bytes_input(self):
        """Test processing raw bytes input"""
        audio_data = b"fake audio data"
        result = process_audio_file(audio_data)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == audio_data
        assert result.filename == "audio.wav"
        assert result.content_type == "audio/wav"

    def test_process_bytearray_input(self):
        """Test processing bytearray input"""
        audio_data = bytearray(b"fake audio data")
        result = process_audio_file(audio_data)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == bytes(audio_data)
        assert result.filename == "audio.wav"
        assert result.content_type == "audio/wav"

    def test_process_pathlib_input(self):
        """pathlib.Path is a Python-level type HTTP form values can't fabricate."""
        from pathlib import Path

        test_content = b"test audio content"

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            result = process_audio_file(Path(temp_file_path))

            assert isinstance(result, ProcessedAudioFile)
            assert result.file_content == test_content
            assert result.filename == os.path.basename(temp_file_path)
            assert result.content_type == "audio/mpeg"
        finally:
            os.unlink(temp_file_path)

    def test_process_bare_str_path_rejected(self):
        """Bare str paths are rejected — when this runs in a proxy request
        handler the value is attacker-controlled, and opening it as a path
        is an arbitrary local file read."""
        with pytest.raises(ValueError, match="does not accept bare str inputs"):
            process_audio_file("/etc/passwd")

    def test_process_tuple_input_with_bytes(self):
        """Test processing tuple input with bytes content"""
        filename = "test.wav"
        audio_data = b"fake audio data"
        audio_tuple = (filename, audio_data)

        result = process_audio_file(audio_tuple)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == audio_data
        assert result.filename == filename
        assert result.content_type == "audio/wav"

    def test_process_tuple_input_with_pathlib_content(self):
        """Tuple input with pathlib.Path content is allowed; bare str content is not."""
        from pathlib import Path

        test_content = b"test audio content"

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            filename = "custom_name.flac"
            audio_tuple = (filename, Path(temp_file_path))

            result = process_audio_file(audio_tuple)

            assert isinstance(result, ProcessedAudioFile)
            assert result.file_content == test_content
            assert result.filename == filename
            assert result.content_type == "audio/flac"
        finally:
            os.unlink(temp_file_path)

    def test_process_file_like_object(self):
        """Test processing file-like object input"""
        test_content = b"test audio content"
        file_obj = io.BytesIO(test_content)
        file_obj.name = "test_audio.ogg"

        result = process_audio_file(file_obj)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == test_content
        assert result.filename == "test_audio.ogg"
        assert result.content_type == "audio/ogg"

        # Verify file pointer was reset
        assert file_obj.tell() == 0

    def test_process_file_like_object_without_name(self):
        """Test processing file-like object without name attribute"""
        test_content = b"test audio content"
        file_obj = io.BytesIO(test_content)

        result = process_audio_file(file_obj)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == test_content
        assert result.filename == "audio.wav"
        assert result.content_type == "audio/wav"

    def test_process_tuple_with_file_like_object(self):
        """Test processing tuple with file-like object as content"""
        test_content = b"test audio content"
        file_obj = io.BytesIO(test_content)

        filename = "custom.mp3"
        audio_tuple = (filename, file_obj)

        result = process_audio_file(audio_tuple)

        assert isinstance(result, ProcessedAudioFile)
        assert result.file_content == test_content
        assert result.filename == filename
        assert result.content_type == "audio/mpeg"

        # Verify file pointer was reset
        assert file_obj.tell() == 0

    def test_mime_type_detection_various_extensions(self):
        """Test MIME type detection for various audio file extensions"""
        test_cases = [
            ("test.wav", "audio/wav"),
            ("test.mp3", "audio/mpeg"),
            ("test.flac", "audio/flac"),
            ("test.ogg", "audio/ogg"),
            ("test.aac", "audio/aac"),
            ("test.m4a", "audio/x-m4a"),
        ]

        for filename, expected_mime_type in test_cases:
            audio_tuple = (filename, b"fake content")
            result = process_audio_file(audio_tuple)
            assert result.content_type == expected_mime_type, f"Failed for {filename}"

    def test_mime_type_fallback_for_unknown_extension(self):
        """Test MIME type fallback for unknown file extensions"""
        audio_tuple = ("test.unknown", b"fake content")
        result = process_audio_file(audio_tuple)

        assert result.content_type == "audio/wav"  # Should fallback to default

    def test_process_pathlike_object(self):
        """Test processing os.PathLike object"""
        test_content = b"test audio content"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            # Convert to pathlib.Path
            from pathlib import Path

            path_obj = Path(temp_file_path)

            result = process_audio_file(path_obj)

            assert isinstance(result, ProcessedAudioFile)
            assert result.file_content == test_content
            assert result.filename == os.path.basename(temp_file_path)
            assert result.content_type == "audio/wav"
        finally:
            os.unlink(temp_file_path)

    def test_invalid_input_type(self):
        """Test that invalid input types raise ValueError"""
        with pytest.raises(ValueError, match="Unsupported audio_file type"):
            process_audio_file(123)  # Invalid type

    def test_invalid_tuple_length(self):
        """Test that tuple with less than 2 elements raises ValueError"""
        with pytest.raises(ValueError, match="Tuple must have at least 2 elements"):
            process_audio_file(("only_one_element",))

    def test_invalid_tuple_content_type(self):
        """Test that tuple with unsupported content type raises ValueError"""
        with pytest.raises(ValueError, match="Unsupported content type in tuple"):
            process_audio_file(("filename", 123))  # Invalid content type

    def test_tuple_with_none_filename(self):
        """Test tuple with None filename gets default name"""
        audio_tuple = (None, b"fake content")
        result = process_audio_file(audio_tuple)

        assert result.filename == "audio.wav"
        assert result.content_type == "audio/wav"


class TestCalculateRequestDuration:
    """Test the calculate_request_duration function"""

    @pytest.mark.skipif(
        os.environ.get("SKIP_AUDIO_TESTS") == "true",
        reason="Skipping audio tests - soundfile may not be available",
    )
    def test_bytesio_at_end_position(self):
        """
        Test that calculate_request_duration handles BytesIO with file pointer at end.
        This reproduces and verifies the fix for the OGG file bug where BytesIO
        position was at the end after a previous read(), causing "Format not recognised" error.
        """
        # Create a simple WAV file in memory (44 bytes header + some data)
        # This is a minimal valid WAV file
        wav_header = (
            b"RIFF"
            + (36 + 8).to_bytes(4, "little")  # ChunkSize
            + b"WAVE"
            + b"fmt "
            + (16).to_bytes(4, "little")  # Subchunk1Size
            + (1).to_bytes(2, "little")  # AudioFormat (PCM)
            + (1).to_bytes(2, "little")  # NumChannels
            + (16000).to_bytes(4, "little")  # SampleRate
            + (32000).to_bytes(4, "little")  # ByteRate
            + (2).to_bytes(2, "little")  # BlockAlign
            + (16).to_bytes(2, "little")  # BitsPerSample
            + b"data"
            + (8).to_bytes(4, "little")  # Subchunk2Size
            + b"\x00\x00\x00\x00\x00\x00\x00\x00"  # Sample data
        )

        # Create BytesIO object
        file_obj = io.BytesIO(wav_header)
        file_obj.name = "test_audio.wav"

        # Simulate the bug: something reads from the file first, moving position to end
        _ = file_obj.read()
        assert file_obj.tell() == len(wav_header), "File position should be at end"

        # Call calculate_request_duration - this would fail before the fix
        duration = calculate_request_duration(file_obj)

        # Verify it succeeded (returns a duration, not None)
        assert (
            duration is not None
        ), "Duration should be calculated even when BytesIO is at end"
        assert isinstance(duration, float), "Duration should be a float"
        assert duration > 0, "Duration should be positive"

        # Verify the file position was restored
        assert file_obj.tell() == len(
            wav_header
        ), "File position should be restored to original position"


def _wav(seconds: float) -> bytes:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buffer.getvalue()


def _named(name: str, content: bytes) -> io.BytesIO:
    upload: Final = io.BytesIO(content)
    upload.name = name
    return upload


class TestCalculateRequestDurationOfEveryUploadFormat:
    @pytest.mark.parametrize("name", sorted(path.name for path in RECORDINGS.glob("tone-1.5s*")))
    @pytest.mark.parametrize("wrap", (lambda name, content: content, lambda name, content: (name, content), _named))
    def test_a_recording_of_each_format_reads_its_length(self, name, wrap):
        seconds: Final = calculate_request_duration(wrap(name, (RECORDINGS / name).read_bytes()))

        assert seconds is not None
        assert 1.5 <= seconds <= 1.5 + 2 * 1024 / 16000

    def test_a_proxy_upload_is_read_from_its_start_and_left_where_it_was(self):
        upload: Final = _named("note.webm", (RECORDINGS / "chromium-recorder.webm").read_bytes())
        upload.seek(100)

        assert calculate_request_duration(upload) == pytest.approx(1.367)
        assert upload.tell() == 100

    @pytest.mark.parametrize("content", (_wav(0), b"", b"not audio"))
    def test_audio_without_a_positive_length_reads_none(self, content):
        assert calculate_request_duration(("note.wav", content)) is None

    def test_a_tuple_may_carry_a_bytearray_or_a_path(self, tmp_path):
        m4a: Final = (RECORDINGS / "tone-1.5s.m4a").read_bytes()
        upload: Final = tmp_path / "note.m4a"
        upload.write_bytes(m4a)

        assert calculate_request_duration(("note.m4a", bytearray(m4a))) == calculate_request_duration(m4a)
        assert calculate_request_duration(("note.m4a", upload)) == calculate_request_duration(m4a)

    def test_a_streamed_wav_whose_sizes_were_never_written_reads_its_data(self):
        streamed: Final = _wav(0.5)[:4] + bytes(4) + _wav(0.5)[8:40] + bytes(4) + _wav(0.5)[44:]

        assert calculate_request_duration(("note.wav", streamed)) == pytest.approx(0.5)

    def test_a_length_is_never_more_than_the_bytes_carry_at_100_bits_a_second(self):
        m4a: Final = (RECORDINGS / "tone-1.5s.m4a").read_bytes()
        stts: Final = m4a.index(b"stts") + 4
        hour_long_samples: Final = m4a[: stts + 12] + (16000 * 3600).to_bytes(4, "big") + m4a[stts + 16 :]

        assert calculate_request_duration(hour_long_samples) == pytest.approx(len(m4a) * 8 / 100)

    def test_an_mp3_is_measured_by_its_frames_even_when_its_payload_looks_like_aac(self):
        mp3: Final = (RECORDINGS / "tone-1.5s.mp3").read_bytes()
        # 24 kbps at 16 kHz: every frame is 108 bytes. Four 7-byte ADTS frames go inside the fourth one's payload.
        tiny_adts_frames: Final = b"\xff\xf1\x60\x40\x00\xff\xfc" * 4
        payload: Final = mp3.index(b"\xff\xf3") + 3 * 108 + 40
        disguised: Final = mp3[:payload] + tiny_adts_frames + mp3[payload + len(tiny_adts_frames) :]

        assert calculate_request_duration(disguised) == calculate_request_duration(mp3)

    def test_a_format_soundfile_reads_is_measured_by_soundfile_even_when_its_samples_look_like_frames(self):
        import numpy
        import soundfile

        mp3: Final = (RECORDINGS / "tone-1.5s.mp3").read_bytes()
        four_mp3_frames: Final = mp3[mp3.index(b"\xff\xf3\x38") :][: 4 * 108]
        buffer: Final = io.BytesIO()
        soundfile.write(buffer, numpy.zeros(16000, dtype="int16"), 16000, format="HTK")
        # HTK opens with a 12-byte header and no magic, so its samples are all the container readers see
        htk: Final = buffer.getvalue()
        disguised: Final = htk[:12] + four_mp3_frames + htk[12 + len(four_mp3_frames) :]

        assert calculate_request_duration(("note.htk", disguised)) == pytest.approx(1.0)

    def test_an_mp3_is_measured_by_its_frames_when_stray_bytes_split_them_and_its_info_header_understates(self):
        mp3: Final = (RECORDINGS / "tone-1.5s.mp3").read_bytes()
        info_frames: Final = mp3.index(b"Info") + 8
        # 10 of its 45 frames, which soundfile reads as 0.276 s once it takes off the encoder delay and padding
        understated: Final = mp3[:info_frames] + (10).to_bytes(4, "big") + mp3[info_frames + 4 :]
        # A 180-byte Info frame, then 24 kbps frames at 16 kHz of 108 bytes: a stray byte after every third frame
        # leaves no four in a row
        first_frame: Final = mp3.index(b"\xff\xf3")
        audio: Final = first_frame + 180
        frames: Final = (understated[first_frame:audio], *(mp3[at : at + 108] for at in range(audio, len(mp3), 108)))
        split: Final = mp3[:first_frame] + b"".join(
            b"".join(frames[group : group + 3]) + b"\x00" for group in range(0, len(frames), 3)
        )

        assert calculate_request_duration(split) == calculate_request_duration(mp3)

    def test_flac_silence_is_not_capped_below_its_length(self):
        import numpy
        import soundfile

        buffer: Final = io.BytesIO()
        soundfile.write(buffer, numpy.zeros(16000 * 30, dtype="int16"), 16000, format="FLAC")

        assert calculate_request_duration(("silence.flac", buffer.getvalue())) == pytest.approx(30)

    def test_a_flac_that_never_stated_its_length_reads_none(self):
        import numpy
        import soundfile

        buffer: Final = io.BytesIO()
        soundfile.write(buffer, numpy.zeros(16000 * 20, dtype="int16"), 16000, format="FLAC")
        flac: Final = buffer.getvalue()
        # STREAMINFO keeps the total samples in the low 36 bits of its bytes 10-17, 0 when the encoder wrote to a pipe
        fields: Final = int.from_bytes(flac[18:26], "big")
        piped: Final = flac[:18] + (fields >> 36 << 36).to_bytes(8, "big") + flac[26:]

        assert calculate_request_duration(("note.flac", flac)) == pytest.approx(20)
        assert calculate_request_duration(("note.flac", piped)) is None


class TestLongestAudioSeconds:
    @pytest.mark.parametrize(
        ("readings", "expected"),
        [
            ((1.5,), 1.5),
            ((1.5, None, None, 2), 2),
            ((1.5, None, 3.25, None), 3.25),
            ((9.0, 4.5, 3.25, 2), 9.0),
            ((None, None, None, 2), 2),
            ((None, None, None, None), None),
            ((), None),
        ],
    )
    def test_the_longest_reading_is_billed(self, readings, expected):
        assert longest_audio_seconds(*readings) == expected

    @pytest.mark.parametrize("reading", (True, -3.0, 0, math.nan, math.inf, "12"))
    def test_a_reading_that_is_not_a_positive_finite_number_is_ignored(self, reading):
        assert longest_audio_seconds(1.5, reading) == 1.5


class TestPricedByAudioLength:
    PER_SECOND: Final = {"input_cost_per_second": 0.0001}
    PER_TOKEN: Final = {"input_cost_per_audio_token": 0.0000025, "output_cost_per_token": 0.00001}

    @pytest.mark.parametrize(
        ("deployment", "cost_map", "expected"),
        [
            (PER_SECOND, {}, True),
            ({"output_cost_per_second": 0.0001}, {}, True),
            ({}, PER_SECOND, True),
            ({"id": "route", "mode": "audio_transcription"}, PER_SECOND, True),
            (PER_SECOND, PER_TOKEN, True),
            (PER_TOKEN, PER_SECOND, False),
            ({}, {**PER_SECOND, **PER_TOKEN}, False),
            ({"input_cost_per_second": 0}, PER_SECOND, False),
            ({"input_cost_per_second": 0, "input_cost_per_token": 0}, {}, False),
            ({}, {}, False),
        ],
    )
    def test_the_deployment_prices_decide_when_it_states_any(self, deployment, cost_map, expected):
        assert priced_by_audio_length(deployment, cost_map) is expected


class TestGetAudioFileContentHash:
    """Test the get_audio_file_content_hash function for cache key generation"""

    def test_different_content_same_filename_different_hash(self):
        """Test that different content with same filename produces different hashes"""
        content1 = b"audio content 1"
        content2 = b"audio content 2"
        filename = "test.mp3"

        hash1 = get_audio_file_content_hash((filename, content1))
        hash2 = get_audio_file_content_hash((filename, content2))

        assert hash1 != hash2, "Different content should produce different hashes"

    def test_same_content_same_hash(self):
        """Test that same content produces same hash"""
        content = b"same audio content"
        filename1 = "test1.mp3"
        filename2 = "test2.mp3"

        hash1 = get_audio_file_content_hash((filename1, content))
        hash2 = get_audio_file_content_hash((filename2, content))

        assert (
            hash1 == hash2
        ), "Same content should produce same hash regardless of filename"

    def test_bytes_input(self):
        """Test that raw bytes input works"""
        content = b"raw bytes content"
        hash1 = get_audio_file_content_hash(content)
        hash2 = get_audio_file_content_hash(content)

        assert hash1 == hash2, "Same bytes should produce same hash"
        assert len(hash1) == 64, "SHA-256 hash should be 64 characters"

    def test_fallback_to_filename(self):
        """Test that function falls back to filename when content extraction fails"""

        # Use a non-readable object that will trigger fallback
        class UnreadableFile:
            def __init__(self, name):
                self.name = name

        file_obj = UnreadableFile("test.mp3")
        hash_result = get_audio_file_content_hash(file_obj)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64, "Should return valid hash even on fallback"


class TestNormalizeTranscriptionLanguageToBcp47:
    @pytest.mark.parametrize(
        "language,expected",
        [
            ("en", "en-US"),
            ("EN", "en-US"),
            ("ja", "ja-JP"),
            ("en-US", "en-US"),
            ("en-GB", "en-GB"),
            ("auto", "auto"),
            ("xx", "xx"),
        ],
    )
    def test_normalization(self, language, expected):
        from litellm.litellm_core_utils.audio_utils.utils import (
            normalize_transcription_language_to_bcp47,
        )

        assert normalize_transcription_language_to_bcp47(language) == expected


class TestResolveSpeechMediaType:
    @pytest.mark.parametrize(
        ("upstream_content_type", "response_format", "expected"),
        [
            ("audio/wav", None, "audio/wav"),
            ("AUDIO/WAV", None, "audio/wav"),
            ("audio/flac; charset=binary", "mp3", "audio/flac"),
            ("application/json", "flac", "audio/flac"),
            ("application/octet-stream", "pcm", "audio/pcm"),
            (None, "wav", "audio/wav"),
            (None, "WAV", "audio/wav"),
            (None, "opus", "audio/opus"),
            (None, "aac", "audio/aac"),
            (None, "mp3", "audio/mpeg"),
            (None, "mp4", "audio/mpeg"),
            (None, "bogus", "audio/mpeg"),
            (None, None, "audio/mpeg"),
            ("", None, "audio/mpeg"),
        ],
    )
    def test_resolution(self, upstream_content_type, response_format, expected):
        from litellm.litellm_core_utils.audio_utils.utils import resolve_speech_media_type

        resolved = resolve_speech_media_type(
            upstream_content_type=upstream_content_type,
            response_format=response_format,
        )
        assert resolved == expected


class TestSpeechMediaTypeFromAudioBytes:
    @pytest.mark.parametrize(
        ("audio", "expected"),
        [
            (b"RIFF\x24\x00\x00\x00WAVEfmt ", "audio/wav"),
            (b"fLaC\x00\x00\x00\x22", "audio/flac"),
            (b"OggS" + b"\x00" * 24 + b"OpusHead", "audio/opus"),
            (b"OggS" + b"\x00" * 24 + b"\x01vorbis", "audio/ogg"),
            (b"ID3\x04\x00\x00\x00\x00\x00\x00", "audio/mpeg"),
            (b"\xff\xfb\x90\x64", "audio/mpeg"),
            (b"\xff\xf3\x80\x00", "audio/mpeg"),
            (b"\xff\xf1\x50\x80", "audio/aac"),
            (b"\xff\xf9\x50\x80", "audio/aac"),
            (b"RIFF\x24\x00\x00\x00AVI LIST", None),
            (b"\xff\xff\xff\xff\xff\xff", None),
            (b"\xff\xfb\xf0\x00", None),
            (b"\xff\xfb\x9c\x00", None),
            (b"\xff\xeb\x90\x00", None),
            (b"\xff\xf1\xf4\x80", None),
            (b"\xff\x00\x00\x00", None),
            (b"\x00\x01\x02\x03\x04\x05", None),
            (b"\xff\xfb", None),
            (b"\xff", None),
            (b"", None),
        ],
    )
    def test_sniffing(self, audio, expected):
        from litellm.litellm_core_utils.audio_utils.utils import speech_media_type_from_audio_bytes

        assert speech_media_type_from_audio_bytes(audio) == expected
