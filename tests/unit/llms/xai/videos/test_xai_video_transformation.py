import base64
import io
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Final

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.llms.xai.videos.transformation import XAIVideoConfig, XAIVideoError
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.utils import ProviderConfigManager

MODEL: Final = "grok-imagine-video"
API_BASE: Final = "https://api.x.ai"
PNG: Final = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
MP4: Final = b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 32


def _create(params: dict[str, object], prompt: str = "a paper boat on a rainy street") -> dict[str, object]:
    body, files, url = XAIVideoConfig().transform_video_create_request(
        model=MODEL,
        prompt=prompt,
        api_base=API_BASE,
        video_create_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert url == "https://api.x.ai/v1/videos/generations"
    assert files == []
    return body


def _mapped(params: dict[str, object], model: str = MODEL) -> dict[str, object]:
    return XAIVideoConfig().map_openai_params(params, model, False)  # pyright: ignore[reportArgumentType]  # test passes plain dicts


def _job_response(body: dict[str, object], request_id: str = "req-1") -> httpx.Response:
    return httpx.Response(200, json=body, request=httpx.Request("GET", f"{API_BASE}/v1/videos/{request_id}"))


def test_the_provider_serves_video_jobs():
    assert isinstance(ProviderConfigManager.get_provider_video_config(MODEL, LlmProviders.XAI), XAIVideoConfig)


def test_a_request_always_names_the_seconds_and_resolution_it_is_charged_for():
    assert _create({}) == {
        "model": MODEL,
        "prompt": "a paper boat on a rainy street",
        "duration": 8,
        "resolution": "480p",
    }


@pytest.mark.parametrize(
    ("openai_params", "xai_params"),
    [
        ({"seconds": "5", "size": "1280x720"}, {"duration": "5", "resolution": "720p", "aspect_ratio": "16:9"}),
        ({"size": "720x1280"}, {"resolution": "720p", "aspect_ratio": "9:16"}),
        ({"size": "854x480"}, {"resolution": "480p", "aspect_ratio": "16:9"}),
        ({"size": "1024x1024"}, {"resolution": "720p", "aspect_ratio": "1:1"}),
        ({"size": "1920x1080"}, {"resolution": "720p", "aspect_ratio": "16:9"}),
        ({"size": "1792x1024"}, {"resolution": "720p", "aspect_ratio": "16:9"}),
        ({"size": "720p"}, {"resolution": "720p"}),
        ({"size": "huge"}, {}),
        ({"size": "0x720"}, {}),
        ({"user": "agent-7"}, {"user": "agent-7"}),
        ({"input_reference": "https://example.com/first.png"}, {"image": "https://example.com/first.png"}),
        ({"generate_audio": False}, {"generate_audio": False}),
    ],
)
def test_openai_params_map_to_xai_fields(openai_params: dict[str, object], xai_params: dict[str, object]):
    assert _mapped(openai_params) == xai_params


def test_a_size_beyond_the_model_s_largest_resolution_asks_for_the_largest():
    assert _mapped({"size": "1920x1080"}, "grok-imagine-video-1.5") == {"resolution": "1080p", "aspect_ratio": "16:9"}


@pytest.mark.parametrize(
    "params",
    [
        {"duration": 0},
        {"duration": 16},
        {"duration": "7.5"},
        {"duration": "eight"},
        {"duration": True},
        {"resolution": "1080p"},
        {"resolution": 720},
        {"aspect_ratio": "7:4"},
        {"generate_audio": "no"},
    ],
)
def test_a_request_xai_would_fail_after_accepting_it_is_refused_before_it_is_sent(params: dict[str, object]):
    with pytest.raises(ValueError, match=r"^xAI (video|grok-imagine-video) "):
        _create(params)


def test_a_request_needs_a_prompt_or_an_image():
    with pytest.raises(ValueError, match="needs a prompt or an image"):
        _create({}, prompt="  ")
    assert _create({"image": "https://example.com/first.png"}, prompt="") == {
        "model": MODEL,
        "duration": 8,
        "resolution": "480p",
        "image": {"url": "https://example.com/first.png"},
    }


def test_fields_that_cost_more_than_the_per_second_price_never_reach_xai():
    body = _create(
        {
            "storage_options": {"filename": "clip.mp4"},
            "output": {"upload_url": "https://example.com/upload"},
            "reference_images": [{"url": "https://example.com/ref.png"}],
            "keyframes": [],
            "aspect_ratio": "9:16",
        }
    )

    assert body == {
        "model": MODEL,
        "prompt": "a paper boat on a rainy street",
        "duration": 8,
        "resolution": "480p",
        "aspect_ratio": "9:16",
    }


@pytest.mark.parametrize(
    "upload",
    [io.BytesIO(PNG), PNG, ("first.png", PNG, "image/png"), ("first.png", io.BytesIO(PNG))],
)
def test_an_uploaded_image_is_sent_as_a_data_url(upload: object):
    body = _create({"image": upload})

    assert body["image"] == {"url": f"data:image/png;base64,{base64.b64encode(PNG).decode()}"}


def test_an_uploaded_file_that_is_not_an_image_xai_reads_is_refused():
    with pytest.raises(ValueError, match="JPEG, PNG or WebP"):
        _create({"image": io.BytesIO(b"GIF89a" + b"\x00" * 8)})


def test_api_base_with_or_without_the_version_prefix_reaches_the_same_endpoint():
    config = XAIVideoConfig()

    assert config.get_complete_url(MODEL, "https://api.x.ai/v1/", {}) == API_BASE
    assert config.get_complete_url(MODEL, "https://api.x.ai", {}) == API_BASE


def test_the_key_comes_from_the_deployment():
    headers = XAIVideoConfig().validate_environment({}, MODEL, litellm_params=GenericLiteLLMParams(api_key="xai-k"))

    assert headers["Authorization"] == "Bearer xai-k"


def test_a_created_job_is_charged_for_the_seconds_and_resolution_it_asked_for():
    request = _create({"duration": "5", "resolution": "720p"})
    response = httpx.Response(200, json={"request_id": "req-1"})

    video = XAIVideoConfig().transform_video_create_response(
        model=MODEL, raw_response=response, logging_obj=None, custom_llm_provider="xai", request_data=request
    )

    decoded = decode_video_id_with_provider(video.id)
    assert (decoded.get("custom_llm_provider"), decoded.get("model_id"), decoded.get("video_id")) == (
        "xai",
        MODEL,
        "req-1",
    )
    assert (video.status, video.seconds) == ("queued", "5")
    assert video.usage == {"duration_seconds": 5.0, "video_resolution": "720p"}


def test_polls_and_downloads_ask_for_the_provider_s_own_request_id():
    video_id = encode_video_id_with_provider("req/../1", "xai", MODEL)
    config = XAIVideoConfig()
    params = GenericLiteLLMParams()

    assert config.transform_video_status_retrieve_request(video_id, API_BASE, params, {}) == (
        f"{API_BASE}/v1/videos/req%2F..%2F1",
        {},
    )
    assert config.transform_video_content_request(video_id, API_BASE, params, {}) == (
        f"{API_BASE}/v1/videos/req%2F..%2F1",
        {},
    )


@pytest.mark.parametrize(
    ("job", "status", "error_code"),
    [
        ({"status": "pending", "progress": 40}, "in_progress", None),
        (
            {"status": "done", "model": MODEL, "video": {"url": "https://vidgen.x.ai/a.mp4", "duration": 6}},
            "completed",
            None,
        ),
        (
            {"status": "failed", "error": {"code": "invalid_argument", "message": "bad image"}},
            "failed",
            "invalid_argument",
        ),
        ({"status": "failed"}, "failed", "unknown"),
        ({"status": "expired"}, "failed", "unknown"),
        (
            {"status": "done", "model": MODEL, "video": {"url": "", "respect_moderation": False}},
            "failed",
            "moderation_blocked",
        ),
    ],
)
def test_a_poll_reports_the_job_in_openai_terms(job: dict[str, object], status: str, error_code: str | None):
    video = XAIVideoConfig().transform_video_status_retrieve_response(_job_response(job), None, "xai")

    assert video.status == status
    assert (video.error or {}).get("code") == error_code
    assert video.usage is None
    assert decode_video_id_with_provider(video.id).get("video_id") == "req-1"


def test_a_poll_reports_progress_and_the_finished_length():
    config = XAIVideoConfig()

    pending = config.transform_video_status_retrieve_response(
        _job_response({"status": "pending", "progress": 40}), None
    )
    done = config.transform_video_status_retrieve_response(
        _job_response({"status": "done", "video": {"url": "https://vidgen.x.ai/a.mp4", "duration": 6}}), None
    )

    assert pending.progress == 40
    assert done.seconds == "6"


def _download_handler(recorded: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        recorded.append(request)  # mutable-ok: records what the client sent
        return httpx.Response(200, content=MP4, headers={"content-type": "video/mp4"})

    return handle


def test_a_finished_job_s_file_is_downloaded_without_the_api_key():
    recorded: list[httpx.Request] = []  # mutable-ok: records what the client sent
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_download_handler(recorded))))
    config = XAIVideoConfig(sync_client_factory=lambda: client)
    job = _job_response({"status": "done", "video": {"url": "https://vidgen.x.ai/a.mp4", "duration": 6}})

    assert config.transform_video_content_response(job, None) == MP4
    assert [str(request.url) for request in recorded] == ["https://vidgen.x.ai/a.mp4"]
    assert "authorization" not in recorded[0].headers


@pytest.mark.asyncio
async def test_a_finished_job_s_file_is_downloaded_asynchronously():
    recorded: list[httpx.Request] = []  # mutable-ok: records what the client sent
    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(_download_handler(recorded)))
    config = XAIVideoConfig(async_client_factory=lambda: client)
    job = _job_response({"status": "done", "video": {"url": "https://vidgen.x.ai/a.mp4", "duration": 6}})

    assert await config.async_transform_video_content_response(job, None) == MP4
    assert [str(request.url) for request in recorded] == ["https://vidgen.x.ai/a.mp4"]


@pytest.mark.parametrize(
    "job",
    [
        {"status": "pending", "progress": 10},
        {"status": "failed", "error": {"code": "internal_error", "message": "boom"}},
        {"status": "done", "video": {"url": "", "respect_moderation": False}},
    ],
)
def test_a_job_without_a_file_downloads_nothing(job: dict[str, object]):
    recorded: list[httpx.Request] = []  # mutable-ok: records what the client sent
    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_download_handler(recorded))))
    config = XAIVideoConfig(sync_client_factory=lambda: client)

    with pytest.raises(ValueError, match=r"Video generation failed|still processing"):
        config.transform_video_content_response(_job_response(job), None)
    assert recorded == []


def test_errors_carry_the_provider_status():
    error = XAIVideoConfig().get_error_class("rate limited", 429, {})

    assert isinstance(error, XAIVideoError)
    assert error.status_code == 429


def test_a_forwarding_deployment_keeps_the_resolution_the_upstream_proxy_priced():
    upstream = httpx.Response(
        200,
        json={
            "id": encode_video_id_with_provider("req-1", "xai", MODEL),
            "object": "video",
            "status": "queued",
            "seconds": "5",
            "usage": {"duration_seconds": 5.0, "video_resolution": "720p"},
        },
    )

    video = OpenAIVideoConfig().transform_video_create_response(
        model="xai/grok-imagine-video", raw_response=upstream, logging_obj=None, custom_llm_provider="openai"
    )

    assert video.usage == {"duration_seconds": 5.0, "video_resolution": "720p"}


class _CostRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.costs: list[tuple[str, float]] = []  # mutable-ok: records each logged call's cost

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.costs.append((kwargs["call_type"], kwargs["response_cost"]))  # mutable-ok: records each logged call


@contextmanager
def _fake_xai(sent: list[dict[str, object]]) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            assert self.path == "/v1/videos/generations"
            assert self.headers["authorization"] == "Bearer xai-k"
            sent.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))  # mutable-ok: records it
            self._reply(200, json.dumps({"request_id": "req-1"}).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path == "/files/req-1.mp4":
                assert "authorization" not in self.headers
                self._reply(200, MP4, "video/mp4")
                return
            assert self.path == "/v1/videos/req-1"
            assert self.headers["authorization"] == "Bearer xai-k"
            job = {"status": "done", "model": MODEL, "video": {"url": f"{base}/files/req-1.mp4", "duration": 4}}
            self._reply(200, json.dumps(job).encode(), "application/json")

        def log_message(self, format: str, *args: object) -> None:
            return None

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base: Final = f"http://127.0.0.1:{server.server_address[1]}"
    thread: Final = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield base
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("size", "seconds", "rate"), [("854x480", "4", 0.05), ("1280x720", "6", 0.07)])
async def test_a_routed_job_is_charged_once_at_the_route_s_price_for_its_resolution(
    size: str, seconds: str, rate: float, monkeypatch: pytest.MonkeyPatch
):
    sent: list[dict[str, object]] = []  # mutable-ok: records what reached xAI
    recorder = _CostRecorder()
    litellm.logging_callback_manager._reset_all_callbacks()  # pyright: ignore[reportPrivateUsage]  # isolate the recorder
    await GLOBAL_LOGGING_WORKER.flush()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    try:
        with _fake_xai(sent) as base:
            router = litellm.Router(
                model_list=[
                    {
                        "model_name": "xai/grok-imagine-video",
                        "litellm_params": {"model": "xai/grok-imagine-video", "api_key": "xai-k", "api_base": base},
                        "model_info": {
                            "id": "grok-imagine-video-xai",
                            "mode": "video_generation",
                            "custom_pricing": True,
                            "output_cost_per_second_480p": 0.05,
                            "output_cost_per_second_720p": 0.07,
                        },
                    }
                ],
                num_retries=0,
            )
            video = await router.avideo_generation(
                model="xai/grok-imagine-video", prompt="a lighthouse at dusk", seconds=seconds, size=size
            )
            status = await router.avideo_status(video_id=video.id, model="xai/grok-imagine-video")
            content = await router.avideo_content(video_id=video.id, model="xai/grok-imagine-video")
            await GLOBAL_LOGGING_WORKER.flush()
    finally:
        litellm.logging_callback_manager._reset_all_callbacks()  # pyright: ignore[reportPrivateUsage]  # isolate the recorder

    assert [body["duration"] for body in sent] == [int(seconds)]
    assert (status.status, content) == ("completed", MP4)
    charged = [cost for call_type, cost in recorder.costs if cost]
    assert charged == [pytest.approx(rate * int(seconds))]
