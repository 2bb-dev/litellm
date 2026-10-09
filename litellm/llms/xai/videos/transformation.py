"""
Translates OpenAI's `/v1/videos` to xAI's Grok Imagine video API: create a job, poll it, then download the clip.

https://docs.x.ai/developers/rest-api-reference/inference/videos
"""

import base64
import io
import math
import re
import time
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, NoReturn, TypeAlias
from urllib.parse import unquote

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # shared HTTP factory is private
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory lacks typed params
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

from ..common_utils import XAIModelInfo

_VideoParams: TypeAlias = dict[str, object]
_VideoHeaders: TypeAlias = dict[str, str]

_PROVIDER: Final = LlmProviders.XAI.value
_DEFAULT_API_BASE: Final = "https://api.x.ai"

# xAI's defaults when a request leaves them out (docs.x.ai video generation guide, 2026-10-09). Every request
# names both, so the seconds and resolution priced when the job is created are the ones xAI renders.
_DEFAULT_SECONDS: Final = 8
_DEFAULT_RESOLUTION: Final = "480p"
_SECONDS: Final = range(1, 16)
_RESOLUTION_SHORT_SIDES: Final = (("480p", 480), ("720p", 720), ("1080p", 1080))
_ALL_RESOLUTIONS: Final = tuple(label for label, _ in _RESOLUTION_SHORT_SIDES)
_MODEL_RESOLUTIONS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({"grok-imagine-video": ("480p", "720p")})
_ASPECT_RATIOS: Final = ("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9", "5:2")

# Stored outputs, upload URLs, reference media and keyframes cost more than the per-second price, so they stay out.
_REQUEST_FIELDS: Final = frozenset(
    {"model", "prompt", "duration", "resolution", "aspect_ratio", "image", "generate_audio", "user"}
)
_STATUSES: Final[Mapping[str, str]] = MappingProxyType(
    {"pending": "in_progress", "done": "completed", "failed": "failed", "expired": "failed"}
)
_MODERATED: Final[Mapping[str, str]] = MappingProxyType(
    {"code": "moderation_blocked", "message": "xAI's moderation filtered this video, so it has no file"}
)
_OBJECT_TUPLE: Final = TypeAdapter(tuple[object, ...])
_STRING_OBJECT_DICT: Final = TypeAdapter(dict[str, object])


class XAIVideoError(BaseLLMException):
    pass


class _CreatedJob(BaseModel):
    request_id: str


class _JobVideo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str | None = None
    duration: float | None = None
    respect_moderation: bool | None = None


class _JobError(BaseModel):
    model_config = ConfigDict(extra="ignore")
    code: str = "unknown"
    message: str = "Video generation failed"


class _Job(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: str
    model: str | None = None
    progress: float | None = None
    video: _JobVideo | None = None
    error: _JobError | None = None

    def is_filtered(self) -> bool:
        return self.status == "done" and (
            self.video is None or not self.video.url or self.video.respect_moderation is False
        )


def _resolutions(model: str) -> tuple[str, ...]:
    return _MODEL_RESOLUTIONS.get(model, _ALL_RESOLUTIONS)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _whole_seconds(value: object) -> int:
    seconds: Final = _number(value)
    if seconds is not None and seconds.is_integer() and int(seconds) in _SECONDS:
        return int(seconds)
    raise ValueError(f"xAI video seconds must be a whole number from {_SECONDS[0]} to {_SECONDS[-1]}")


def _checked_resolution(model: str, value: object) -> str:
    supported: Final = _resolutions(model)
    if isinstance(value, str) and value.lower() in supported:
        return value.lower()
    raise ValueError(f"xAI {model} renders {', '.join(supported)}")


def _ratio(aspect_ratio: str) -> float:
    width, _, height = aspect_ratio.partition(":")
    return int(width) / int(height)


def _size_params(model: str, size: object) -> Mapping[str, str]:
    if not isinstance(size, str):
        return MappingProxyType({})
    normalized: Final = size.strip().lower()
    if normalized in _ALL_RESOLUTIONS:
        return MappingProxyType({"resolution": normalized})
    dimensions: Final = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", normalized)
    if dimensions is None:
        return MappingProxyType({})
    width: Final = int(dimensions[1])
    height: Final = int(dimensions[2])
    supported: Final = _resolutions(model)
    resolution: Final = next(
        (label for label, side in _RESOLUTION_SHORT_SIDES if label in supported and min(width, height) <= side),
        supported[-1],
    )
    aspect_ratio: Final = min(_ASPECT_RATIOS, key=lambda ratio: abs(math.log(_ratio(ratio) * height / width)))
    return MappingProxyType({"resolution": resolution, "aspect_ratio": aspect_ratio})


def _image_bytes(value: object) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, io.BytesIO):
        return value.getvalue()
    if isinstance(value, io.BufferedIOBase):
        return value.read()
    if isinstance(value, tuple):
        file_tuple: Final = _OBJECT_TUPLE.validate_python(value)
        if len(file_tuple) >= 2:
            return _image_bytes(file_tuple[1])
    raise ValueError("xAI video takes an image URL or the image file itself")


def _image_media_type(content: bytes) -> str:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("xAI video takes a JPEG, PNG or WebP image")


def _image(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return MappingProxyType(_STRING_OBJECT_DICT.validate_python(value))
    if isinstance(value, str):
        return MappingProxyType({"url": value})
    content: Final = _image_bytes(value)
    encoded: Final = base64.b64encode(content).decode("ascii")
    return MappingProxyType({"url": f"data:{_image_media_type(content)};base64,{encoded}"})


def _checked_optional_fields(params: Mapping[str, object]) -> None:
    aspect_ratio: Final = params.get("aspect_ratio")
    if aspect_ratio is not None and aspect_ratio not in _ASPECT_RATIOS:
        raise ValueError(f"xAI video aspect_ratio must be one of {', '.join(_ASPECT_RATIOS)}")
    generate_audio: Final = params.get("generate_audio")
    if generate_audio is not None and not isinstance(generate_audio, bool):
        raise ValueError("xAI video generate_audio must be true or false")
    user: Final = params.get("user")
    if user is not None and not isinstance(user, str):
        raise ValueError("xAI video user must be a string")


def _request_id_from_url(raw_response: httpx.Response) -> str:
    return unquote(raw_response.request.url.path.rsplit("/", 1)[-1])


def _job_error(job: _Job) -> Mapping[str, object] | None:
    if job.is_filtered():
        return _MODERATED
    if job.error is not None:
        return MappingProxyType({"code": job.error.code, "message": job.error.message})
    if _STATUSES.get(job.status) == "failed":
        return MappingProxyType(_JobError().model_dump())
    return None


def _status_video(job: _Job, request_id: str, provider: str) -> VideoObject:
    error: Final = _job_error(job)
    seconds: Final = job.video.duration if job.video is not None else None
    return VideoObject(
        id=encode_video_id_with_provider(request_id, provider, job.model),
        object="video",
        status="failed" if error is not None else _STATUSES.get(job.status, "in_progress"),
        progress=min(100, max(0, round(job.progress))) if job.progress is not None else None,
        seconds=f"{seconds:g}" if seconds is not None else None,
        model=job.model,
        error=dict(error) if error is not None else None,  # mutable-ok: VideoObject.error is a dict
    )


def _video_url(raw_response: httpx.Response) -> str:
    job: Final = _Job.model_validate(raw_response.json())
    if job.status == "done" and not job.is_filtered() and job.video is not None and job.video.url:
        return job.video.url
    error: Final = _job_error(job)
    if error is not None:
        raise ValueError(f"Video generation failed: {error['message']}")
    raise ValueError(f"Video is still processing (status: {job.status}). Please wait and try again.")


def _xai_async_httpx_client() -> AsyncHTTPHandler:
    return get_async_httpx_client(llm_provider=LlmProviders.XAI)


class XAIVideoConfig(BaseVideoConfig):
    def __init__(
        self,
        sync_client_factory: Callable[[], HTTPHandler] = _get_httpx_client,
        async_client_factory: Callable[[], AsyncHTTPHandler] = _xai_async_httpx_client,
    ) -> None:
        super().__init__()
        self._sync_client_factory: Final = sync_client_factory
        self._async_client_factory: Final = async_client_factory

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig returns a list
        return ["model", "prompt", "input_reference", "seconds", "size", "user", "extra_headers"]  # mutable-ok: same

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> _VideoParams:
        supported: Final = frozenset(self.get_supported_openai_params(model))
        input_reference: Final = video_create_optional_params.get("input_reference")
        seconds: Final = video_create_optional_params.get("seconds")
        user: Final = video_create_optional_params.get("user")
        passthrough: Final = MappingProxyType(
            {key: value for key, value in video_create_optional_params.items() if key not in supported}
        )
        mapped: Final[_VideoParams] = {
            **passthrough,
            **({"image": input_reference} if input_reference is not None else {}),
            **({"duration": seconds} if seconds is not None else {}),
            **_size_params(model, video_create_optional_params.get("size")),
            **({"user": user} if user is not None else {}),
        }
        return mapped

    def validate_environment(
        self,
        headers: _VideoHeaders,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> _VideoHeaders:
        final_api_key: Final = XAIModelInfo.get_api_key(
            api_key or (litellm_params.api_key if litellm_params is not None else None)
        )
        if not final_api_key:
            raise ValueError("xAI API key is required: set XAI_API_KEY or pass api_key")
        validated: Final[_VideoHeaders] = {
            **headers,
            "Authorization": f"Bearer {final_api_key}",
            "Content-Type": "application/json",
        }
        return validated

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: _VideoParams,
    ) -> str:
        return (XAIModelInfo.get_api_base(api_base) or _DEFAULT_API_BASE).rstrip("/").removesuffix("/v1")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: _VideoParams,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[_VideoParams, RequestFiles, str]:
        params: Final = video_create_optional_request_params
        raw_image: Final = params.get("image")
        image: Final = _image(raw_image) if raw_image is not None else None
        if not prompt.strip() and image is None:
            raise ValueError("xAI video needs a prompt or an image")
        _checked_optional_fields(params)
        sent_fields: Final = MappingProxyType({key: value for key, value in params.items() if key in _REQUEST_FIELDS})
        request_data: Final[_VideoParams] = {
            **sent_fields,
            "model": model,
            **({"prompt": prompt} if prompt.strip() else {}),
            "duration": _whole_seconds(params.get("duration", _DEFAULT_SECONDS)),
            "resolution": _checked_resolution(model, params.get("resolution", _DEFAULT_RESOLUTION)),
            **({"image": dict(image)} if image is not None else {}),  # mutable-ok: json.dumps needs a dict
        }
        files: Final[RequestFiles] = []  # mutable-ok: HTTP files payload requires a list
        return request_data, files, f"{api_base}/v1/videos/generations"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        created: Final = _CreatedJob.model_validate(raw_response.json())
        request_params: Final = request_data or MappingProxyType({})
        duration: Final = request_params.get("duration")
        resolution: Final = request_params.get("resolution")
        seconds: Final = duration if isinstance(duration, int) and not isinstance(duration, bool) else None
        video: Final = VideoObject(
            id=encode_video_id_with_provider(created.request_id, custom_llm_provider or _PROVIDER, model),
            object="video",
            status="queued",
            created_at=int(time.time()),
            model=model,
            seconds=str(seconds) if seconds is not None else None,
        )
        video.usage = {  # mutable-ok: VideoObject.usage is a dict
            key: value
            for key, value in (
                ("duration_seconds", float(seconds) if seconds is not None else None),
                ("video_resolution", resolution if isinstance(resolution, str) else None),
            )
            if value is not None
        }
        return video

    def _job_url(self, video_id: str, api_base: str) -> tuple[str, _VideoParams]:
        request_id: Final = encode_url_path_segment(extract_original_video_id(video_id), field_name="video_id")
        no_params: Final[_VideoParams] = {}
        return f"{api_base}/v1/videos/{request_id}", no_params

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        return self._job_url(video_id, api_base)

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        return _status_video(
            _Job.model_validate(raw_response.json()),
            _request_id_from_url(raw_response),
            custom_llm_provider or _PROVIDER,
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        variant: str | None = None,
    ) -> tuple[str, _VideoParams]:
        return self._job_url(video_id, api_base)

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> bytes:
        download: Final[httpx.Response] = self._sync_client_factory().get(  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
            _video_url(raw_response)
        )
        download.raise_for_status()
        return download.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> bytes:
        download: Final[httpx.Response] = await self._async_client_factory().get(  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
            _video_url(raw_response)
        )
        download.raise_for_status()
        return download.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        extra_body: Mapping[str, object] | None = None,
    ) -> NoReturn:
        raise NotImplementedError("video remix is not supported for xAI")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> NoReturn:
        raise NotImplementedError("video remix is not supported for xAI")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> NoReturn:
        raise NotImplementedError("video listing is not supported for xAI")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> NoReturn:
        raise NotImplementedError("video listing is not supported for xAI")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> NoReturn:
        raise NotImplementedError("video delete is not supported for xAI")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> NoReturn:
        raise NotImplementedError("video delete is not supported for xAI")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: _VideoHeaders | httpx.Headers,
    ) -> BaseLLMException:
        return XAIVideoError(status_code=status_code, message=error_message, headers=headers)
