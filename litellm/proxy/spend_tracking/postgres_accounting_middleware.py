import asyncio

import anyio

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.accounting_context import accounting_request
from litellm.proxy.spend_tracking import postgres_accounting


class AccountingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        runtime = postgres_accounting.runtime
        path = scope.get("path", "")
        if runtime is None or scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "http" and (
            (
                scope.get("method") == "GET"
                and (
                    path.startswith("/health/")
                    or path in {"/models", "/v1/models", "/model/info", "/key/info", "/user/info", "/team/info"}
                )
            )
            or (
                scope.get("method") == "POST"
                and path
                in {
                    "/key/generate",
                    "/key/update",
                    "/user/new",
                    "/user/update",
                    "/team/new",
                    "/team/update",
                    "/health/resume",
                    "/model/new",
                    "/model/update",
                    "/model/delete",
                }
            )
            or (scope.get("method") == "PATCH" and path.startswith("/model/") and path.endswith("/update"))
        ):
            await self.app(scope, receive, send)
            return
        if (
            path not in {"/chat/completions", "/v1/chat/completions", "/responses", "/v1/responses"}
            or scope["type"] != "http"
            or scope.get("method") != "POST"
        ):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1013})
            else:
                await JSONResponse(
                    {"error": "PostgreSQL accounting first vertical: route integration pending"}, status_code=503
                )(scope, receive, send)
            return
        try:
            request = await runtime.begin()
        except Exception:
            await JSONResponse({"error": "PostgreSQL accounting unavailable or draining"}, status_code=503)(
                scope, receive, send
            )
            return
        token = accounting_request.set(request)
        cancelled = False
        try:
            await self.app(scope, receive, send)
        except (asyncio.CancelledError, GeneratorExit):
            cancelled = True
            raise
        except BaseException:
            request.failed = request.failed or request.dispatched
            raise
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    if cancelled:
                        await runtime.cancel()
                    await request.finish()
            except BaseException:
                verbose_proxy_logger.exception("Accounting finalization failed; durable request remains pending")
            accounting_request.reset(token)
