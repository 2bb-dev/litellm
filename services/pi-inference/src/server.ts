import { randomBytes, timingSafeEqual } from "node:crypto";
import { once } from "node:events";
import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import { performance } from "node:perf_hooks";
import { z } from "zod";
import type {
  AssistantMessage,
  Models,
  Model,
  Api,
} from "@earendil-works/pi-ai";
import { chatResponse, createChatEncoder, prepareChat } from "./chat.js";
import {
  createMessagesEncoder,
  messagesResponse,
  prepareMessages,
} from "./messages.js";
import {
  failed,
  type ApiError,
  type Result,
  type WireEvent,
} from "./protocol.js";

export interface InferenceRuntime {
  models: Models;
  routes: ReadonlyMap<string, Model<Api>>;
}

export interface ServerOptions {
  apiKey: string;
  runtime: InferenceRuntime;
  log: (record: Record<string, unknown>) => void;
  timeoutMs?: number;
  maxBodyBytes?: number;
  maxInflight?: number;
}

const genericFailure: ApiError = {
  status: 502,
  type: "api_error",
  message: "Provider request failed; consult the correlated backend trace",
};
const timeoutFailure: ApiError = {
  status: 504,
  type: "timeout_error",
  message: "Inference deadline exceeded",
};
const modelRequest = z.object({ model: z.string().min(1) });

export function createInferenceServer(options: ServerOptions) {
  const active = new Set<AbortController>();
  const server = createServer((req, res) => {
    void handle(req, res).catch(() => {
      if (!res.headersSent)
        sendError(res, genericFailure, req.url === "/v1/messages");
      else res.destroy();
    });
  });
  server.requestTimeout = 30_000;
  server.headersTimeout = 10_000;
  server.keepAliveTimeout = 5_000;

  async function handle(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    const native = req.url === "/v1/messages";
    if (
      req.method === "GET" &&
      ["/health/liveliness", "/health/readiness"].includes(req.url ?? "")
    ) {
      sendJson(res, 200, { status: "ok" });
      return;
    }
    if (!authorized(req, options.apiKey)) {
      sendError(
        res,
        {
          status: 401,
          type: "authentication_error",
          message: "Invalid backend credential",
        },
        native,
      );
      return;
    }
    if (req.method === "GET" && req.url === "/v1/models") {
      sendJson(res, 200, {
        object: "list",
        data: [...options.runtime.routes].map(([alias, model]) => ({
          id: alias,
          object: "model",
          created: 0,
          owned_by: model.provider,
          pi_api:
            model.api === "anthropic-messages"
              ? "anthropic-messages"
              : "openai-completions",
          pi_provider_api: model.api,
          context_window: model.contextWindow,
          max_output_tokens: model.maxTokens,
          reasoning: model.reasoning,
        })),
      });
      return;
    }
    if (req.method === "GET" && req.url === "/v1/providers") {
      sendJson(res, 200, {
        data: options.runtime.models.getProviders().map((provider) => ({
          id: provider.id,
          name: provider.name,
          api_key: Boolean(provider.auth.apiKey),
          oauth: Boolean(provider.auth.oauth),
        })),
      });
      return;
    }
    if (
      req.method !== "POST" ||
      !["/v1/chat/completions", "/v1/messages"].includes(req.url ?? "")
    ) {
      sendError(
        res,
        {
          status: req.url === "/v1/responses" ? 501 : 404,
          type: "invalid_request_error",
          message:
            "Endpoint not supported; use Chat Completions or native Messages",
        },
        native,
      );
      return;
    }
    if (active.size >= (options.maxInflight ?? 16)) {
      res.setHeader("retry-after", "1");
      sendError(
        res,
        {
          status: 429,
          type: "rate_limit_error",
          message: "Backend concurrency limit reached",
        },
        native,
      );
      return;
    }
    const controller = new AbortController();
    const deadline = AbortSignal.timeout(options.timeoutMs ?? 90_000);
    const signal = AbortSignal.any([controller.signal, deadline]);
    const started = performance.now();
    const id =
      safeId(req.headers["x-request-id"]) ??
      `pi_${randomBytes(16).toString("hex")}`;
    const trace = traceContext(req.headers.traceparent);
    const result: {
      message?: AssistantMessage;
      error?: ApiError;
      firstTokenMs?: number;
      upstreamStatus?: number;
      upstreamId?: string;
      requests: number;
      alias?: string;
      provider?: string;
      model?: string;
      api?: string;
    } = { requests: 0 };
    const abort = () => controller.abort();
    active.add(controller);
    req.once("aborted", abort);
    res.once("close", abort);
    res.setHeader("x-request-id", id);
    res.setHeader("traceparent", trace.header);
    try {
      const body = await readBody(
        req,
        options.maxBodyBytes ?? 16 * 1024 * 1024,
        signal,
      );
      if (!body.ok) {
        result.error = body.error;
        sendError(res, body.error, native);
        return;
      }
      const selected = modelRequest.safeParse(body.value);
      if (!selected.success) {
        result.error = {
          status: 400,
          type: "invalid_request_error",
          message: "model is required",
        };
        sendError(res, result.error, native);
        return;
      }
      const alias = selected.data.model;
      const model = options.runtime.routes.get(alias);
      if (!model) {
        result.error = {
          status: 404,
          type: "not_found_error",
          message: "Model is not enabled on this backend",
        };
        sendError(res, result.error, native);
        return;
      }
      Object.assign(result, {
        alias,
        provider: model.provider,
        model: model.id,
        api: model.api,
      });
      const prepared = native
        ? prepareMessages(body.value, model)
        : prepareChat(body.value, model);
      if (!prepared.ok) {
        result.error = prepared.error;
        sendError(res, prepared.error, native);
        return;
      }
      const call = prepared.value;
      const stream = options.runtime.models.streamSimple(model, call.context, {
        ...call.options,
        signal,
        transport: "sse",
        maxRetries: 0,
        timeoutMs: options.timeoutMs ?? 90_000,
        headers: { traceparent: trace.header },
        onPayload: async (payload, resolved) => {
          result.requests += 1;
          return call.options.onPayload
            ? call.options.onPayload(payload, resolved)
            : undefined;
        },
        onResponse: (response) => {
          result.upstreamStatus = response.status;
          result.upstreamId = safeId(
            response.headers["request-id"] ?? response.headers["x-request-id"],
          );
        },
      });
      const encode = native
        ? createMessagesEncoder(alias, id)
        : createChatEncoder(alias, id, call.includeUsage);
      for await (const update of withDeadline(stream, signal)) {
        const event =
          update.type === "error" &&
          update.reason === "error" &&
          update.error.rawStopReason === "refusal"
            ? {
                type: "done" as const,
                reason: "stop" as const,
                message: {
                  ...update.error,
                  stopReason: "stop" as const,
                  errorMessage: undefined,
                },
              }
            : update;
        if (event.type === "text_delta" && result.firstTokenMs === undefined)
          result.firstTokenMs = performance.now() - started;
        if (event.type === "error") {
          result.message = event.error;
          result.error = deadline.aborted
            ? timeoutFailure
            : providerFailure(result.upstreamStatus);
          if (call.stream && res.headersSent)
            await writeEvent(
              res,
              {
                event: native ? "error" : undefined,
                data: errorBody(result.error, native),
              },
              signal,
            );
          else sendError(res, result.error, native);
          return;
        }
        if (event.type === "done") result.message = event.message;
        if (call.stream) {
          if (!res.headersSent)
            res.writeHead(200, {
              "content-type": "text/event-stream",
              "cache-control": "no-cache",
              "x-accel-buffering": "no",
            });
          for (const frame of encode(event)) {
            await writeEvent(res, frame, signal);
            if (frame.error) {
              result.error = frame.error;
              return;
            }
          }
        }
      }
      if (!result.message || failed(result.message)) {
        result.error = genericFailure;
        if (!res.headersSent) sendError(res, result.error, native);
        else
          await writeEvent(
            res,
            {
              event: native ? "error" : undefined,
              data: errorBody(result.error, native),
            },
            signal,
          );
        return;
      }
      if (!call.stream)
        sendJson(
          res,
          200,
          native
            ? messagesResponse(result.message, alias, id)
            : chatResponse(result.message, alias, id),
        );
    } catch {
      result.error = deadline.aborted
        ? timeoutFailure
        : controller.signal.aborted
          ? { status: 499, type: "aborted", message: "Client disconnected" }
          : genericFailure;
      if (!res.destroyed) {
        if (!res.headersSent) sendError(res, result.error, native);
        else
          res.write(
            formatEvent({
              event: native ? "error" : undefined,
              data: errorBody(result.error, native),
            }),
          );
      }
    } finally {
      controller.abort();
      active.delete(controller);
      req.off("aborted", abort);
      res.off("close", abort);
      res.end();
      const usage = result.message?.usage;
      options.log({
        event: "pi_inference.request",
        request_id: id,
        litellm_request_id: safeId(req.headers["x-litellm-call-id"]),
        trace_id: trace.traceId,
        span_id: trace.spanId,
        parent_span_id: trace.parentSpanId,
        protocol: native ? "anthropic-messages" : "openai-completions",
        alias: result.alias,
        provider: result.provider,
        model: result.model,
        provider_api: result.api,
        status: result.error?.status ?? 200,
        error_type: result.error?.type,
        duration_ms: Math.round(performance.now() - started),
        ttft_ms:
          result.firstTokenMs === undefined
            ? undefined
            : Math.round(result.firstTokenMs),
        provider_requests: result.requests,
        upstream_status: result.upstreamStatus,
        upstream_request_id: result.upstreamId,
        upstream_response_id: safeId(result.message?.responseId),
        usage: usage
          ? {
              input: usage.input,
              output: usage.output,
              cache_read: usage.cacheRead,
              cache_write: usage.cacheWrite,
              reasoning: usage.reasoning,
            }
          : undefined,
      });
    }
  }

  return {
    server,
    abortAll: () => {
      for (const controller of active) controller.abort();
    },
  };
}

function authorized(req: IncomingMessage, key: string): boolean {
  const authorization = req.headers.authorization;
  const candidate = authorization?.startsWith("Bearer ")
    ? authorization.slice(7)
    : req.headers["x-api-key"];
  if (typeof candidate !== "string" || !key) return false;
  const left = Buffer.from(candidate);
  const right = Buffer.from(key);
  return left.length === right.length && timingSafeEqual(left, right);
}

async function readBody(
  req: IncomingMessage,
  maxBytes: number,
  signal: AbortSignal,
): Promise<Result<unknown>> {
  if (
    !req.headers["content-type"]
      ?.split(";")[0]
      ?.trim()
      .toLowerCase()
      .endsWith("/json")
  )
    return {
      ok: false,
      error: {
        status: 415,
        type: "invalid_request_error",
        message: "Content-Type must be application/json",
      },
    };
  const chunks: Buffer[] = [];
  const size = { bytes: 0 };
  for await (const chunk of withDeadline(req, signal)) {
    const buffer = Buffer.isBuffer(chunk)
      ? chunk
      : Buffer.from(chunk as string);
    size.bytes += buffer.length;
    if (size.bytes > maxBytes)
      return {
        ok: false,
        error: {
          status: 413,
          type: "invalid_request_error",
          message: "Request body exceeds backend limit",
        },
      };
    chunks.push(buffer);
  }
  try {
    return {
      ok: true,
      value: JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown,
    };
  } catch {
    return {
      ok: false,
      error: {
        status: 400,
        type: "invalid_request_error",
        message: "Invalid JSON",
      },
    };
  }
}

async function* withDeadline<T>(
  source: AsyncIterable<T>,
  signal: AbortSignal,
): AsyncGenerator<T> {
  const iterator = source[Symbol.asyncIterator]();
  while (true) {
    signal.throwIfAborted();
    const abort = new AbortController();
    try {
      const next = await Promise.race([
        iterator.next(),
        once(signal, "abort", { signal: abort.signal }).then(() => {
          throw new Error("aborted");
        }),
      ]);
      if (next.done) return;
      yield next.value;
    } finally {
      abort.abort();
    }
  }
}

async function writeEvent(
  res: ServerResponse,
  frame: WireEvent,
  signal: AbortSignal,
): Promise<void> {
  if (!res.write(formatEvent(frame))) await once(res, "drain", { signal });
}

function formatEvent(frame: WireEvent): string {
  return `${frame.event ? `event: ${frame.event}\n` : ""}data: ${frame.data === "[DONE]" ? "[DONE]" : JSON.stringify(frame.data)}\n\n`;
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

function errorBody(error: ApiError, native: boolean): unknown {
  return {
    ...(native ? { type: "error" } : {}),
    error: { type: error.type, message: error.message },
  };
}

function sendError(
  res: ServerResponse,
  error: ApiError,
  native: boolean,
): void {
  sendJson(res, error.status, errorBody(error, native));
}

function providerFailure(status: number | undefined): ApiError {
  if (status === 429)
    return {
      status: 429,
      type: "rate_limit_error",
      message: "Provider rate limit reached",
    };
  return genericFailure;
}

function safeId(value: string | string[] | undefined): string | undefined {
  return typeof value === "string" && /^[A-Za-z0-9_.:-]{1,128}$/.test(value)
    ? value
    : undefined;
}

function traceContext(value: string | string[] | undefined) {
  const match =
    typeof value === "string"
      ? /^00-([a-f0-9]{32})-([a-f0-9]{16})-([a-f0-9]{2})$/.exec(value)
      : null;
  const valid = match && !/^0+$/.test(match[1]!) && !/^0+$/.test(match[2]!);
  const traceId = valid ? match[1]! : randomBytes(16).toString("hex");
  const spanId = randomBytes(8).toString("hex");
  return {
    traceId,
    spanId,
    parentSpanId: valid ? match[2] : undefined,
    header: `00-${traceId}-${spanId}-${valid ? match[3] : "00"}`,
  };
}
