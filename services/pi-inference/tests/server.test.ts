import assert from "node:assert/strict";
import { EventEmitter, once } from "node:events";
import { createServer, request, type Server } from "node:http";
import { test } from "node:test";
import {
  createModels,
  createProvider,
  envApiKeyAuth,
  fauxProvider,
  createAssistantMessageEventStream,
  type Api,
  type StreamOptions,
  type Model,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { anthropicMessagesApi } from "@earendil-works/pi-ai/api/anthropic-messages.lazy";
import {
  builtinModels,
  getBuiltinModel,
} from "@earendil-works/pi-ai/providers/all";
import { createInferenceServer } from "../src/server.js";

const internalKey = "internal-test-key-never-forward-to-provider";
const providerKey = "provider-test-key-never-return-to-caller";

async function listen(server: Server): Promise<string> {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  assert(address && typeof address === "object");
  return `http://127.0.0.1:${address.port}`;
}

async function close(server: Server) {
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );
}

test("Google Chat reaches its provider without an unsupported custom fetch", async (t) => {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    throw new Error("Fixture blocks provider network");
  };
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  const models = builtinModels({
    authContext: {
      env: async (name) =>
        name === "GEMINI_API_KEY" ? "fixture-key" : undefined,
      fileExists: async () => false,
    },
  });
  const model = getBuiltinModel("google", "gemini-2.5-flash");
  const backend = createInferenceServer({
    apiKey: internalKey,
    runtime: { models, routes: new Map([["gemini", model]]) },
    log: () => {},
  });
  const url = await listen(backend.server);
  t.after(() => close(backend.server));
  const port = Number(new URL(url).port);
  const status = await new Promise<number | undefined>((resolve) => {
    const client = request(
      {
        port,
        method: "POST",
        path: "/v1/chat/completions",
        headers: {
          authorization: `Bearer ${internalKey}`,
          "content-type": "application/json",
        },
      },
      (res) => {
        res.resume();
        res.on("end", () => resolve(res.statusCode));
      },
    );
    client.end(
      JSON.stringify({
        model: "gemini",
        messages: [{ role: "user", content: "hi" }],
      }),
    );
  });
  assert.equal(status, 502);
  assert(calls > 0, "Google adapter should reach the provider transport");
});

test("Chat hides provider error details while native Messages preserves them", async (t) => {
  const upstream = createServer((_req, res) => {
    res.writeHead(401, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        type: "error",
        error: {
          type: "authentication_error",
          message: `Incorrect API key: ${providerKey}`,
        },
      }),
    );
  });
  const upstreamUrl = await listen(upstream);
  t.after(() => close(upstream));
  const model = {
    ...getBuiltinModel("anthropic", "claude-haiku-4-5"),
    baseUrl: upstreamUrl,
  };
  const models = createModels({
    authContext: {
      env: async (name) =>
        name === "ANTHROPIC_API_KEY" ? providerKey : undefined,
      fileExists: async () => false,
    },
  });
  models.setProvider(
    createProvider({
      id: "anthropic",
      auth: { apiKey: envApiKeyAuth("Anthropic", ["ANTHROPIC_API_KEY"]) },
      models: [model],
      api: anthropicMessagesApi(),
    }),
  );
  const backend = createInferenceServer({
    apiKey: internalKey,
    runtime: { models, routes: new Map([["claude", model]]) },
    log: () => {},
  });
  const url = await listen(backend.server);
  t.after(() => close(backend.server));
  const response = await fetch(`${url}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${internalKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "claude",
      messages: [{ role: "user", content: "hi" }],
    }),
  });
  assert.equal(response.status, 401);
  assert.doesNotMatch(await response.text(), new RegExp(providerKey));
  const native = await fetch(`${url}/v1/messages`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${internalKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "claude",
      max_tokens: 128,
      messages: [{ role: "user", content: "hi" }],
    }),
  });
  assert.equal(native.status, 401);
  assert.match(await native.text(), new RegExp(providerKey));
});

test("HTTP ingress crosses real Pi adapter with streaming, isolated auth, usage and correlated traces", async (t) => {
  const captured: {
    authorization?: string;
    trace?: string;
    body?: Record<string, unknown>;
  } = {};
  const upstream = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    captured.body = JSON.parse(Buffer.concat(chunks).toString()) as Record<
      string,
      unknown
    >;
    captured.authorization = req.headers.authorization;
    captured.trace = req.headers.traceparent as string;
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "x-request-id": "upstream-1",
    });
    for (const chunk of [
      {
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "Hello " },
            finish_reason: null,
          },
        ],
      },
      {
        choices: [
          { index: 0, delta: { content: "world" }, finish_reason: null },
        ],
      },
      {
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        usage: {
          prompt_tokens: 30,
          completion_tokens: 5,
          total_tokens: 35,
          prompt_tokens_details: { cached_tokens: 20 },
        },
      },
    ])
      res.write(
        `data: ${JSON.stringify({ id: "provider-response-1", object: "chat.completion.chunk", created: 0, model: "stub-model", ...chunk })}\n\n`,
      );
    res.end("data: [DONE]\n\n");
  });
  const upstreamUrl = await listen(upstream);
  t.after(() => close(upstream));
  const model: Model<"openai-completions"> = {
    id: "stub-model",
    name: "Stub",
    provider: "stub",
    api: "openai-completions",
    baseUrl: `${upstreamUrl}/v1`,
    reasoning: false,
    input: ["text"],
    contextWindow: 4096,
    maxTokens: 1024,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  };
  const models = createModels({
    authContext: {
      env: async (name) => (name === "STUB_KEY" ? providerKey : undefined),
      fileExists: async () => false,
    },
  });
  models.setProvider(
    createProvider({
      id: "stub",
      auth: { apiKey: envApiKeyAuth("Stub", ["STUB_KEY"]) },
      models: [model],
      api: openAICompletionsApi(),
    }),
  );
  const logs: Record<string, unknown>[] = [];
  const backend = createInferenceServer({
    apiKey: internalKey,
    runtime: { models, routes: new Map([["public-model", model]]) },
    log: (record) => logs.push(record),
  });
  const url = await listen(backend.server);
  t.after(() => close(backend.server));
  const traceId = "1234567890abcdef1234567890abcdef";
  const response = await fetch(`${url}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${internalKey}`,
      "content-type": "application/json",
      "x-request-id": "request-1",
      traceparent: `00-${traceId}-1234567890abcdef-01`,
    },
    body: JSON.stringify({
      model: "public-model",
      messages: [{ role: "user", content: "private prompt do not log" }],
      stream: true,
      stream_options: { include_usage: true },
    }),
  });
  assert.equal(response.status, 200);
  const wire = await response.text();
  const events = wire
    .split("\n\n")
    .filter(Boolean)
    .map((line) => line.slice("data: ".length));
  assert.equal(events.at(-1), "[DONE]");
  const chunks = events.slice(0, -1).map((value) => JSON.parse(value));
  assert.equal(
    chunks
      .flatMap((chunk) => chunk.choices)
      .map((choice) => choice.delta.content ?? "")
      .join(""),
    "Hello world",
  );
  assert.equal(chunks.at(-1).usage.prompt_tokens, 30);
  assert.equal(chunks.at(-1).usage.completion_tokens, 5);
  assert.equal(chunks.at(-1).usage.prompt_tokens_details.cached_tokens, 20);
  assert.equal(captured.authorization, `Bearer ${providerKey}`);
  assert.equal(captured.body?.model, "stub-model");
  assert.equal(captured.body?.stream, true);
  assert.match(
    captured.trace ?? "",
    new RegExp(`^00-${traceId}-[a-f0-9]{16}-01$`),
  );
  assert.equal(logs.length, 1);
  assert.equal(logs[0]?.trace_id, traceId);
  assert.equal(logs[0]?.upstream_request_id, "upstream-1");
  assert.equal(logs[0]?.status, 200);
  assert.deepEqual(logs[0]?.usage, {
    input: 10,
    output: 5,
    cache_read: 20,
    cache_write: 0,
    reasoning: 0,
  });
  assert(!JSON.stringify(logs).includes(providerKey));
  assert(!JSON.stringify(logs).includes("private prompt"));
  assert(!wire.includes(providerKey));

  const nonstream = await fetch(`${url}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${internalKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "public-model",
      messages: [{ role: "user", content: "hi" }],
    }),
  });
  assert.equal(nonstream.status, 200);
  const complete = (await nonstream.json()) as {
    choices: { message: { content: string } }[];
    usage: { total_tokens: number };
  };
  assert.equal(complete.choices[0]?.message.content, "Hello world");
  assert.equal(complete.usage.total_tokens, 35);
});

test("health is local; catalog and inference require backend auth, allowlisted models and supported endpoints", async (t) => {
  const models = createModels();
  const backend = createInferenceServer({
    apiKey: internalKey,
    runtime: { models, routes: new Map() },
    log: () => {},
  });
  const url = await listen(backend.server);
  t.after(() => close(backend.server));
  assert.equal((await fetch(`${url}/health/liveliness`)).status, 200);
  assert.equal((await fetch(`${url}/v1/models`)).status, 401);
  assert.equal(
    (
      await fetch(`${url}/v1/models`, {
        headers: { authorization: `Bearer ${internalKey}` },
      })
    ).status,
    200,
  );
  assert.equal(
    (
      await fetch(`${url}/v1/responses`, {
        method: "POST",
        headers: { authorization: `Bearer ${internalKey}` },
      })
    ).status,
    501,
  );
  const unknown = await fetch(`${url}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${internalKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "not-enabled",
      messages: [{ role: "user", content: "hi" }],
    }),
  });
  assert.equal(unknown.status, 404);
});

test("request validation bounds bodies and rejects malformed JSON without dispatch", async (t) => {
  const models = createModels();
  const logs: Record<string, unknown>[] = [];
  const backend = createInferenceServer({
    apiKey: internalKey,
    maxBodyBytes: 32,
    runtime: { models, routes: new Map() },
    log: (record) => logs.push(record),
  });
  const url = await listen(backend.server);
  t.after(() => close(backend.server));
  const headers = {
    authorization: `Bearer ${internalKey}`,
    "content-type": "application/json",
  };
  const huge = await fetch(`${url}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify({ model: "x", messages: "a".repeat(100) }),
  });
  assert.equal(huge.status, 413);
  await huge.text();
  const malformed = await fetch(`${url}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: "{",
  });
  assert.equal(malformed.status, 400);
  await malformed.text();
  assert(logs.every((record) => record.provider_requests === 0));
});

test("provider setup errors are HTTP failures, never successful completion bodies", async (t) => {
  const models = createModels();
  const faux = fauxProvider();
  models.setProvider(faux.provider);
  const logs: Record<string, unknown>[] = [];
  const backend = createInferenceServer({
    apiKey: internalKey,
    runtime: { models, routes: new Map([["empty", faux.getModel()]]) },
    log: (record) => logs.push(record),
  });
  const url = await listen(backend.server);
  t.after(() => close(backend.server));
  const response = await fetch(`${url}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${internalKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "empty",
      messages: [{ role: "user", content: "hi" }],
    }),
  });
  assert.equal(response.status, 502);
  assert(!(await response.text()).includes("choices"));
  assert.equal(logs[0]?.status, 502);
});

test("provider policy refusals remain HTTP 200 results and never trigger infrastructure retry", async (t) => {
  const models = createModels();
  const base = fauxProvider().getModel();
  const model = { ...base, api: "anthropic-messages" };
  const refusal = () => {
    const events = createAssistantMessageEventStream();
    events.push({
      type: "error",
      reason: "error",
      error: {
        role: "assistant",
        content: [{ type: "text", text: "Cannot help with that" }],
        provider: model.provider,
        model: model.id,
        api: model.api,
        timestamp: 0,
        usage: {
          input: 2,
          output: 3,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 5,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "error",
        rawStopReason: "refusal",
        errorMessage: "private provider explanation",
      },
    });
    return events;
  };
  models.setProvider(
    createProvider({
      id: model.provider,
      auth: { apiKey: { name: "test", resolve: async () => ({ auth: {} }) } },
      models: [model],
      api: { stream: refusal, streamSimple: refusal },
    }),
  );
  const logs: Record<string, unknown>[] = [];
  const backend = createInferenceServer({
    apiKey: internalKey,
    runtime: { models, routes: new Map([["claude", model]]) },
    log: (record) => logs.push(record),
  });
  const url = await listen(backend.server);
  t.after(() => close(backend.server));
  for (const stream of [false, true]) {
    const response = await fetch(`${url}/v1/messages`, {
      method: "POST",
      headers: { "x-api-key": internalKey, "content-type": "application/json" },
      body: JSON.stringify({
        model: "claude",
        max_tokens: 32,
        messages: [{ role: "user", content: "hi" }],
        stream,
      }),
    });
    assert.equal(response.status, 200);
    const output = await response.text();
    assert(output.includes('\"stop_reason\":\"refusal\"'));
    assert(output.includes("Cannot help with that"));
    assert(!output.includes("private provider explanation"));
    if (stream) assert(output.includes("event: message_stop"));
  }
  assert.equal(logs.length, 2);
  assert(logs.every((record) => record.status === 200));
});

test(
  "backend deadline bounds a noncooperative provider and releases capacity",
  { timeout: 2000 },
  async (t) => {
    const models = createModels();
    const model = fauxProvider().getModel();
    const calls: StreamOptions[] = [];
    const hang = (
      _model: Model<Api>,
      _context: unknown,
      options?: StreamOptions,
    ) => {
      if (options) calls.push(options);
      return createAssistantMessageEventStream();
    };
    models.setProvider(
      createProvider({
        id: model.provider,
        auth: { apiKey: { name: "test", resolve: async () => ({ auth: {} }) } },
        models: [model],
        api: { stream: hang, streamSimple: hang },
      }),
    );
    const logs: Record<string, unknown>[] = [];
    const backend = createInferenceServer({
      apiKey: internalKey,
      timeoutMs: 30,
      maxInflight: 1,
      runtime: { models, routes: new Map([["slow", model]]) },
      log: (record) => logs.push(record),
    });
    const url = await listen(backend.server);
    t.after(() => close(backend.server));
    for (const attempt of [1, 2]) {
      const response = await fetch(`${url}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${internalKey}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "slow",
          messages: [{ role: "user", content: `attempt ${attempt}` }],
        }),
      });
      assert.equal(response.status, 504);
      await response.text();
    }
    assert.equal(calls.length, 2);
    assert(calls.every((call) => call.signal?.aborted));
    assert(logs.every((record) => record.status === 504));
  },
);

test(
  "client disconnect cancels provider work and concurrent excess receives 429",
  { timeout: 3000 },
  async (t) => {
    const models = createModels();
    const model = fauxProvider().getModel();
    const events = new EventEmitter();
    const hang = (
      _model: Model<Api>,
      _context: unknown,
      options?: StreamOptions,
    ) => {
      options?.signal?.addEventListener("abort", () => events.emit("aborted"), {
        once: true,
      });
      events.emit("started");
      return createAssistantMessageEventStream();
    };
    models.setProvider(
      createProvider({
        id: model.provider,
        auth: { apiKey: { name: "test", resolve: async () => ({ auth: {} }) } },
        models: [model],
        api: { stream: hang, streamSimple: hang },
      }),
    );
    const backend = createInferenceServer({
      apiKey: internalKey,
      timeoutMs: 2000,
      maxInflight: 1,
      runtime: { models, routes: new Map([["slow", model]]) },
      log: () => {},
    });
    const url = await listen(backend.server);
    t.after(() => close(backend.server));
    const client = new AbortController();
    const started = once(events, "started");
    const request = {
      method: "POST",
      headers: {
        authorization: `Bearer ${internalKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "slow",
        messages: [{ role: "user", content: "hi" }],
      }),
    };
    const first = fetch(`${url}/v1/chat/completions`, {
      ...request,
      signal: client.signal,
    });
    const rejected = assert.rejects(first, /abort/i);
    await started;
    const excess = await fetch(`${url}/v1/chat/completions`, request);
    assert.equal(excess.status, 429);
    const aborted = once(events, "aborted");
    client.abort();
    await aborted;
    await rejected;
  },
);
