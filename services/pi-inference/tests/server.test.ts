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

test("models without the Anthropic Messages API are rejected before reaching their provider", async (t) => {
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
        path: "/v1/messages",
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
        max_tokens: 64,
        messages: [{ role: "user", content: "hi" }],
      }),
    );
  });
  assert.equal(status, 400);
  assert.equal(calls, 0);
});

test("provider errors never return provider credentials to clients", async (t) => {
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
  const nativeError = await native.text();
  assert.doesNotMatch(nativeError, new RegExp(providerKey));
  assert.match(nativeError, /Provider request failed/);
});

test("HTTP ingress crosses real Pi adapter with streaming, isolated auth, usage and correlated traces", async (t) => {
  const captured: {
    apiKey?: string;
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
    captured.apiKey = req.headers["x-api-key"] as string;
    captured.trace = req.headers.traceparent as string;
    const usage = {
      input_tokens: 10,
      output_tokens: 5,
      cache_read_input_tokens: 20,
      cache_creation_input_tokens: 0,
    };
    const message = {
      id: "provider-response-1",
      type: "message",
      role: "assistant",
      model: "stub-model",
      content: [{ type: "text", text: "Hello world" }],
      stop_reason: "end_turn",
      stop_sequence: null,
      usage,
    };
    if (!captured.body.stream) {
      res.writeHead(200, {
        "content-type": "application/json",
        "x-request-id": "upstream-1",
      });
      res.end(JSON.stringify(message));
      return;
    }
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "x-request-id": "upstream-1",
    });
    for (const data of [
      {
        type: "message_start",
        message: {
          ...message,
          content: [],
          stop_reason: null,
          usage: { ...usage, output_tokens: 1 },
        },
      },
      {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "Hello " },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: "world" },
      },
      { type: "content_block_stop", index: 0 },
      {
        type: "message_delta",
        delta: { stop_reason: "end_turn", stop_sequence: null },
        usage,
      },
      { type: "message_stop" },
    ])
      res.write(`event: ${data.type}\ndata: ${JSON.stringify(data)}\n\n`);
    res.end();
  });
  const upstreamUrl = await listen(upstream);
  t.after(() => close(upstream));
  const model: Model<"anthropic-messages"> = {
    id: "stub-model",
    name: "Stub",
    provider: "stub",
    api: "anthropic-messages",
    baseUrl: upstreamUrl,
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
      api: anthropicMessagesApi(),
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
  const response = await fetch(`${url}/v1/messages`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${internalKey}`,
      "content-type": "application/json",
      "x-request-id": "request-1",
      traceparent: `00-${traceId}-1234567890abcdef-01`,
    },
    body: JSON.stringify({
      model: "public-model",
      max_tokens: 64,
      messages: [{ role: "user", content: "private prompt do not log" }],
      stream: true,
    }),
  });
  assert.equal(response.status, 200);
  const wire = await response.text();
  const events = wire
    .split("\n\n")
    .filter(Boolean)
    .map((frame) => JSON.parse(frame.slice(frame.indexOf("data: ") + 6)));
  assert.equal(events.at(-1).type, "message_stop");
  assert.equal(
    events
      .filter((event) => event.type === "content_block_delta")
      .map((event) => event.delta.text ?? "")
      .join(""),
    "Hello world",
  );
  const finalUsage = events
    .filter((event: { type: string }) => event.type === "message_delta")
    .at(-1).usage;
  assert.equal(finalUsage.input_tokens, 10);
  assert.equal(finalUsage.output_tokens, 5);
  assert.equal(finalUsage.cache_read_input_tokens, 20);
  assert.equal(captured.apiKey, providerKey);
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
    reasoning: undefined,
  });
  assert(!JSON.stringify(logs).includes(providerKey));
  assert(!JSON.stringify(logs).includes("private prompt"));
  assert(!wire.includes(providerKey));

  const nonstream = await fetch(`${url}/v1/messages`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${internalKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "public-model",
      max_tokens: 64,
      messages: [{ role: "user", content: "hi" }],
    }),
  });
  assert.equal(nonstream.status, 200);
  const complete = (await nonstream.json()) as {
    content: { type: string; text: string }[];
    usage: { input_tokens: number; output_tokens: number };
  };
  assert.equal(complete.content[0]?.text, "Hello world");
  assert.equal(complete.usage.input_tokens, 10);
  assert.equal(complete.usage.output_tokens, 5);
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
  assert.equal(
    (
      await fetch(`${url}/v1/chat/completions`, {
        method: "POST",
        headers: { authorization: `Bearer ${internalKey}` },
      })
    ).status,
    501,
  );
  const unknown = await fetch(`${url}/v1/messages`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${internalKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "not-enabled",
      max_tokens: 64,
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
  const huge = await fetch(`${url}/v1/messages`, {
    method: "POST",
    headers,
    body: JSON.stringify({ model: "x", messages: "a".repeat(100) }),
  });
  assert.equal(huge.status, 413);
  await huge.text();
  const malformed = await fetch(`${url}/v1/messages`, {
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
    runtime: {
      models,
      routes: new Map([
        ["empty", { ...faux.getModel(), api: "anthropic-messages" }],
      ]),
    },
    log: (record) => logs.push(record),
  });
  const url = await listen(backend.server);
  t.after(() => close(backend.server));
  const response = await fetch(`${url}/v1/messages`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${internalKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "empty",
      max_tokens: 64,
      messages: [{ role: "user", content: "hi" }],
    }),
  });
  assert.equal(response.status, 502);
  assert.equal(((await response.json()) as { type: string }).type, "error");
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
    const model = { ...fauxProvider().getModel(), api: "anthropic-messages" };
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
      const response = await fetch(`${url}/v1/messages`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${internalKey}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "slow",
          max_tokens: 64,
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
    const model = { ...fauxProvider().getModel(), api: "anthropic-messages" };
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
        max_tokens: 64,
        messages: [{ role: "user", content: "hi" }],
      }),
    };
    const first = fetch(`${url}/v1/messages`, {
      ...request,
      signal: client.signal,
    });
    const rejected = assert.rejects(first, /abort/i);
    await started;
    const excess = await fetch(`${url}/v1/messages`, request);
    assert.equal(excess.status, 429);
    assert.equal(excess.headers.get("retry-after"), "1");
    assert.equal(excess.headers.get("x-openorange-pi-slot-at-capacity"), "1");
    assert.equal(
      excess.headers.get("x-openorange-subscription-exhausted-until"),
      null,
    );
    assert.equal(
      ((await excess.json()) as { error: { code: string } }).error.code,
      "slot_at_capacity",
    );
    const aborted = once(events, "aborted");
    client.abort();
    await aborted;
    await rejected;
  },
);

async function quotaFixture(
  t: { after: (fn: () => unknown) => void },
  answer: (model: string) => {
    status: number;
    headers?: Record<string, string>;
  },
) {
  const calls: string[] = [];
  const upstream = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    const body = JSON.parse(Buffer.concat(chunks).toString()) as {
      model: string;
    };
    calls.push(body.model);
    const { status, headers = {} } = answer(body.model);
    res.writeHead(status, { "content-type": "application/json", ...headers });
    res.end(
      JSON.stringify(
        status === 200
          ? {
              id: "provider-response",
              type: "message",
              role: "assistant",
              model: body.model,
              content: [{ type: "text", text: "ok" }],
              stop_reason: "end_turn",
              stop_sequence: null,
              usage: { input_tokens: 1, output_tokens: 1 },
            }
          : {
              type: "error",
              error: {
                type: status === 529 ? "overloaded_error" : "rate_limit_error",
                message: "provider said no",
              },
            },
      ),
    );
  });
  const upstreamUrl = await listen(upstream);
  t.after(() => close(upstream));
  const stub = (id: string): Model<"anthropic-messages"> => ({
    id,
    name: id,
    provider: "stub",
    api: "anthropic-messages",
    baseUrl: upstreamUrl,
    reasoning: false,
    input: ["text"],
    contextWindow: 4096,
    maxTokens: 1024,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  });
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
      models: [stub("stub-opus"), stub("stub-sonnet")],
      api: anthropicMessagesApi(),
    }),
  );
  const clock = { now: 1_800_000_000_000 };
  const logs: Record<string, unknown>[] = [];
  const backend = createInferenceServer({
    apiKey: internalKey,
    now: () => clock.now,
    runtime: {
      models,
      routes: new Map([
        ["opus", stub("stub-opus")],
        ["sonnet", stub("stub-sonnet")],
      ]),
    },
    log: (record) => logs.push(record),
  });
  const url = await listen(backend.server);
  t.after(() => close(backend.server));
  const send = (model: string, stream = false) =>
    fetch(`${url}/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${internalKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        max_tokens: 64,
        messages: [{ role: "user", content: "hi" }],
        stream,
      }),
    });
  return { calls, clock, logs, send };
}

test("an out-of-quota account answers that model at once until its reset, without calling Anthropic", async (t) => {
  const reset = 1_800_003_600;
  const nextReset = reset + 18_000;
  let answerReset = reset;
  const { calls, clock, logs, send } = await quotaFixture(t, (model) =>
    model === "stub-opus"
      ? {
          status: 429,
          headers: {
            "retry-after": "3600",
            "anthropic-ratelimit-unified-status": "rejected",
            "anthropic-ratelimit-unified-reset": String(answerReset),
            "anthropic-ratelimit-unified-representative-claim": "five_hour",
          },
        }
      : { status: 200 },
  );
  for (const stream of [false, true, false]) {
    const response = await send("opus", stream);
    assert.equal(response.status, 429);
    assert.equal(
      response.headers.get("x-openorange-subscription-exhausted-until"),
      String(reset),
    );
    assert.equal(response.headers.get("retry-after"), null);
    assert.deepEqual(await response.json(), {
      type: "error",
      error: {
        type: "rate_limit_error",
        code: "subscription_exhausted",
        message:
          "Claude subscription usage limit reached (five_hour); it resets at 2027-01-15T09:00:00.000Z",
      },
    });
  }
  assert.deepEqual(calls, ["stub-opus"]);
  const sonnet = await send("sonnet");
  assert.equal(sonnet.status, 200);
  await sonnet.text();
  assert.deepEqual(calls, ["stub-opus", "stub-sonnet"]);
  clock.now = reset * 1000;
  answerReset = nextReset;
  const again = await send("opus");
  assert.equal(again.status, 429);
  assert.equal(
    again.headers.get("x-openorange-subscription-exhausted-until"),
    String(nextReset),
  );
  assert.deepEqual(calls, ["stub-opus", "stub-sonnet", "stub-opus"]);
  assert.deepEqual(
    logs.map((record) => [
      record.alias,
      record.status,
      record.provider_requests,
      record.upstream_status,
      record.upstream_unified_status,
      record.subscription_exhausted_until,
    ]),
    [
      ["opus", 429, 1, 429, "rejected", reset],
      ["opus", 429, 0, undefined, undefined, reset],
      ["opus", 429, 0, undefined, undefined, reset],
      ["sonnet", 200, 1, 200, undefined, undefined],
      ["opus", 429, 1, 429, "rejected", nextReset],
    ],
  );
});

test("a plain 429, a 529 and a rejection already reset stay retryable and mark nothing", async (t) => {
  const answers: { status: number; headers: Record<string, string> }[] = [
    { status: 429, headers: { "retry-after": "2" } },
    {
      status: 429,
      headers: {
        "anthropic-ratelimit-unified-status": "allowed",
        "anthropic-ratelimit-unified-reset": "1800003600",
      },
    },
    {
      status: 429,
      headers: {
        "anthropic-ratelimit-unified-status": "rejected",
        "anthropic-ratelimit-unified-reset": "1799999000",
      },
    },
    { status: 529, headers: { "retry-after": "1" } },
  ];
  let next = 0;
  const { calls, logs, send } = await quotaFixture(t, () => answers[next]!);
  for (const [index, answer] of answers.entries()) {
    next = index;
    for (const stream of [false, true]) {
      const response = await send("opus", stream);
      assert.equal(response.status, answer.status);
      assert.equal(
        response.headers.get("x-openorange-subscription-exhausted-until"),
        null,
      );
      assert.equal(
        response.headers.get("retry-after"),
        answer.headers["retry-after"] ?? null,
      );
      const body = (await response.json()) as { error: { code?: string } };
      assert.equal(body.error.code, undefined);
    }
  }
  assert.equal(calls.length, answers.length * 2);
  assert(
    logs.every((record) => record.subscription_exhausted_until === undefined),
  );
});
