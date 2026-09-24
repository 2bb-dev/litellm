import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
import test from "node:test";
import {
  createModels,
  createProvider,
  Type,
  type AssistantMessage,
  type Context,
  type Model,
} from "@earendil-works/pi-ai";
import { anthropicMessagesApi } from "@earendil-works/pi-ai/api/anthropic-messages.lazy";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";
import { loadRuntime } from "../src/runtime.js";
import { createInferenceServer } from "../src/server.js";

// Parity contract for claude-opus-5-5 on the native Messages route. A real
// pi-ai 0.87.1 client (the same library Pi agents use) talks to the sidecar
// and a loopback server stands in for Anthropic. The body that reaches
// Anthropic must be the client's body; the only allowed differences are the
// documented transport and OAuth transforms asserted below.

const internalKey = "internal-parity-key";
const alias = "anthropic/claude-opus-5-5";
const opus = builtinModels({}).getModel("anthropic", "claude-opus-5-5");
assert(opus, "pi-ai must ship claude-opus-5-5");

type Captured = {
  path: string;
  headers: IncomingHttpHeaders;
  body: Record<string, unknown>;
};

const sse = (data: { type: string; [key: string]: unknown }) =>
  `event: ${data.type}\ndata: ${JSON.stringify(data)}\n\n`;

function upstreamResponse(toolName: string): string {
  return [
    {
      type: "message_start",
      message: {
        id: "msg_upstream",
        type: "message",
        role: "assistant",
        model: "claude-opus-5-5",
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 12,
          output_tokens: 1,
          cache_read_input_tokens: 100,
          cache_creation_input_tokens: 30,
          cache_creation: {
            ephemeral_5m_input_tokens: 10,
            ephemeral_1h_input_tokens: 20,
          },
        },
      },
    },
    { type: "ping" },
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "thinking", thinking: "", signature: "" },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "signature_delta", signature: "sig-opus-5-5" },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "content_block_start",
      index: 1,
      content_block: { type: "text", text: "" },
    },
    {
      type: "content_block_delta",
      index: 1,
      delta: { type: "text_delta", text: "Checking." },
    },
    { type: "content_block_stop", index: 1 },
    {
      type: "content_block_start",
      index: 2,
      content_block: {
        type: "tool_use",
        id: "toolu_1",
        name: toolName,
        input: {},
      },
    },
    {
      type: "content_block_delta",
      index: 2,
      delta: { type: "input_json_delta", partial_json: '{"city":"Paris"}' },
    },
    { type: "content_block_stop", index: 2 },
    {
      type: "message_delta",
      delta: { stop_reason: "tool_use", stop_sequence: null },
      usage: { output_tokens: 40 },
    },
    { type: "message_stop" },
  ]
    .map(sse)
    .join("");
}

async function listen(server: Server): Promise<string> {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  assert(address && typeof address === "object");
  return `http://127.0.0.1:${address.port}`;
}

async function close(server: Server) {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

const betas = (headers: IncomingHttpHeaders) =>
  new Set(
    String(headers["anthropic-beta"] ?? "")
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean),
  );

const context: Context = {
  systemPrompt: "You are a weather agent. Keep answers short.",
  tools: [
    {
      name: "lookup_weather",
      description: "Look up the weather",
      parameters: Type.Object({ city: Type.String() }),
    },
  ],
  messages: [{ role: "user", content: "Weather in Paris?", timestamp: 1 }],
};

async function run(providerKey: string, clientBetas?: string) {
  const received: Captured[] = [];
  const oauth = providerKey.includes("sk-ant-oat");
  const upstream = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    const body = JSON.parse(Buffer.concat(chunks).toString()) as Record<
      string,
      unknown
    >;
    received.push({
      path: new URL(req.url ?? "", "http://fixture.invalid").pathname,
      headers: req.headers,
      body,
    });
    const tools = (body.tools ?? []) as { name: string }[];
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "request-id": "req_upstream",
    });
    res.end(upstreamResponse(tools[0]?.name ?? "lookup_weather"));
  });
  const upstreamUrl = await listen(upstream);
  const runtime = loadRuntime(
    {
      models: [
        {
          alias,
          provider: "anthropic",
          model: "claude-opus-5-5",
          baseUrl: upstreamUrl,
        },
      ],
    },
    {
      authContext: {
        env: async (name) =>
          name === "ANTHROPIC_API_KEY" ? providerKey : undefined,
        fileExists: async () => false,
      },
    },
  );
  assert(runtime.ok, runtime.ok ? "" : runtime.error.message);
  const backend = createInferenceServer({
    apiKey: internalKey,
    runtime: runtime.value,
    log: () => {},
  });
  const backendUrl = await listen(backend.server);

  const clientModel: Model<"anthropic-messages"> = {
    ...(opus as Model<"anthropic-messages">),
    id: alias,
    baseUrl: backendUrl,
  };
  const client = createModels();
  client.setProvider(
    createProvider({
      id: "anthropic",
      models: [clientModel],
      api: anthropicMessagesApi(),
      auth: {
        apiKey: {
          name: "Sidecar",
          resolve: async () => ({ auth: { apiKey: internalKey } }),
        },
      },
    }),
  );
  const sent: Record<string, unknown>[] = [];
  const complete = (ctx: Context) =>
    client.completeSimple(clientModel, ctx, {
      reasoning: "xhigh",
      maxTokens: 64000,
      ...(clientBetas ? { headers: { "anthropic-beta": clientBetas } } : {}),
      timeoutMs: 5000,
      maxRetries: 0,
      onPayload: async (payload) => {
        sent.push(structuredClone(payload) as Record<string, unknown>);
        return undefined;
      },
    });
  try {
    const first = await complete(context);
    // Second turn replays the signed thinking block and the tool call.
    const second = await complete({
      ...context,
      messages: [
        ...context.messages,
        first,
        {
          role: "toolResult",
          toolCallId: "toolu_1",
          toolName: "lookup_weather",
          content: [{ type: "text", text: "Sunny, 21C" }],
          isError: false,
          timestamp: 3,
        },
      ],
    });
    return { oauth, sent, received, first, second };
  } finally {
    await close(backend.server);
    await close(upstream);
  }
}

function assertResponse(message: AssistantMessage) {
  assert.equal(message.stopReason, "toolUse", message.errorMessage);
  const thinking = message.content.find((block) => block.type === "thinking");
  assert(thinking?.type === "thinking");
  assert.equal(thinking.thinkingSignature, "sig-opus-5-5");
  const call = message.content.find((block) => block.type === "toolCall");
  assert(call?.type === "toolCall");
  assert.equal(call.name, "lookup_weather");
  assert.deepEqual(call.arguments, { city: "Paris" });
  assert.equal(message.usage.cacheRead, 100);
  assert.equal(message.usage.cacheWrite, 30);
}

const apiKey = "sk-ant-api03-parity-fixture";
const oauthKey = "sk-ant-oat01-parity-fixture";

for (const providerKey of [apiKey, oauthKey]) {
  const mode = providerKey.includes("oat") ? "OAuth" : "API key";
  test(`claude-opus-5-5 native request reaches Anthropic unchanged (${mode})`, async () => {
    const { oauth, sent, received, first, second } = await run(providerKey);
    assertResponse(first);
    assertResponse(second);
    assert.equal(sent.length, 2);
    assert.equal(received.length, 2);
    for (const [index, outgoing] of received.entries()) {
      const incoming = sent[index]!;
      assert.equal(outgoing.path, "/v1/messages");
      const {
        betas: clientBetas = [],
        model: _model,
        ...clientBody
      } = incoming as {
        betas?: string[];
        model: string;
      } & Record<string, unknown>;
      const expectedBetas = new Set([
        ...clientBetas,
        ...(oauth ? ["claude-code-20250219", "oauth-2025-04-20"] : []),
      ]);
      assert.deepEqual(betas(outgoing.headers), expectedBetas);
      const { model, system, tools, messages, ...rest } = outgoing.body;
      assert.equal(model, "claude-opus-5-5");
      const {
        system: clientSystem,
        tools: clientTools,
        messages: clientMessages,
        ...clientRest
      } = clientBody;
      // Every field other than system/tools/messages is byte-for-byte the client's.
      assert.deepEqual(rest, clientRest);
      if (!oauth) {
        assert.deepEqual(system, clientSystem);
        assert.deepEqual(tools, clientTools);
        assert.deepEqual(messages, clientMessages);
        continue;
      }
      // OAuth: Claude Code identity first, then the client's system blocks.
      const blocks = system as { type: string; text: string }[];
      assert.equal(
        blocks[0]?.text,
        "You are Claude Code, Anthropic's official CLI for Claude.",
      );
      assert.deepEqual(blocks.slice(1), clientSystem);
      // OAuth: tool names are aliased, the rest of each tool is unchanged.
      assert.deepEqual(
        (tools as Record<string, unknown>[]).map(
          ({ name: _name, ...tool }) => tool,
        ),
        (clientTools as Record<string, unknown>[]).map(
          ({ name: _name, ...tool }) => tool,
        ),
      );
      assert.equal(
        (tools as { name: string }[])[0]?.name,
        "mcp__pi__lookup_weather",
      );
      const rename = (value: unknown) =>
        JSON.parse(
          JSON.stringify(value).replaceAll(
            '"lookup_weather"',
            '"mcp__pi__lookup_weather"',
          ),
        );
      assert.deepEqual(messages, rename(clientMessages));
    }
  });
}

for (const providerKey of [apiKey, oauthKey]) {
  const mode = providerKey.includes("oat") ? "OAuth" : "API key";
  test(`client anthropic-beta values reach Anthropic; backend-derived feature betas do not (${mode})`, async () => {
    const own = "context-1m-2025-08-07,thinking-display-updates-2026-08-18";
    const { received, first } = await run(providerKey, own);
    assertResponse(first);
    assert.deepEqual(
      betas(received[0]!.headers),
      new Set([
        "context-1m-2025-08-07",
        "thinking-display-updates-2026-08-18",
        ...(providerKey.includes("oat")
          ? ["claude-code-20250219", "oauth-2025-04-20"]
          : []),
      ]),
    );
  });
}

async function sidecar(
  handler: (
    res: import("node:http").ServerResponse,
    body: Record<string, unknown>,
  ) => void | Promise<void>,
  options: { keepAliveMs?: number } = {},
) {
  const upstream = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    await handler(
      res,
      JSON.parse(Buffer.concat(chunks).toString()) as Record<string, unknown>,
    );
  });
  const upstreamUrl = await listen(upstream);
  const runtime = loadRuntime(
    {
      models: [
        {
          alias,
          provider: "anthropic",
          model: "claude-opus-5-5",
          baseUrl: upstreamUrl,
        },
      ],
    },
    {
      authContext: {
        env: async (name) =>
          name === "ANTHROPIC_API_KEY" ? apiKey : undefined,
        fileExists: async () => false,
      },
    },
  );
  assert(runtime.ok);
  const backend = createInferenceServer({
    apiKey: internalKey,
    runtime: runtime.value,
    log: () => {},
    ...options,
  });
  const url = await listen(backend.server);
  const call = (body: Record<string, unknown>) =>
    fetch(`${url}/v1/messages`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${internalKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: alias,
        max_tokens: 1024,
        messages: [{ role: "user", content: "hi" }],
        ...body,
      }),
    });
  return {
    call,
    close: async () => {
      await close(backend.server);
      await close(upstream);
    },
  };
}

test("API-key native JSON preserves upstream response fields and unknown blocks", async () => {
  const original = {
    id: "msg_original",
    type: "message",
    role: "assistant",
    model: "claude-opus-5-5",
    content: [{ type: "future_block", value: { nested: true } }],
    stop_reason: "stop_sequence",
    stop_sequence: "END",
    stop_details: { reason: "matched" },
    usage: { input_tokens: 4, output_tokens: 2 },
  };
  let requestStream: unknown;
  const backend = await sidecar((res, body) => {
    requestStream = body.stream;
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(original));
  });
  try {
    const response = await backend.call({ stream: false });
    assert.equal(response.status, 200);
    assert.equal(requestStream, false);
    assert.deepEqual(await response.json(), original);
  } finally {
    await backend.close();
  }
});

test("API-key native SSE preserves unknown events and mid-stream errors", async () => {
  const frames = [
    sse({
      type: "message_start",
      message: {
        id: "msg_original",
        type: "message",
        role: "assistant",
        model: "claude-opus-5-5",
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 4, output_tokens: 0 },
      },
    }),
    sse({ type: "future_event", value: { nested: true } }),
    sse({
      type: "error",
      error: { type: "overloaded_error", message: "retry later" },
    }),
  ].join("");
  const backend = await sidecar((res) => {
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.end(frames);
  });
  try {
    const response = await backend.call({ stream: true });
    assert.equal(response.status, 200);
    assert.equal(await response.text(), frames);
  } finally {
    await backend.close();
  }
});

test("Anthropic error statuses, types, messages and retry-after reach the client", async () => {
  const cases = [
    {
      status: 400,
      type: "invalid_request_error",
      message: "tool_choice: type any is not supported for this model.",
    },
    { status: 401, type: "authentication_error", message: "invalid x-api-key" },
    {
      status: 429,
      type: "rate_limit_error",
      message: "Number of requests has exceeded your rate limit",
      retryAfter: "17",
    },
    { status: 529, type: "overloaded_error", message: "Overloaded" },
  ];
  for (const entry of cases) {
    const backend = await sidecar((res) => {
      res.writeHead(entry.status, {
        "content-type": "application/json",
        ...(entry.retryAfter ? { "retry-after": entry.retryAfter } : {}),
        "x-should-retry": "false",
      });
      res.end(
        JSON.stringify({
          type: "error",
          error: { type: entry.type, message: entry.message },
        }),
      );
    });
    try {
      for (const stream of [false, true]) {
        const response = await backend.call({ stream });
        assert.equal(
          response.status,
          entry.status,
          `${entry.status} stream=${stream}`,
        );
        assert.equal(
          response.headers.get("retry-after"),
          entry.retryAfter ?? null,
        );
        assert.deepEqual(await response.json(), {
          type: "error",
          error: { type: entry.type, message: entry.message },
        });
      }
    } finally {
      await backend.close();
    }
  }
});

test("long silent thinking keeps the client stream alive with pings", async () => {
  const backend = await sidecar(
    async (res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      const frames = upstreamResponse("lookup_weather")
        .split("\n\n")
        .filter(Boolean);
      res.write(`${frames[0]}\n\n`);
      await new Promise((resolve) => setTimeout(resolve, 350));
      res.end(
        frames
          .slice(1)
          .map((frame) => `${frame}\n\n`)
          .join(""),
      );
    },
    { keepAliveMs: 100 },
  );
  try {
    const response = await backend.call({ stream: true });
    assert.equal(response.status, 200);
    const wire = await response.text();
    const events = [...wire.matchAll(/^event: (\S+)$/gm)].map(
      (match) => match[1],
    );
    assert.equal(events[0], "message_start");
    assert(
      events.slice(1, events.indexOf("content_block_start")).includes("ping"),
      wire,
    );
    assert.equal(events.at(-1), "message_stop");
  } finally {
    await backend.close();
  }
});

test("Chat without reasoning_effort leaves effort to Anthropic's default", async () => {
  const bodies: Record<string, unknown>[] = [];
  const upstream = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    bodies.push(JSON.parse(Buffer.concat(chunks).toString()));
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.end(upstreamResponse("lookup_weather"));
  });
  const upstreamUrl = await listen(upstream);
  const runtime = loadRuntime(
    {
      models: [
        {
          alias,
          provider: "anthropic",
          model: "claude-opus-5-5",
          baseUrl: upstreamUrl,
        },
      ],
    },
    {
      authContext: {
        env: async (name) =>
          name === "ANTHROPIC_API_KEY" ? apiKey : undefined,
        fileExists: async () => false,
      },
    },
  );
  assert(runtime.ok);
  const backend = createInferenceServer({
    apiKey: internalKey,
    runtime: runtime.value,
    log: () => {},
  });
  const url = await listen(backend.server);
  const chat = (extra: Record<string, unknown>) =>
    fetch(`${url}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${internalKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: alias,
        messages: [{ role: "user", content: "hi" }],
        ...extra,
      }),
    });
  const effortMessages = (body: Record<string, unknown>) =>
    (body.messages as { role: string; output_config?: unknown }[]).filter(
      (message) => message.role === "system",
    );
  try {
    assert.equal((await chat({})).status, 200);
    assert.equal(bodies[0]!.output_config, undefined);
    assert.deepEqual(effortMessages(bodies[0]!), []);
    assert.equal((await chat({ reasoning_effort: "low" })).status, 200);
    assert.deepEqual(effortMessages(bodies[1]!).at(-1)?.output_config, {
      effort: "low",
    });
  } finally {
    await close(backend.server);
    await close(upstream);
  }
});
