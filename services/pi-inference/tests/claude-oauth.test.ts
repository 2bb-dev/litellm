import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer, type Server } from "node:http";
import { test, type TestContext } from "node:test";
import { createModels, createProvider } from "@earendil-works/pi-ai";
import { anthropicMessagesApi } from "@earendil-works/pi-ai/api/anthropic-messages.lazy";
import { messagesResponse, prepareMessages } from "../src/messages.js";
import { loadRuntime } from "../src/runtime.js";
import { createInferenceServer } from "../src/server.js";

interface Payload {
  model: string;
  stream?: boolean;
  system?: { type: string; text?: string; cache_control?: unknown }[];
  tools?: { name: string; input_schema: unknown; [key: string]: unknown }[];
  tool_choice?: { type: string; name?: string };
  messages: { role: string; content: unknown }[];
}

const key = "internal-claude-oauth-fixture-1234567890";
const cache = { type: "ephemeral", ttl: "1h" };
const tool = (name: string) => ({
  name,
  description: "Fixture tool, never executed",
  input_schema: { type: "object", properties: { city: { type: "string" } } },
});
const request = {
  model: "claude",
  max_tokens: 2048,
  system: [
    {
      type: "text",
      text: "Read pi .md files about pi itself and pi packages",
      cache_control: cache,
    },
  ],
  tools: [
    {
      ...tool("lookup_weather"),
      cache_control: cache,
      strict: true,
      eager_input_streaming: true,
    },
  ],
  tool_choice: { type: "tool", name: "lookup_weather" },
  messages: [
    { role: "user", content: "pi itself is user text, do not change it" },
    {
      role: "assistant",
      content: [
        {
          type: "thinking",
          thinking: "signed pi packages",
          signature: "signed-fixture",
        },
        { type: "redacted_thinking", data: "opaque-fixture" },
        {
          type: "tool_use",
          id: "previous",
          name: "lookup_weather",
          input: { city: "Vienna" },
        },
      ],
    },
    {
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "previous",
          content: "sunny",
          cache_control: cache,
        },
      ],
    },
  ],
};

async function listen(server: Server): Promise<string> {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  assert(address && typeof address === "object");
  return `http://127.0.0.1:${address.port}`;
}

async function close(server: Server): Promise<void> {
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );
}

function response(name: string, thinking = false) {
  const index = thinking ? 2 : 0;
  return [
    {
      type: "message_start",
      message: {
        id: "msg_fixture",
        type: "message",
        role: "assistant",
        model: "claude-haiku-4-5",
        content: [],
        usage: {
          input_tokens: 5,
          output_tokens: 0,
          cache_read_input_tokens: 7,
          cache_creation_input_tokens: 3,
        },
      },
    },
    ...(thinking
      ? [
          {
            type: "content_block_start",
            index: 0,
            content_block: { type: "thinking", thinking: "", signature: "" },
          },
          {
            type: "content_block_delta",
            index: 0,
            delta: { type: "thinking_delta", thinking: "signed pi itself 🌍" },
          },
          {
            type: "content_block_delta",
            index: 0,
            delta: {
              type: "signature_delta",
              signature: "signature-must-survive",
            },
          },
          { type: "content_block_stop", index: 0 },
          {
            type: "content_block_start",
            index: 1,
            content_block: {
              type: "redacted_thinking",
              data: "opaque-must-survive",
            },
          },
          { type: "content_block_stop", index: 1 },
        ]
      : []),
    {
      type: "content_block_start",
      index,
      content_block: { type: "tool_use", id: "new_call", name, input: {} },
    },
    {
      type: "content_block_delta",
      index,
      delta: { type: "input_json_delta", partial_json: '{"city":"Vienna"}' },
    },
    { type: "content_block_stop", index },
    {
      type: "message_delta",
      delta: { stop_reason: "tool_use", stop_sequence: null },
      usage: { output_tokens: 4 },
    },
    { type: "message_stop" },
  ]
    .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
    .join("");
}

async function fixture(
  t: TestContext,
  options: {
    apiKey?: string;
    provider?: string;
    respond?: (payload: Payload) => string | Promise<string>;
  } = {},
) {
  const apiKey = options.apiKey ?? "sk-ant-oat-fixture";
  const provider = options.provider ?? "anthropic";
  const envKey = `${provider.toUpperCase().replace(/[^A-Z0-9]/g, "_")}_API_KEY`;
  const captured: Payload[] = [];
  const logs: Record<string, unknown>[] = [];
  const authReads: string[] = [];
  const upstream = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    const payload = JSON.parse(Buffer.concat(chunks).toString()) as Payload;
    captured.push(payload);
    if (
      provider === "anthropic" &&
      !apiKey.includes("sk-ant-oat") &&
      !payload.stream
    ) {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          id: "msg_fixture",
          type: "message",
          role: "assistant",
          model: "claude-haiku-4-5",
          content: [
            {
              type: "tool_use",
              id: "new_call",
              name: payload.tools?.[0]?.name ?? "lookup_weather",
              input: { city: "Vienna" },
            },
          ],
          stop_reason: "tool_use",
          stop_sequence: null,
          usage: { input_tokens: 5, output_tokens: 4 },
        }),
      );
      return;
    }
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.end(
      await (options.respond?.(payload) ??
        response(payload.tools?.[0]?.name ?? "lookup_weather")),
    );
  });
  const baseUrl = await listen(upstream);
  t.after(() => close(upstream));
  const runtime = loadRuntime(
    {
      models: [
        {
          alias: "claude",
          provider,
          model: "claude-haiku-4-5",
          baseUrl,
          ...(provider !== "anthropic"
            ? {
                metadata: {
                  api: "anthropic-messages",
                  contextWindow: 200000,
                  maxTokens: 8192,
                  reasoning: true,
                  input: ["text"],
                  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                },
              }
            : {}),
        },
      ],
    },
    {
      authContext: {
        env: async (name) => {
          authReads.push(name);
          return name === envKey ? apiKey : undefined;
        },
        fileExists: async () => false,
      },
    },
  );
  assert(runtime.ok);
  const backend = createInferenceServer({
    apiKey: key,
    runtime: runtime.value,
    log: (record) => logs.push(record),
  });
  const url = await listen(backend.server);
  t.after(() => close(backend.server));
  return {
    captured,
    logs,
    authReads,
    url,
    model: runtime.value.routes.get("claude")!,
    post: (body: unknown, endpoint = "messages") =>
      fetch(`${url}/v1/${endpoint}`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${key}`,
          "content-type": "application/json",
        },
        body: JSON.stringify(body),
      }),
  };
}

test("Anthropic OAuth adapts native payload after overlays and returns the client's tool name", async (t) => {
  const f = await fixture(t);
  const before = structuredClone(request);
  const result = await f.post(request);
  assert.equal(result.status, 200);
  const body = await result.json();
  assert.equal(f.captured[0]?.tools?.[0]?.name, "mcp__pi__lookup_weather");
  assert.equal(f.captured[0]?.tool_choice?.name, "mcp__pi__lookup_weather");
  assert.deepEqual(f.captured[0]?.system?.at(-1), {
    ...request.system[0],
    text: "Read cli .md files about the cli itself and cli packages",
  });
  assert.deepEqual(f.captured[0]?.tools?.[0], {
    ...request.tools[0],
    name: "mcp__pi__lookup_weather",
  });
  const expected = structuredClone(request.messages);
  (expected[1]!.content as { name?: string }[])[2]!.name =
    "mcp__pi__lookup_weather";
  assert.deepEqual(f.captured[0]?.messages, expected);
  assert.equal(body.content[0].name, "lookup_weather");
  assert.deepEqual(body.content[0].input, { city: "Vienna" });
  assert.deepEqual(request, before);
  assert.equal(
    f.authReads.filter((name) => name === "ANTHROPIC_API_KEY").length,
    1,
  );
  assert.equal(f.logs.length, 1);
  assert(!JSON.stringify(f.logs).includes("sk-ant-oat-fixture"));
  assert(!JSON.stringify(f.logs).includes("signed-fixture"));
});

for (const options of [
  { apiKey: "sk-ant-api-fixture" },
  { provider: "another-anthropic" },
]) {
  test(`leaves non-OAuth or non-Anthropic requests unchanged: ${JSON.stringify(options)}`, async (t) => {
    const f = await fixture(t, options);
    const result = await f.post(request);
    assert.equal(result.status, 200);
    const body = await result.json();
    assert.deepEqual(f.captured[0]?.tools, request.tools);
    assert.deepEqual(f.captured[0]?.messages, request.messages);
    assert.deepEqual(f.captured[0]?.tool_choice, request.tool_choice);
    assert.deepEqual(f.captured[0]?.system?.slice(-1), request.system);
    assert.equal(body.content[0].name, "lookup_weather");
  });
}

function chatRequest(name = "lookup_weather") {
  return {
    model: "claude",
    messages: [
      {
        role: "system",
        content: "Read pi .md files about pi itself and pi packages",
      },
      { role: "user", content: "pi itself is user text, do not change it" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "previous",
            type: "function",
            function: { name, arguments: '{"city":"Vienna"}' },
          },
        ],
      },
      { role: "tool", tool_call_id: "previous", name, content: "sunny" },
    ],
    tools: [
      {
        type: "function",
        function: { name, parameters: tool(name).input_schema },
      },
    ],
  };
}

for (const protocol of ["messages", "chat/completions"]) {
  for (const stream of [false, true]) {
    test(`canonical tool names and usage survive ${protocol}, streaming=${stream}`, async (t) => {
      const f = await fixture(t);
      const result = await f.post(
        {
          ...(protocol === "messages" ? request : chatRequest()),
          stream,
          ...(protocol !== "messages" && stream
            ? { stream_options: { include_usage: true } }
            : {}),
        },
        protocol,
      );
      assert.equal(result.status, 200);
      const wire = await result.text();
      assert.match(wire, /"name":"lookup_weather"/);
      assert(!wire.includes("mcp__pi__lookup_weather"));
      assert.equal(f.captured[0]?.tools?.[0]?.name, "mcp__pi__lookup_weather");
      assert.equal(
        f.captured[0]?.system?.at(-1)?.text,
        "Read cli .md files about the cli itself and cli packages",
      );
      assert(
        JSON.stringify(f.captured[0]?.messages).includes(
          '"name":"mcp__pi__lookup_weather"',
        ),
      );
      assert(
        JSON.stringify(f.captured[0]?.messages).includes(
          "pi itself is user text",
        ),
      );
      if (stream)
        assert(
          wire.includes(protocol === "messages" ? "message_stop" : "[DONE]"),
        );
      assert.equal(f.logs[0]?.status, 200);
      assert.deepEqual(f.logs[0]?.usage, {
        input: 5,
        output: 4,
        cache_read: 7,
        cache_write: 3,
        reasoning: undefined,
      });
      assert.equal(
        f.authReads.filter((name) => name === "ANTHROPIC_API_KEY").length,
        1,
      );
    });
  }
}

test("aliases collide safely with each other and direct MCP tools, without changing core tools or schemas", async (t) => {
  const f = await fixture(t);
  const names = [
    "lookup-weather",
    "lookup_weather",
    "mcp__pi__lookup_weather",
    "BaSh",
    "MCP__foreign__tool",
  ];
  const result = await f.post({
    ...request,
    tool_choice: { type: "auto" },
    tools: names.map((name) => ({ ...tool(name), description: name })),
  });
  assert.equal(result.status, 200);
  const payload = f.captured[0];
  assert(payload?.tools);
  assert.deepEqual(
    payload.tools.map((tool) => tool.name),
    [
      "mcp__pi__lookup_weather_2",
      "mcp__pi__lookup_weather_3",
      "mcp__pi__lookup_weather",
      "Bash",
      "MCP__foreign__tool",
    ],
  );
  assert.deepEqual(
    payload.tools.map((tool) => tool.description),
    names,
  );
  assert.equal(
    new Set(payload.tools.map((tool) => tool.name.toLowerCase())).size,
    names.length,
  );
  const body = await result.json();
  assert.equal(body.content[0].name, "lookup-weather");
});

test("direct MCP and canonical core response names are not rewritten as managed aliases", async (t) => {
  const f = await fixture(t);
  for (const name of [
    "mcp__pi__lookup_weather",
    "MCP__foreign__tool",
    "BaSh",
  ]) {
    const result = await f.post({
      ...request,
      tools: [tool(name)],
      tool_choice: { type: "tool", name },
      messages: [{ role: "user", content: "hello" }],
    });
    assert.equal(result.status, 200);
    const body = await result.json();
    assert.equal(body.content[0].name, name);
  }
});

test("parallel requests keep separate reverse maps for sanitized alias collisions", async (t) => {
  let release!: () => void;
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  let arrived = 0;
  const f = await fixture(t, {
    respond: async (payload) => {
      if (++arrived === 2) release();
      await pending;
      return response(payload.tools![0]!.name);
    },
  });
  const names = ["lookup-weather", "lookup_weather"];
  const results = await Promise.all(
    names.map(async (name) => {
      const result = await f.post({
        model: "claude",
        max_tokens: 128,
        tools: [tool(name)],
        messages: [{ role: "user", content: "hello" }],
      });
      assert.equal(result.status, 200);
      return result.json();
    }),
  );
  assert.deepEqual(
    f.captured.map((payload) => payload.tools?.[0]?.name),
    ["mcp__pi__lookup_weather", "mcp__pi__lookup_weather"],
  );
  assert.deepEqual(
    results.map((body) => body.content[0].name),
    names,
  );
});

test("bounded aliases stay unique when long Chat tool names are truncated", async (t) => {
  const f = await fixture(t);
  const names = ["x".repeat(150) + "a", "x".repeat(150) + "b"];
  const input = chatRequest(names[0]);
  const result = await f.post(
    {
      ...input,
      tools: names.map((name) => ({
        type: "function",
        function: { name, parameters: tool(name).input_schema },
      })),
    },
    "chat/completions",
  );
  assert.equal(result.status, 200);
  const aliases = f.captured[0]?.tools?.map((tool) => tool.name);
  assert(aliases);
  assert.equal(new Set(aliases).size, 2);
  for (const name of aliases) assert.match(name, /^mcp__pi__[a-z0-9_]{1,119}$/);
  const body = await result.json();
  assert.equal(body.choices[0].message.tool_calls[0].function.name, names[0]);
});

test("actual Pi client can replay signed streamed thinking and canonical tools through the OAuth backend", async (t) => {
  const f = await fixture(t, {
    respond: (payload) => response(payload.tools![0]!.name, true),
  });
  const model = { ...f.model, id: "claude", baseUrl: f.url };
  const client = createModels();
  client.setProvider(
    createProvider({
      id: "anthropic",
      models: [model],
      api: anthropicMessagesApi(),
      auth: {
        apiKey: {
          name: "Fixture",
          resolve: async () => ({ auth: { apiKey: key } }),
        },
      },
    }),
  );
  const prepared = prepareMessages(request, model);
  assert(prepared.ok);
  const stream = client.streamSimple(model, prepared.value.context, {
    ...prepared.value.options,
    apiKey: key,
    timeoutMs: 2000,
    maxRetries: 0,
  });
  const result = await stream.result();
  assert.equal(result.stopReason, "toolUse", result.errorMessage);
  const final = messagesResponse(result, "claude", "msg_client") as {
    content: unknown[];
  };
  const expected = [
    {
      type: "thinking",
      thinking: "signed pi itself 🌍",
      signature: "signature-must-survive",
    },
    { type: "redacted_thinking", data: "opaque-must-survive" },
    {
      type: "tool_use",
      id: "new_call",
      name: "lookup_weather",
      input: { city: "Vienna" },
    },
  ];
  assert.deepEqual(final.content, expected);
  for await (const event of stream) {
    const message =
      event.type === "done"
        ? event.message
        : event.type === "error"
          ? event.error
          : event.partial;
    for (const block of message.content)
      if (block.type === "toolCall") assert.equal(block.name, "lookup_weather");
  }
  const replay = await f.post({
    ...request,
    messages: [
      ...request.messages,
      { role: "assistant", content: final.content },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "new_call", content: "sunny" },
        ],
      },
    ],
  });
  assert.equal(replay.status, 200);
  const body = await replay.json();
  assert.deepEqual(body.content, expected);
  assert.deepEqual(f.captured[1]?.messages.at(-2)?.content, [
    expected[0],
    expected[1],
    { ...expected[2], name: "mcp__pi__lookup_weather" },
  ]);
  for (const secret of [
    "sk-ant-oat-fixture",
    "signature-must-survive",
    "opaque-must-survive",
  ])
    assert(!JSON.stringify(f.logs).includes(secret));
});
