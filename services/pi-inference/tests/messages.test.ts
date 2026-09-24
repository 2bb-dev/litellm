import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer } from "node:http";
import test from "node:test";
import { createModels, createProvider } from "@earendil-works/pi-ai";
import type {
  AssistantMessage,
  AssistantMessageEvent,
  Model,
  ThinkingContent,
  ToolCall,
} from "@earendil-works/pi-ai";
import { anthropicMessagesApi } from "@earendil-works/pi-ai/api/anthropic-messages.lazy";
import {
  createMessagesEncoder,
  messagesResponse,
  prepareMessages,
} from "../src/messages.js";
import { emptyUsage, type WireEvent } from "../src/protocol.js";

const model: Model<"anthropic-messages"> = {
  id: "claude-test",
  name: "Test",
  api: "anthropic-messages",
  provider: "fixture",
  baseUrl: "http://127.0.0.1:1",
  reasoning: true,
  input: ["text", "image"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 200000,
  maxTokens: 8192,
};
const request = {
  model: "public-alias",
  max_tokens: 2048,
  messages: [{ role: "user", content: "Hello" }],
};

test("prepares native text without accepting lossy cross-protocol routing", async () => {
  const result = prepareMessages(request, model);
  assert.ok(result.ok);
  assert.equal(result.value.stream, false);
  assert.equal(result.value.options.maxTokens, 2048);
  assert.equal(result.value.options.cacheRetention, "none");
  assert.equal(result.value.context.messages[0]?.content, "Hello");
  const payload = await result.value.options.onPayload?.(
    { model: model.id, messages: [], max_tokens: 1, stream: true },
    model,
  );
  assert.deepEqual(payload, {
    model: model.id,
    messages: request.messages,
    max_tokens: 2048,
    stream: true,
  });
  const rejected = prepareMessages(request, {
    ...model,
    api: "openai-completions",
  });
  assert.equal(rejected.ok, false);
});

test("rejects tool additions that cannot be mapped to a stable declaration", () => {
  const addition = {
    role: "system",
    content: [
      {
        type: "tool_addition",
        tool: { type: "tool_reference", name: "lookup_weather" },
      },
    ],
  };
  const removal = {
    role: "system",
    content: [
      {
        type: "tool_removal",
        tool: { type: "tool_reference", name: "lookup_weather" },
      },
    ],
  };
  assert.equal(
    prepareMessages(
      { ...request, messages: [request.messages[0], addition] },
      model,
    ).ok,
    false,
  );
  assert.equal(
    prepareMessages(
      {
        ...request,
        tools: [
          {
            name: "lookup_weather",
            input_schema: { type: "object", properties: {} },
          },
        ],
        messages: [request.messages[0], removal, addition],
      },
      model,
    ).ok,
    false,
  );
});

const cache = { type: "ephemeral", ttl: "1h" };
const image = {
  type: "image",
  source: { type: "base64", media_type: "image/png", data: "aGVsbG8=" },
  cache_control: cache,
};
const nativeRequest = {
  ...request,
  stream: true,
  system: [
    { type: "text", text: "first", cache_control: cache },
    { type: "text", text: "second" },
  ],
  thinking: { type: "enabled", budget_tokens: 1024 },
  output_config: { effort: "high" },
  tool_choice: { type: "auto", disable_parallel_tool_use: true },
  metadata: { user_id: "test-user" },
  tools: [
    {
      name: "bash",
      description: "Run",
      input_schema: {
        type: "object",
        properties: { command: { $ref: "#/$defs/command" } },
        required: ["command"],
        additionalProperties: false,
        $defs: { command: { type: "string", minLength: 1 } },
      },
      cache_control: cache,
    },
  ],
  messages: [
    { role: "user", content: [image, { type: "text", text: "go" }] },
    {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "reason", signature: "signed" },
        { type: "redacted_thinking", data: "opaque" },
        { type: "text", text: "running", cache_control: cache },
        {
          type: "tool_use",
          id: "tool_1",
          name: "bash",
          input: { command: "pwd" },
        },
      ],
    },
    {
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "tool_1",
          content: [
            { type: "text", text: "result", cache_control: cache },
            image,
          ],
          is_error: false,
          cache_control: cache,
        },
        { type: "text", text: "continue" },
      ],
    },
  ],
};

const sse = (data: { type: string; [key: string]: unknown }) =>
  `event: ${data.type}\ndata: ${JSON.stringify(data)}\n\n`;
const fixtureResponse = [
  {
    type: "message_start",
    message: {
      id: "upstream",
      type: "message",
      role: "assistant",
      content: [],
      model: model.id,
      usage: {
        input_tokens: 10,
        output_tokens: 0,
        cache_read_input_tokens: 30,
        cache_creation_input_tokens: 20,
        cache_creation: {
          ephemeral_1h_input_tokens: 5,
          ephemeral_5m_input_tokens: 15,
        },
      },
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
    delta: { type: "text_delta", text: "Hello" },
  },
  { type: "content_block_stop", index: 0 },
  {
    type: "message_delta",
    delta: { stop_reason: "end_turn", stop_sequence: null },
    usage: { output_tokens: 4 },
  },
  { type: "message_stop" },
]
  .map(sse)
  .join("");

test("real Pi native HTTP preserves cache, signed thinking, schema and OAuth transforms", async () => {
  for (const apiKey of ["local-test-key", "sk-ant-oat-local-test"]) {
    const received: Record<string, unknown>[] = [];
    const server = createServer(async (req, res) => {
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(Buffer.from(chunk));
      received.push(
        JSON.parse(Buffer.concat(chunks).toString()) as Record<string, unknown>,
      );
      // The Anthropic SDK's beta client adds `?beta=true`.
      assert.equal(
        new URL(req.url ?? "", "http://fixture.invalid").pathname,
        "/v1/messages",
      );
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end(fixtureResponse);
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address();
    assert.ok(address && typeof address !== "string");
    try {
      const localModel = {
        ...model,
        baseUrl: `http://127.0.0.1:${address.port}`,
      };
      const models = createModels();
      models.setProvider(
        createProvider({
          id: model.provider,
          models: [localModel],
          api: anthropicMessagesApi(),
          auth: {
            apiKey: {
              name: "Fixture",
              resolve: async () => ({ auth: { apiKey } }),
            },
          },
        }),
      );
      const prepared = prepareMessages(nativeRequest, localModel);
      assert.ok(prepared.ok);
      const result = await models.completeSimple(
        localModel,
        prepared.value.context,
        { ...prepared.value.options, timeoutMs: 2000, maxRetries: 0 },
      );
      assert.equal(result.stopReason, "stop", result.errorMessage);
      const payload = received[0];
      assert.ok(payload);
      const oauth = apiKey.includes("oat");
      const expectedMessages = structuredClone(nativeRequest.messages);
      if (oauth)
        (expectedMessages[1]!.content[3] as { name: string }).name = "Bash";
      assert.deepEqual(payload.messages, expectedMessages);
      assert.deepEqual(
        payload.system,
        oauth
          ? [
              {
                type: "text",
                text: "You are Claude Code, Anthropic's official CLI for Claude.",
              },
              ...nativeRequest.system,
            ]
          : nativeRequest.system,
      );
      assert.deepEqual(payload.thinking, nativeRequest.thinking);
      assert.deepEqual(payload.output_config, nativeRequest.output_config);
      assert.deepEqual(payload.tool_choice, nativeRequest.tool_choice);
      assert.deepEqual(payload.metadata, nativeRequest.metadata);
      assert.equal(payload.stop_sequences, undefined);
      assert.equal(payload.max_tokens, 2048);
      const tools = payload.tools as Record<string, unknown>[];
      assert.equal(tools[0]?.name, oauth ? "Bash" : "bash");
      assert.deepEqual(
        tools[0]?.input_schema,
        nativeRequest.tools[0]?.input_schema,
      );
      assert.deepEqual(tools[0]?.cache_control, cache);
    } finally {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  }
});

test("strictly rejects unsupported nested protocol features and malformed limits", () => {
  const invalidBodies: unknown[] = [
    null,
    [],
    { ...request, model: "" },
    { ...request, max_tokens: 0 },
    { ...request, max_tokens: 1.5 },
    { ...request, max_tokens: 9000 },
    { ...request, messages: [] },
    { ...request, stream: "true" },
    { ...request, temperature: -1 },
    { ...request, temperature: 2 },
    { ...request, top_p: 2 },
    { ...request, top_k: -1 },
    { ...request, service_tier: "auto" },
    { ...request, container: "opaque" },
    { ...request, mcp_servers: [] },
    { ...request, context_management: {} },
    { ...request, metadata: { user_id: "x", secret: "must-not-leak" } },
    { ...request, system: [{ type: "text", text: "x", citations: [] }] },
    { ...request, messages: [{ role: "system", content: "x" }] },
    {
      ...request,
      messages: [
        {
          role: "user",
          content: [
            { type: "document", source: { type: "text", data: "document" } },
          ],
        },
      ],
    },
    {
      ...request,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "x",
              cache_control: { type: "ephemeral", ttl: "2h" },
            },
          ],
        },
      ],
    },
    {
      ...request,
      messages: [
        {
          role: "assistant",
          content: [{ type: "text", text: "x", citations: [] }],
        },
      ],
    },
    {
      ...request,
      messages: [
        {
          role: "assistant",
          content: [
            { type: "server_tool_use", id: "s", name: "search", input: {} },
          ],
        },
      ],
    },
    {
      ...request,
      messages: [
        {
          role: "assistant",
          content: [{ type: "thinking", thinking: "x", signature: "" }],
        },
      ],
    },
    {
      ...request,
      messages: [
        {
          role: "assistant",
          content: [{ type: "redacted_thinking", data: 1 }],
        },
      ],
    },
    {
      ...request,
      messages: [
        {
          role: "user",
          content: [
            {
              ...image,
              source: { type: "url", url: "https://invalid.test/a.png" },
            },
          ],
        },
      ],
    },
    {
      ...request,
      messages: [
        {
          role: "user",
          content: [
            { ...image, source: { ...image.source, data: "not-base64" } },
          ],
        },
      ],
    },
    {
      ...request,
      tools: [{ type: "web_search_20250305", name: "web_search" }],
    },
    {
      ...request,
      tools: [
        {
          ...nativeRequest.tools[0],
          input_schema: { type: "array" },
        },
      ],
    },
    {
      ...request,
      tools: [
        {
          ...nativeRequest.tools[0],
          input_schema: [],
        },
      ],
    },
    { ...request, thinking: { type: "enabled", budget_tokens: 2048 } },
    { ...request, thinking: { type: "enabled", budget_tokens: 100 } },
    { ...request, thinking: { type: "adaptive", budget_tokens: 1024 } },
    { ...request, thinking: { type: "disabled", extra: true } },
    {
      ...request,
      output_config: { effort: "high", format: { type: "json_schema" } },
    },
    { ...request, tool_choice: { type: "none", name: "bash" } },
    { ...request, tool_choice: { type: "any" } },
    {
      ...request,
      tools: nativeRequest.tools,
      tool_choice: { type: "tool", name: "unknown" },
    },
    { ...nativeRequest, temperature: 0.5 },
    { ...nativeRequest, tool_choice: { type: "any" } },
    {
      ...nativeRequest,
      tools: [...nativeRequest.tools, ...nativeRequest.tools],
    },
  ];
  for (const body of invalidBodies) {
    const result = prepareMessages(body, model);
    assert.equal(result.ok, false, JSON.stringify(body));
    if (!result.ok) {
      assert.equal(result.error.status, 400);
      assert.doesNotMatch(result.error.message, /must-not-leak/);
    }
  }
  assert.equal(
    prepareMessages(nativeRequest, { ...model, input: ["text"] }).ok,
    false,
  );
  assert.equal(
    prepareMessages(nativeRequest, { ...model, reasoning: false }).ok,
    false,
  );
  assert.equal(
    prepareMessages(
      { ...request, temperature: 1 },
      { ...model, compat: { supportsTemperature: false } },
    ).ok,
    false,
  );
  assert.equal(
    prepareMessages(
      { ...request, thinking: { type: "disabled" } },
      { ...model, thinkingLevelMap: { off: null } },
    ).ok,
    false,
  );
  assert.equal(
    prepareMessages(nativeRequest, {
      ...model,
      compat: { forceAdaptiveThinking: true },
    }).ok,
    false,
  );
});

test("bounded JSON input cannot recurse forever", () => {
  const cyclic: Record<string, unknown> = { ...request };
  cyclic.self = cyclic;
  const bodyWithInput = (input: unknown) => ({
    ...request,
    messages: [
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "tool_1", name: "bash", input }],
      },
    ],
  });
  assert.equal(prepareMessages(bodyWithInput(cyclic), model).ok, false);
  const deep = Array.from({ length: 10000 }).reduce<unknown>(
    (value) => ({ x: value }),
    {},
  );
  assert.equal(prepareMessages(bodyWithInput(deep), model).ok, false);
});

test("forwards disabled/adaptive thinking, sampling and native forced tool choice", async () => {
  for (const thinking of [{ type: "disabled" }, { type: "adaptive" }]) {
    const body = {
      ...request,
      thinking,
      ...(thinking.type === "disabled"
        ? {
            temperature: 0.2,
            top_p: 0.9,
            top_k: 4,
            tools: nativeRequest.tools,
            tool_choice: {
              type: "tool",
              name: "bash",
              disable_parallel_tool_use: true,
            },
          }
        : { output_config: { effort: "max" } }),
    };
    const prepared = prepareMessages(body, model);
    assert.ok(prepared.ok);
    const output = (await prepared.value.options.onPayload?.(
      {
        model: model.id,
        messages: [],
        tools: [{ name: "Bash" }],
        max_tokens: 1,
        stream: true,
      },
      model,
    )) as Record<string, unknown>;
    assert.deepEqual(output.thinking, thinking);
    if (thinking.type === "disabled") {
      assert.equal(output.temperature, 0.2);
      assert.equal(output.top_p, 0.9);
      assert.equal(output.top_k, 4);
      assert.deepEqual(output.tool_choice, {
        type: "tool",
        name: "Bash",
        disable_parallel_tool_use: true,
      });
    } else assert.deepEqual(output.output_config, { effort: "max" });
  }
});

const assistant = (
  content: AssistantMessage["content"] = [],
): AssistantMessage => ({
  role: "assistant",
  api: model.api,
  provider: model.provider,
  model: model.id,
  content,
  timestamp: 0,
  stopReason: "toolUse",
  rawStopReason: "tool_use",
  usage: {
    ...emptyUsage(),
    input: 10,
    output: 4,
    cacheRead: 30,
    cacheWrite: 20,
    cacheWrite1h: 5,
    reasoning: 2,
    totalTokens: 64,
  },
});
const usage = {
  input_tokens: 10,
  output_tokens: 4,
  cache_read_input_tokens: 30,
  cache_creation_input_tokens: 20,
  cache_creation: {
    ephemeral_1h_input_tokens: 5,
    ephemeral_5m_input_tokens: 15,
  },
  output_tokens_details: { thinking_tokens: 2 },
};
const wire = (
  event: string,
  fields: Record<string, unknown> = {},
): WireEvent => ({ event, data: { type: event, ...fields } });

test("native response preserves signed and redacted thinking, tool input and cache split", () => {
  const message = assistant([
    { type: "thinking", thinking: "reason", thinkingSignature: "sig" },
    {
      type: "thinking",
      thinking: "[Reasoning redacted]",
      thinkingSignature: "opaque",
      redacted: true,
    },
    { type: "text", text: "answer" },
    {
      type: "toolCall",
      id: "tool_1",
      name: "bash",
      arguments: { command: "pwd" },
    },
  ]);
  assert.deepEqual(messagesResponse(message, "public-alias", "msg_local"), {
    id: "msg_local",
    type: "message",
    role: "assistant",
    model: "public-alias",
    content: [
      { type: "thinking", thinking: "reason", signature: "sig" },
      { type: "redacted_thinking", data: "opaque" },
      { type: "text", text: "answer" },
      {
        type: "tool_use",
        id: "tool_1",
        name: "bash",
        input: { command: "pwd" },
      },
    ],
    stop_reason: "tool_use",
    stop_sequence: null,
    usage,
  });
  for (const reason of [
    "end_turn",
    "max_tokens",
    "pause_turn",
    "stop_sequence",
    "refusal",
  ]) {
    const response = messagesResponse(
      { ...message, stopReason: "stop", rawStopReason: reason },
      "a",
      "i",
    ) as Record<string, unknown>;
    assert.equal(response.stop_reason, reason);
    assert.equal(response.stop_sequence, null);
  }
  assert.doesNotMatch(
    JSON.stringify(
      messagesResponse(
        { ...message, stopReason: "error", errorMessage: "secret-upstream" },
        "a",
        "i",
      ),
    ),
    /secret-upstream/,
  );
});

test("SSE uses immutable deltas, never the mutable partial as a start snapshot", () => {
  const thinking: ThinkingContent = {
    type: "thinking",
    thinking: "reason",
    thinkingSignature: "sig",
  };
  const redacted: ThinkingContent = {
    type: "thinking",
    thinking: "[Reasoning redacted]",
    thinkingSignature: "opaque",
    redacted: true,
  };
  const tool: ToolCall = {
    type: "toolCall",
    id: "tool_1",
    name: "bash",
    arguments: { command: "pwd" },
  };
  const message = assistant([
    thinking,
    redacted,
    { type: "text", text: "answer" },
    tool,
  ]);
  const events: AssistantMessageEvent[] = [
    { type: "start", partial: message },
    { type: "thinking_start", contentIndex: 0, partial: message },
    { type: "thinking_delta", contentIndex: 0, delta: "rea", partial: message },
    { type: "thinking_delta", contentIndex: 0, delta: "son", partial: message },
    {
      type: "thinking_end",
      contentIndex: 0,
      content: "reason",
      partial: message,
    },
    { type: "thinking_start", contentIndex: 1, partial: message },
    {
      type: "thinking_end",
      contentIndex: 1,
      content: "[Reasoning redacted]",
      partial: message,
    },
    { type: "text_start", contentIndex: 2, partial: message },
    { type: "text_delta", contentIndex: 2, delta: "answer", partial: message },
    { type: "text_end", contentIndex: 2, content: "answer", partial: message },
    { type: "toolcall_start", contentIndex: 3, partial: message },
    {
      type: "toolcall_delta",
      contentIndex: 3,
      delta: '{"command":',
      partial: message,
    },
    {
      type: "toolcall_delta",
      contentIndex: 3,
      delta: '"pwd"}',
      partial: message,
    },
    { type: "toolcall_end", contentIndex: 3, toolCall: tool, partial: message },
    { type: "done", reason: "toolUse", message },
  ];
  const encode = createMessagesEncoder("public-alias", "msg_local");
  const frames = events.flatMap(encode);
  assert.deepEqual(frames, [
    wire("message_start", {
      message: {
        id: "msg_local",
        type: "message",
        role: "assistant",
        model: "public-alias",
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 0,
          output_tokens: 0,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      },
    }),
    wire("content_block_start", {
      index: 0,
      content_block: { type: "thinking", thinking: "", signature: "" },
    }),
    wire("content_block_delta", {
      index: 0,
      delta: { type: "thinking_delta", thinking: "rea" },
    }),
    wire("content_block_delta", {
      index: 0,
      delta: { type: "thinking_delta", thinking: "son" },
    }),
    wire("content_block_delta", {
      index: 0,
      delta: { type: "signature_delta", signature: "sig" },
    }),
    wire("content_block_stop", { index: 0 }),
    wire("content_block_start", {
      index: 1,
      content_block: { type: "redacted_thinking", data: "opaque" },
    }),
    wire("content_block_stop", { index: 1 }),
    wire("content_block_start", {
      index: 2,
      content_block: { type: "text", text: "" },
    }),
    wire("content_block_delta", {
      index: 2,
      delta: { type: "text_delta", text: "answer" },
    }),
    wire("content_block_stop", { index: 2 }),
    wire("content_block_start", {
      index: 3,
      content_block: {
        type: "tool_use",
        id: "tool_1",
        name: "bash",
        input: {},
      },
    }),
    wire("content_block_delta", {
      index: 3,
      delta: { type: "input_json_delta", partial_json: '{"command":' },
    }),
    wire("content_block_delta", {
      index: 3,
      delta: { type: "input_json_delta", partial_json: '"pwd"}' },
    }),
    wire("content_block_stop", { index: 3 }),
    wire("message_delta", {
      delta: { stop_reason: "tool_use", stop_sequence: null },
      usage,
    }),
    wire("message_stop"),
  ]);
  assert.deepEqual(encode({ type: "done", reason: "toolUse", message }), []);
  assert.deepEqual(
    createMessagesEncoder(
      "a",
      "i",
    )({
      type: "error",
      reason: "error",
      error: {
        ...message,
        stopReason: "error",
        errorMessage: "secret-upstream",
      },
    }),
    [],
  );
});

test("native HTTP response survives queued mutable Pi events and signed replay", async () => {
  const response = [
    {
      type: "message_start",
      message: {
        id: "upstream",
        type: "message",
        role: "assistant",
        content: [],
        model: model.id,
        usage: {
          input_tokens: 10,
          output_tokens: 0,
          cache_read_input_tokens: 30,
          cache_creation_input_tokens: 20,
          cache_creation: {
            ephemeral_1h_input_tokens: 5,
            ephemeral_5m_input_tokens: 15,
          },
        },
      },
    },
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "thinking", thinking: "", signature: "" },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "thinking_delta", thinking: "reason" },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "signature_delta", signature: "sig" },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "content_block_start",
      index: 1,
      content_block: { type: "redacted_thinking", data: "opaque" },
    },
    { type: "content_block_stop", index: 1 },
    {
      type: "content_block_start",
      index: 2,
      content_block: { type: "text", text: "" },
    },
    {
      type: "content_block_delta",
      index: 2,
      delta: { type: "text_delta", text: "answer 🌍" },
    },
    { type: "content_block_stop", index: 2 },
    {
      type: "content_block_start",
      index: 3,
      content_block: {
        type: "tool_use",
        id: "tool_2",
        name: "Bash",
        input: {},
      },
    },
    {
      type: "content_block_delta",
      index: 3,
      delta: { type: "input_json_delta", partial_json: '{"command":' },
    },
    {
      type: "content_block_delta",
      index: 3,
      delta: { type: "input_json_delta", partial_json: '"pwd"}' },
    },
    { type: "content_block_stop", index: 3 },
    {
      type: "message_delta",
      delta: { stop_reason: "tool_use", stop_sequence: null },
      usage: {
        output_tokens: 4,
        output_tokens_details: { thinking_tokens: 2 },
      },
    },
    { type: "message_stop" },
  ]
    .map(sse)
    .join("");
  const server = createServer(async (req, res) => {
    req.resume();
    res.writeHead(200, { "content-type": "text/event-stream" });
    const bytes = Buffer.from(response);
    for (let offset = 0; offset < bytes.length; offset += 7)
      res.write(bytes.subarray(offset, offset + 7));
    res.end();
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  assert.ok(address && typeof address !== "string");
  try {
    const localModel = {
      ...model,
      baseUrl: `http://127.0.0.1:${address.port}`,
    };
    const models = createModels();
    models.setProvider(
      createProvider({
        id: model.provider,
        models: [localModel],
        api: anthropicMessagesApi(),
        auth: {
          apiKey: {
            name: "Fixture",
            resolve: async () => ({ auth: { apiKey: "sk-ant-oat-fixture" } }),
          },
        },
      }),
    );
    const prepared = prepareMessages(nativeRequest, localModel);
    assert.ok(prepared.ok);
    const stream = models.streamSimple(localModel, prepared.value.context, {
      ...prepared.value.options,
      timeoutMs: 2000,
      maxRetries: 0,
    });
    const result = await stream.result();
    assert.equal(result.stopReason, "toolUse", result.errorMessage);
    const encode = createMessagesEncoder("public-alias", "msg_local");
    const frames: WireEvent[] = [];
    for await (const event of stream) frames.push(...encode(event));
    const final = messagesResponse(result, "public-alias", "msg_local") as {
      content: unknown[];
      usage: unknown;
    };
    assert.deepEqual(final.usage, usage);
    assert.deepEqual(final.content, [
      { type: "thinking", thinking: "reason", signature: "sig" },
      { type: "redacted_thinking", data: "opaque" },
      { type: "text", text: "answer 🌍" },
      {
        type: "tool_use",
        id: "tool_2",
        name: "bash",
        input: { command: "pwd" },
      },
    ]);
    assert.ok(
      frames.some((frame) =>
        JSON.stringify(frame.data).includes('"signature":"sig"'),
      ),
    );
    assert.ok(
      frames.some((frame) =>
        JSON.stringify(frame.data).includes(
          '"type":"redacted_thinking","data":"opaque"',
        ),
      ),
    );
    assert.equal(
      frames.filter((frame) => JSON.stringify(frame.data).includes("answer 🌍"))
        .length,
      1,
    );
    assert.deepEqual(
      frames.at(-2),
      wire("message_delta", {
        delta: { stop_reason: "tool_use", stop_sequence: null },
        usage,
      }),
    );
    assert.deepEqual(frames.at(-1), wire("message_stop"));
    const replay = prepareMessages(
      {
        ...request,
        tools: nativeRequest.tools,
        messages: [
          ...request.messages,
          { role: "assistant", content: final.content },
          {
            role: "user",
            content: [
              { type: "tool_result", tool_use_id: "tool_2", content: "ok" },
            ],
          },
        ],
      },
      localModel,
    );
    assert.ok(replay.ok);
    const replayed = (await replay.value.options.onPayload?.(
      { model: model.id, messages: [], max_tokens: 2048, stream: true },
      localModel,
    )) as { messages: { content: unknown }[] };
    assert.deepEqual(replayed.messages[1]?.content, final.content);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test("SSE keeps interleaved block indices separate and flushes blocks with no deltas", () => {
  const partial = assistant([
    { type: "text", text: "hello" },
    { type: "toolCall", id: "t1", name: "bash", arguments: {} },
  ]);
  const events: AssistantMessageEvent[] = [
    { type: "text_start", contentIndex: 0, partial },
    { type: "toolcall_start", contentIndex: 1, partial },
    { type: "text_end", contentIndex: 0, partial, content: "hello" },
    {
      type: "toolcall_end",
      contentIndex: 1,
      partial,
      toolCall: partial.content[1] as ToolCall,
    },
    { type: "done", reason: "toolUse", message: partial },
  ];
  const frames = events.flatMap(createMessagesEncoder("a", "i"));
  assert.deepEqual(frames.slice(1, -2), [
    wire("content_block_start", {
      index: 0,
      content_block: { type: "text", text: "" },
    }),
    wire("content_block_start", {
      index: 1,
      content_block: { type: "tool_use", id: "t1", name: "bash", input: {} },
    }),
    wire("content_block_delta", {
      index: 0,
      delta: { type: "text_delta", text: "hello" },
    }),
    wire("content_block_stop", { index: 0 }),
    wire("content_block_delta", {
      index: 1,
      delta: { type: "input_json_delta", partial_json: "{}" },
    }),
    wire("content_block_stop", { index: 1 }),
  ]);
});

test("SSE fails closed when Pi cannot reconstruct an upstream initial text prefix", () => {
  const partial = assistant([{ type: "text", text: "initial suffix" }]);
  const encode = createMessagesEncoder("a", "i");
  const frames = [
    encode({ type: "text_start", contentIndex: 0, partial }),
    encode({ type: "text_delta", contentIndex: 0, partial, delta: " suffix" }),
    encode({
      type: "text_end",
      contentIndex: 0,
      partial,
      content: "initial suffix",
    }),
    encode({ type: "done", reason: "toolUse", message: partial }),
  ].flat();
  assert.equal(frames.at(-1)?.event, "error");
  assert.deepEqual(frames.at(-1)?.error, {
    status: 502,
    type: "api_error",
    message: "Unsupported upstream content stream",
  });
  assert.equal(
    frames.some((frame) => frame.event === "message_stop"),
    false,
  );
  assert.doesNotMatch(JSON.stringify(frames.at(-1)), /initial suffix/);
});

test("localhost provider errors never become native response error details", async () => {
  const server = createServer((req, res) => {
    req.resume();
    res.writeHead(400, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        type: "error",
        error: {
          type: "invalid_request_error",
          message: "private-provider-detail",
        },
      }),
    );
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  assert.ok(address && typeof address !== "string");
  try {
    const localModel = {
      ...model,
      baseUrl: `http://127.0.0.1:${address.port}`,
    };
    const models = createModels();
    models.setProvider(
      createProvider({
        id: model.provider,
        models: [localModel],
        api: anthropicMessagesApi(),
        auth: {
          apiKey: {
            name: "Fixture",
            resolve: async () => ({ auth: { apiKey: "local-test-key" } }),
          },
        },
      }),
    );
    const prepared = prepareMessages(request, localModel);
    assert.ok(prepared.ok);
    const stream = models.streamSimple(localModel, prepared.value.context, {
      ...prepared.value.options,
      timeoutMs: 2000,
      maxRetries: 0,
    });
    const encode = createMessagesEncoder("a", "i");
    const frames: WireEvent[] = [];
    for await (const event of stream) frames.push(...encode(event));
    const result = await stream.result();
    assert.equal(result.stopReason, "error");
    assert.match(result.errorMessage ?? "", /private-provider-detail/);
    assert.deepEqual(frames, []);
    assert.deepEqual(messagesResponse(result, "a", "i"), {
      type: "error",
      error: { type: "api_error", message: "Upstream inference failed" },
    });
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test("actual Pi client payload passes the LiteLLM cache callback and native adapter unchanged", async () => {
  const received: Record<string, unknown>[] = [];
  const upstream = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    received.push(
      JSON.parse(Buffer.concat(chunks).toString()) as Record<string, unknown>,
    );
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.end(fixtureResponse);
  });
  upstream.listen(0, "127.0.0.1");
  await once(upstream, "listening");
  const upstreamAddress = upstream.address();
  assert.ok(upstreamAddress && typeof upstreamAddress !== "string");
  const backendModel = {
    ...model,
    baseUrl: `http://127.0.0.1:${upstreamAddress.port}`,
    compat: { supportsStrictTools: true },
  };
  const backend = createModels();
  backend.setProvider(
    createProvider({
      id: model.provider,
      models: [backendModel],
      api: anthropicMessagesApi(),
      auth: {
        apiKey: {
          name: "Fixture",
          resolve: async () => ({ auth: { apiKey: "fixture-key" } }),
        },
      },
    }),
  );
  const clientPayloads: Record<string, unknown>[] = [];
  const bridge = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    const payload = JSON.parse(Buffer.concat(chunks).toString()) as Record<
      string,
      unknown
    >;
    clientPayloads.push(payload);
    const cache_control =
      clientPayloads.length % 2 ? { type: "ephemeral" } : null;
    const prepared = prepareMessages(
      { ...payload, cache_control },
      backendModel,
    );
    if (!prepared.ok) {
      res.writeHead(prepared.error.status, {
        "content-type": "application/json",
      });
      res.end(JSON.stringify({ type: "error", error: prepared.error }));
      return;
    }
    const encode = createMessagesEncoder("public-alias", "msg_bridge");
    res.writeHead(200, { "content-type": "text/event-stream" });
    const stream = backend.streamSimple(backendModel, prepared.value.context, {
      ...prepared.value.options,
      timeoutMs: 2000,
      maxRetries: 0,
    });
    for await (const event of stream)
      for (const frame of encode(event))
        res.write(
          `event: ${frame.event}\ndata: ${JSON.stringify(frame.data)}\n\n`,
        );
    res.end();
  });
  bridge.listen(0, "127.0.0.1");
  await once(bridge, "listening");
  const bridgeAddress = bridge.address();
  assert.ok(bridgeAddress && typeof bridgeAddress !== "string");
  try {
    const parameters = {
      type: "object",
      // pi-ai >= 0.85 refuses `not` and `patternProperties` for strict tools
      // client-side, so the fixture keeps only keywords a real client can send.
      properties: { command: { type: "string", enum: ["ls", "pwd"] } },
      required: ["command"],
      additionalProperties: false,
      $comment: "Retain schema annotations",
      dependentRequired: { command: ["cwd"] },
    };
    for (const adaptive of [false, true])
      for (const _cacheVariant of [false, true]) {
        const clientModel = {
          ...model,
          id: "public-alias",
          baseUrl: `http://127.0.0.1:${bridgeAddress.port}`,
          compat: {
            supportsStrictTools: true,
            forceAdaptiveThinking: adaptive,
          },
        };
        const client = createModels();
        client.setProvider(
          createProvider({
            id: model.provider,
            models: [clientModel],
            api: anthropicMessagesApi(),
            auth: {
              apiKey: {
                name: "Fixture",
                resolve: async () => ({ auth: { apiKey: "fixture-key" } }),
              },
            },
          }),
        );
        const message = await client.completeSimple(
          clientModel,
          {
            systemPrompt: "Test system",
            messages: [{ role: "user", content: "Hello", timestamp: 0 }],
            tools: [
              {
                name: "bash",
                description: "Run command",
                parameters,
                constrainedSampling: { type: "json_schema", strict: "require" },
              },
            ],
          },
          {
            reasoning: "medium",
            thinkingBudgets: { medium: 1024 },
            maxTokens: 4096,
            timeoutMs: 2000,
            maxRetries: 0,
          },
        );
        assert.equal(message.stopReason, "stop", message.errorMessage);
        const incoming = clientPayloads.at(-1)!;
        const outgoing = received.at(-1)!;
        assert.deepEqual(
          incoming.thinking,
          adaptive
            ? { type: "adaptive", display: "summarized" }
            : { type: "enabled", budget_tokens: 1024, display: "summarized" },
        );
        assert.equal(
          (incoming.tools as Record<string, unknown>[])[0]
            ?.eager_input_streaming,
          true,
        );
        assert.equal(
          (incoming.tools as Record<string, unknown>[])[0]?.strict,
          true,
        );
        assert.deepEqual(
          (incoming.tools as Record<string, unknown>[])[0]?.input_schema,
          parameters,
        );
        assert.deepEqual(outgoing, {
          ...incoming,
          model: backendModel.id,
          cache_control: received.length % 2 ? { type: "ephemeral" } : null,
        });
      }
  } finally {
    for (const server of [bridge, upstream]) {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  }
});

test("stop sequences fail explicitly instead of discarding the matched sequence", () => {
  for (const stop_sequences of [["STOP"], []]) {
    const result = prepareMessages({ ...request, stop_sequences }, model);
    assert.equal(result.ok, false);
    if (!result.ok) {
      assert.equal(result.error.status, 400);
      assert.match(
        result.error.message,
        /stop_sequences.*not preserve.*matched/i,
      );
    }
  }
});

test("policy refusal remains a native message without exposing Pi's error details", () => {
  const refused = {
    ...assistant([{ type: "text", text: "I cannot help with that." }]),
    stopReason: "error" as const,
    rawStopReason: "refusal",
    errorMessage: "private policy explanation",
  };
  const response = messagesResponse(refused, "a", "i") as Record<
    string,
    unknown
  >;
  assert.equal(response.type, "message");
  assert.equal(response.stop_reason, "refusal");
  assert.doesNotMatch(JSON.stringify(response), /private policy explanation/);
  const frames = createMessagesEncoder(
    "a",
    "i",
  )({
    type: "done",
    reason: "stop",
    message: { ...refused, stopReason: "stop" },
  });
  assert.deepEqual(
    frames.at(-2),
    wire("message_delta", {
      delta: { stop_reason: "refusal", stop_sequence: null },
      usage,
    }),
  );
});

test("native field overrides remain strict and preserve explicit false, omitted display and null cache", async () => {
  const body = {
    ...request,
    cache_control: null,
    thinking: { type: "adaptive", display: "omitted" },
    tools: [
      {
        ...nativeRequest.tools[0],
        eager_input_streaming: false,
        strict: false,
      },
    ],
  };
  const prepared = prepareMessages(body, model);
  assert.ok(prepared.ok);
  const payload = (await prepared.value.options.onPayload?.(
    {
      model: model.id,
      messages: [],
      tools: [{ name: "bash", eager_input_streaming: true, strict: true }],
      stream: true,
    },
    model,
  )) as Record<string, unknown>;
  assert.deepEqual(payload.thinking, body.thinking);
  assert.deepEqual(payload.tools, body.tools);
  assert.equal(payload.cache_control, null);
  for (const invalidBody of [
    { ...request, cache_control: { type: "ephemeral", ttl: "2h" } },
    { ...request, cache_control: { type: "ephemeral", extra: true } },
    {
      ...request,
      thinking: { type: "enabled", budget_tokens: 1024, display: "verbose" },
    },
    { ...request, thinking: { type: "adaptive", display: false } },
    {
      ...request,
      tools: [{ ...nativeRequest.tools[0], eager_input_streaming: "true" }],
    },
    { ...request, tools: [{ ...nativeRequest.tools[0], strict: 1 }] },
  ])
    assert.equal(prepareMessages(invalidBody, model).ok, false);
});
