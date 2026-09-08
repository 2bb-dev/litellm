import assert from "node:assert/strict";
import test from "node:test";
import {
  createAssistantMessageEventStream,
  Type,
  type AnthropicOptions,
  type Api,
  type AssistantMessage,
  type AssistantMessageEvent,
  type AssistantMessageEventStream,
  type Context,
  type Model,
  type Provider,
  type StreamOptions,
} from "@earendil-works/pi-ai";
import { withClaudeOAuthCompatibility } from "../src/claude-oauth.js";

const model: Model<"anthropic-messages"> = {
  id: "claude-offline-fixture",
  name: "Offline fixture",
  api: "anthropic-messages",
  provider: "anthropic",
  baseUrl: "https://offline.invalid",
  reasoning: true,
  input: ["text"],
  contextWindow: 8192,
  maxTokens: 1024,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
// Deliberately not a prefix: compatibility follows the resolved key's includes check.
const apiKey = "resolved:sk-ant-oat-offline-fixture";
const originalName = "lookup_weather";
const alias = "mcp__pi__lookup_weather";
type Method = "stream" | "streamSimple";

function message(names = [originalName]): AssistantMessage {
  return {
    role: "assistant",
    api: model.api,
    provider: model.provider,
    model: model.id,
    responseId: "msg_fixture",
    responseModel: "resolved-fixture-model",
    timestamp: 123,
    stopReason: "toolUse",
    rawStopReason: "tool_use",
    content: [
      {
        type: "text",
        text: `pi itself ${alias}`,
        textSignature: "text-signature",
      },
      {
        type: "thinking",
        thinking: "signed pi packages",
        thinkingSignature: "signed-fixture",
      },
      {
        type: "thinking",
        thinking: "",
        redacted: true,
        thinkingSignature: "opaque-fixture",
      },
      ...names.map((name, index) => ({
        type: "toolCall" as const,
        id: `call_${index}`,
        name,
        arguments: {
          nested: { text: `pi .md files ${alias}`, values: [1, null, true] },
        },
        thoughtSignature: "tool-signature",
      })),
    ],
    usage: {
      input: 11,
      output: 7,
      cacheRead: 5,
      cacheWrite: 3,
      cacheWrite1h: 2,
      reasoning: 4,
      totalTokens: 26,
      cost: {
        input: 0.11,
        output: 0.07,
        cacheRead: 0.05,
        cacheWrite: 0.03,
        total: 0.26,
      },
    },
  };
}

function fixture(
  source = createAssistantMessageEventStream(),
  id = "anthropic",
) {
  const calls: {
    method: Method;
    model: Model<Api>;
    context: Context;
    options?: StreamOptions;
  }[] = [];
  const provider: Provider = {
    id,
    name: "Offline provider",
    auth: {
      apiKey: {
        name: "Fixture",
        resolve: async () => assert.fail("Decorator must not resolve auth"),
      },
    },
    getModels: () => [model],
    stream(model, context, options) {
      calls.push({ method: "stream", model, context, options });
      return source;
    },
    streamSimple(model, context, options) {
      calls.push({ method: "streamSimple", model, context, options });
      return source;
    },
  };
  return {
    provider,
    source,
    calls,
    wrapped: withClaudeOAuthCompatibility(provider),
  };
}

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function collect(stream: AssistantMessageEventStream) {
  const events: AssistantMessageEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

test("full stream aliases forced custom tool choice without changing other provider inputs", async () => {
  const f = fixture();
  const context: Context = {
    systemPrompt:
      "pi itself belongs to payload rewriting, not context rewriting",
    tools: [
      {
        name: originalName,
        description: "pi packages",
        parameters: Type.Object({ city: Type.String() }),
      },
    ],
    messages: [message()],
  };
  const before = structuredClone(context);
  const controller = new AbortController();
  const choice = {
    type: "tool" as const,
    name: originalName,
    disable_parallel_tool_use: true,
  };
  const options: AnthropicOptions = {
    apiKey,
    toolChoice: choice,
    signal: controller.signal,
    thinkingEnabled: true,
    thinkingBudgetTokens: 512,
    maxTokens: 1024,
    temperature: 0.2,
    headers: { "x-fixture": "unchanged" },
    onResponse: async () => undefined,
    fetch: async () =>
      assert.fail("No network is allowed in provider contract tests"),
  };
  const output = f.wrapped.stream(model, context, options);
  assert.notEqual(output, f.source);
  assert.equal(f.calls.length, 1);
  const call = f.calls[0]!;
  assert.equal(call.method, "stream");
  assert.equal(call.model, model);
  assert.notEqual(call.context, context);
  assert.deepEqual(call.context, {
    ...before,
    tools: [{ ...before.tools![0]!, name: alias }],
    messages: [message([alias])],
  });
  assert.deepEqual(call.options, {
    ...options,
    toolChoice: { ...choice, name: alias },
    onPayload: call.options?.onPayload,
  });
  assert.notEqual(call.options, options);
  assert.equal(call.options?.signal, options.signal);
  assert.equal(call.options?.onResponse, options.onResponse);
  assert.equal(call.options?.fetch, options.fetch);
  assert.equal(typeof call.options?.onPayload, "function");
  assert.deepEqual(
    await call.options!.onPayload!(
      { system: "pi itself", messages: [] },
      model,
    ),
    {
      system: "the cli itself",
      messages: [],
    },
  );
  assert.equal(options.toolChoice, choice);
  assert.equal(choice.name, originalName);
  assert.equal(options.onPayload, undefined);
  assert.deepEqual(context, before);
  assert.equal(f.wrapped.auth, f.provider.auth);
  assert.equal(f.wrapped.getModels, f.provider.getModels);

  f.source.push({ type: "done", reason: "toolUse", message: message([alias]) });
  f.source.end();
  assert.deepEqual(await collect(output), [
    { type: "done", reason: "toolUse", message: message() },
  ]);
  assert.deepEqual(await output.result(), message());
});

test("payload callback is awaited once before system-only rewriting, distinguishing undefined and null", async () => {
  for (const mode of ["undefined", "null", "replacement"] as const) {
    const f = fixture();
    const gate = deferred();
    const resolved = { ...model, id: "callback-resolved-model" };
    const payload = {
      system: "original pi packages",
      messages: [{ role: "user", content: "pi itself and pi .md files" }],
      tools: [{ name: originalName, description: "pi packages" }],
      metadata: { text: "pi itself" },
    };
    const replacement = {
      ...payload,
      system: [
        {
          type: "text",
          text: "overlay pi itself; pi .md files; pi packages",
          cache_control: { type: "ephemeral" },
        },
        {
          type: "thinking",
          text: "pi itself",
          signature: "signed-pi-packages",
        },
        { type: "redacted_thinking", data: "opaque-pi-packages" },
      ],
    };
    const replacementBefore = structuredClone(replacement);
    let calls = 0;
    const options: StreamOptions = {
      apiKey,
      onPayload: async (received, receivedModel) => {
        calls++;
        assert.equal(received, payload);
        assert.equal(receivedModel, resolved);
        assert.equal(payload.system, "original pi packages");
        await gate.promise;
        if (mode === "undefined") {
          payload.system = "mutated pi itself; pi .md files; pi packages";
          return undefined;
        }
        return mode === "null" ? null : replacement;
      },
    };
    const output = f.wrapped.streamSimple(model, { messages: [] }, options);
    const forwarded = f.calls[0]!.options!;
    assert.notEqual(forwarded.onPayload, options.onPayload);
    const pending = forwarded.onPayload!(payload, resolved);
    assert.equal(calls, 1);
    gate.resolve();
    const rewritten = await pending;
    assert.equal(calls, 1);
    if (mode === "null") {
      assert.equal(rewritten, null);
    } else if (mode === "undefined") {
      assert.deepEqual(rewritten, {
        ...payload,
        system: "mutated the cli itself; cli .md files; cli packages",
      });
      assert.equal(
        payload.system,
        "mutated pi itself; pi .md files; pi packages",
      );
    } else {
      assert.deepEqual(rewritten, {
        ...replacementBefore,
        system: [
          {
            ...replacementBefore.system[0],
            text: "overlay the cli itself; cli .md files; cli packages",
          },
          ...replacementBefore.system.slice(1),
        ],
      });
    }
    assert.deepEqual(replacement, replacementBefore);
    if (mode !== "undefined")
      assert.equal(payload.system, "original pi packages");
    f.source.end(message());
    await collect(output);
    await output.result();
    assert.equal(calls, 1);
    assert.equal(f.calls.length, 1);
  }
});

test("history-only assistant, result, and addedToolNames aliases round-trip without active tools", async () => {
  const f = fixture();
  const names = [
    "history.only",
    "result.only",
    "added.only",
    "Bash",
    "mcp__foreign__tool",
  ];
  const aliases = [
    "mcp__pi__history_only",
    "mcp__pi__result_only",
    "mcp__pi__added_only",
    "Bash",
    "mcp__foreign__tool",
  ];
  const context: Context = {
    messages: [
      { role: "user", content: "pi itself", timestamp: 1 },
      message([names[0]!]),
      {
        role: "toolResult",
        toolCallId: "earlier_call",
        toolName: names[1]!,
        addedToolNames: names.slice(2),
        content: [{ type: "text", text: "pi packages" }],
        details: { name: "added.only" },
        isError: false,
        timestamp: 124,
      },
    ],
  };
  const before = structuredClone(context);
  const output = f.wrapped.streamSimple(model, context, {
    apiKey,
    reasoning: "high",
  });
  const forwarded = f.calls[0]!.context;
  assert.equal(forwarded.tools, undefined);
  assert.deepEqual(forwarded.messages, [
    before.messages[0],
    message([aliases[0]!]),
    {
      ...before.messages[2],
      toolName: aliases[1],
      addedToolNames: aliases.slice(2),
    },
  ]);
  assert.deepEqual(context, before);
  f.source.push({ type: "done", reason: "toolUse", message: message(aliases) });
  f.source.end();
  assert.deepEqual(await collect(output), [
    { type: "done", reason: "toolUse", message: message(names) },
  ]);
  assert.deepEqual(await output.result(), message(names));
});

for (const method of ["stream", "streamSimple"] as const) {
  test(`${method} maps every event without mutating the SDK's shared mutable message`, async () => {
    const f = fixture();
    const output = f.wrapped[method](
      model,
      {
        messages: [],
        tools: [
          {
            name: originalName,
            description: "fixture",
            parameters: Type.Object({}),
          },
        ],
      },
      { apiKey },
    );
    assert.equal(f.calls[0]!.method, method);
    const iterator = output[Symbol.asyncIterator]();
    const partial = message([alias, "Bash", "mcp__foreign__tool"]);
    partial.stopReason = "pending";
    const toolCall = partial.content[3]!;
    assert(toolCall.type === "toolCall");
    const events: AssistantMessageEvent[] = [
      { type: "start", partial },
      { type: "text_start", contentIndex: 0, partial },
      {
        type: "text_delta",
        contentIndex: 0,
        delta: `pi itself ${alias}`,
        partial,
      },
      {
        type: "text_end",
        contentIndex: 0,
        content: `pi itself ${alias}`,
        partial,
      },
      { type: "thinking_start", contentIndex: 1, partial },
      {
        type: "thinking_delta",
        contentIndex: 1,
        delta: "signed pi packages",
        partial,
      },
      {
        type: "thinking_end",
        contentIndex: 1,
        content: "signed pi packages",
        partial,
      },
      { type: "toolcall_start", contentIndex: 3, partial },
      {
        type: "toolcall_delta",
        contentIndex: 3,
        delta: `{"name":"${alias}"}`,
        partial,
      },
      { type: "toolcall_end", contentIndex: 3, toolCall, partial },
    ];
    for (const event of events) {
      // pi-ai mutates and reuses partials. Check each observation, not frozen historical snapshots.
      partial.usage.output++;
      toolCall.arguments.step = partial.usage.output;
      const before = structuredClone(event);
      const expected = structuredClone(partial);
      const expectedCall = expected.content[3]!;
      assert(expectedCall.type === "toolCall");
      expectedCall.name = originalName;
      f.source.push(event);
      const received = await iterator.next();
      assert.equal(received.done, false);
      assert.deepEqual(received.value, {
        ...before,
        partial: expected,
        ...(event.type === "toolcall_end" ? { toolCall: expectedCall } : {}),
      });
      assert.deepEqual(
        event,
        before,
        `${event.type} must not mutate the provider's event`,
      );
      assert("partial" in received.value);
      assert.notEqual(received.value.partial, partial);
      assert.notEqual(received.value.partial.content, partial.content);
      assert.notEqual(received.value.partial.content[3], toolCall);
      if (received.value.type === "toolcall_end")
        assert.notEqual(received.value.toolCall, toolCall);
    }
    partial.stopReason = method === "stream" ? "toolUse" : "error";
    if (method === "streamSimple")
      partial.errorMessage = "upstream fixture error";
    const before = structuredClone(partial);
    const expected = structuredClone(partial);
    const expectedCall = expected.content[3]!;
    assert(expectedCall.type === "toolCall");
    expectedCall.name = originalName;
    const terminal: AssistantMessageEvent =
      method === "stream"
        ? { type: "done", reason: "toolUse", message: partial }
        : { type: "error", reason: "error", error: partial };
    f.source.push(terminal);
    f.source.end();
    const received = await iterator.next();
    assert.equal(received.done, false);
    assert.deepEqual(
      received.value,
      method === "stream"
        ? { type: "done", reason: "toolUse", message: expected }
        : { type: "error", reason: "error", error: expected },
    );
    assert.equal((await iterator.next()).done, true);
    assert.deepEqual(await output.result(), expected);
    assert.equal(await f.source.result(), partial);
    assert.deepEqual(partial, before);
    assert.equal(f.calls.length, 1);
  });
}

test("result-only SDK streams finish with restored tool names even without a terminal event", async () => {
  const f = fixture();
  const output = f.wrapped.streamSimple(
    model,
    { messages: [message()] },
    { apiKey },
  );
  const final = message([alias]);
  const before = structuredClone(final);
  const result = output.result();
  f.source.end(final);
  assert.deepEqual(await collect(output), []);
  assert.deepEqual(await result, message());
  assert.notEqual(await result, final);
  assert.equal(await f.source.result(), final);
  assert.deepEqual(final, before);
});

test("provider iteration failures terminate the wrapper and honor the forwarded abort signal", async () => {
  for (const aborted of [false, true]) {
    const source = createAssistantMessageEventStream();
    const fail = deferred();
    // Custom providers can reject iteration, unlike the SDK's normal error events.
    source[Symbol.asyncIterator] = async function* () {
      yield { type: "start", partial: message([alias]) };
      await fail.promise;
      throw new Error("private upstream diagnostic");
    };
    const f = fixture(source);
    const controller = new AbortController();
    const output = f.wrapped.streamSimple(
      model,
      { messages: [message()] },
      { apiKey, signal: controller.signal },
    );
    assert.equal(f.calls[0]!.options?.signal, controller.signal);
    const iterator = output[Symbol.asyncIterator]();
    assert.deepEqual((await iterator.next()).value, {
      type: "start",
      partial: message(),
    });
    if (aborted) controller.abort();
    fail.resolve();
    const terminal = (await iterator.next()).value;
    assert(terminal?.type === "error");
    const reason = aborted ? "aborted" : "error";
    assert.equal(terminal.reason, reason);
    assert.equal(terminal.error.stopReason, reason);
    assert.equal(terminal.error.api, model.api);
    assert.equal(terminal.error.provider, model.provider);
    assert.equal(terminal.error.model, model.id);
    assert.deepEqual(terminal.error.content, []);
    assert.equal(
      terminal.error.errorMessage,
      "Anthropic OAuth compatibility stream failed",
    );
    assert.equal((await iterator.next()).done, true);
    assert.equal(await output.result(), terminal.error);
  }
});

test("explicit clients and ineligible provider/API/keys bypass both methods by identity", async () => {
  // Only identity is relevant: no SDK client methods may be called by this decorator.
  const client = {} as NonNullable<AnthropicOptions["client"]>;
  const onPayload = () => assert.fail("Bypass must not invoke callbacks");
  const cases: {
    id?: string;
    model?: Model<Api>;
    options?: AnthropicOptions;
  }[] = [
    { options: { apiKey, client, onPayload } },
    {
      model: { ...model, api: "openai-completions" },
      options: { apiKey, onPayload },
    },
    { id: "other-anthropic", options: { apiKey, onPayload } },
    { options: { apiKey: "sk-ant-api-fixture", onPayload } },
    { options: { onPayload } },
    {},
  ];
  for (const entry of cases) {
    for (const method of ["stream", "streamSimple"] as const) {
      const f = fixture(undefined, entry.id);
      const context: Context = {
        systemPrompt: "pi itself",
        messages: [message()],
      };
      const selectedModel = entry.model ?? model;
      const output = f.wrapped[method](selectedModel, context, entry.options);
      assert.equal(output, f.source);
      assert.equal(f.calls.length, 1);
      assert.equal(f.calls[0]!.method, method);
      assert.equal(f.calls[0]!.model, selectedModel);
      assert.equal(f.calls[0]!.context, context);
      assert.equal(f.calls[0]!.options, entry.options);
      if (entry.id) assert.equal(f.wrapped, f.provider);
      f.source.end(message());
      assert.deepEqual(await output.result(), message());
    }
  }
});
