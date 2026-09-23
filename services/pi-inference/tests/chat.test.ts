import assert from "node:assert/strict";
import { test } from "node:test";
import { getBuiltinModel } from "@earendil-works/pi-ai/providers/all";
import { chatResponse, createChatEncoder, prepareChat } from "../src/chat.js";
import { emptyUsage } from "../src/protocol.js";

const model = getBuiltinModel("anthropic", "claude-haiku-4-5");

test("chat preserves tool history without executing it and totals cached input once", () => {
  const prepared = prepareChat(
    {
      model: "claude",
      messages: [
        { role: "system", content: "Be brief" },
        { role: "user", content: "time?" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "clock", arguments: "{}" },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: "12:00" },
      ],
    },
    model,
  );
  assert.equal(prepared.ok, true);
  if (!prepared.ok) return;
  assert.equal(prepared.value.context.systemPrompt, "Be brief");
  assert.deepEqual(
    prepared.value.context.messages.map((m) => m.role),
    ["user", "assistant", "toolResult"],
  );
  assert.equal(
    prepared.value.context.messages[2]?.role === "toolResult" &&
      prepared.value.context.messages[2].toolName,
    "clock",
  );
  const response = chatResponse(
    {
      role: "assistant",
      content: [{ type: "text", text: "12:00" }],
      model: model.id,
      provider: model.provider,
      api: model.api,
      timestamp: 0,
      stopReason: "stop",
      usage: {
        ...emptyUsage(),
        input: 10,
        cacheRead: 20,
        cacheWrite: 5,
        output: 4,
        reasoning: 2,
        totalTokens: 39,
      },
    },
    "claude",
    "req_1",
  );
  assert.equal(response.usage.prompt_tokens, 35);
  assert.equal(response.usage.completion_tokens, 4);
  assert.equal(response.usage.total_tokens, 39);
  assert.equal(response.model, "claude");
});

test("unsupported fields and incomplete tool histories fail before provider dispatch", () => {
  for (const extra of [
    { n: 2 },
    { response_format: { type: "json_object" } },
    { api_key: "do-not-accept" },
    { api_base: "https://arbitrary-host" },
  ]) {
    assert.equal(
      prepareChat(
        {
          model: "claude",
          messages: [{ role: "user", content: "hi" }],
          ...extra,
        },
        model,
      ).ok,
      false,
    );
  }
  assert.equal(
    prepareChat(
      {
        model: "claude",
        messages: [
          { role: "tool", tool_call_id: "missing", content: "fake result" },
        ],
      },
      model,
    ).ok,
    false,
  );
  assert.equal(
    prepareChat(
      {
        model: "claude",
        messages: [
          { role: "user", content: "time" },
          {
            role: "assistant",
            tool_calls: [
              {
                id: "unanswered",
                type: "function",
                function: { name: "clock", arguments: "{}" },
              },
            ],
          },
        ],
      },
      model,
    ).ok,
    false,
  );
});

test("tool streaming emits complete arguments once despite advanced mutable partial", () => {
  const tool = {
    type: "toolCall" as const,
    id: "call_1",
    name: "clock",
    arguments: { zone: "UTC" },
  };
  const partial = {
    role: "assistant" as const,
    content: [tool],
    api: model.api,
    provider: model.provider,
    model: model.id,
    timestamp: 0,
    stopReason: "toolUse" as const,
    usage: emptyUsage(),
  };
  const encode = createChatEncoder("claude", "req_1", true);
  assert.deepEqual(
    encode({ type: "toolcall_start", contentIndex: 0, partial }),
    [],
  );
  assert.deepEqual(
    encode({
      type: "toolcall_delta",
      contentIndex: 0,
      delta: '{"zone":"UTC"}',
      partial,
    }),
    [],
  );
  const emitted = JSON.stringify(
    encode({ type: "toolcall_end", contentIndex: 0, toolCall: tool, partial }),
  );
  assert.equal(emitted.split("call_1").length, 2);
  assert(emitted.includes("UTC"));
  const final = encode({ type: "done", reason: "toolUse", message: partial });
  assert(JSON.stringify(final).includes("tool_calls"));
  assert.equal(final.at(-1)?.data, "[DONE]");
});

test("chat rejects what Claude Opus 5.5 rejects instead of dropping it", () => {
  const opus = getBuiltinModel("anthropic", "claude-opus-5-5");
  const base = { model: "opus", messages: [{ role: "user", content: "hi" }] };
  for (const body of [
    { ...base, temperature: 0.2 },
    { ...base, reasoning_effort: "none" },
    { ...base, tool_choice: "required" },
  ]) {
    const prepared = prepareChat(body, opus);
    assert.equal(prepared.ok, false, JSON.stringify(body));
    if (!prepared.ok) assert.equal(prepared.error.status, 400);
  }
  const accepted = prepareChat({ ...base, reasoning_effort: "xhigh" }, opus);
  assert.equal(accepted.ok, true);
  if (accepted.ok) assert.equal(accepted.value.options.reasoning, "xhigh");
});
