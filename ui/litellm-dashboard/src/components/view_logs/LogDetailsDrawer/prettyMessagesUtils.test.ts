import { describe, it, expect } from "vitest";
import { parseMessages } from "./prettyMessagesUtils";

describe("parseMessages - Chat Completions", () => {
  it("parses messages[] request and choices[0].message response", () => {
    const { requestMessages, responseMessage } = parseMessages(
      { messages: [{ role: "user", content: "Hello" }] },
      { choices: [{ message: { role: "assistant", content: "Hi" } }] },
    );
    expect(requestMessages).toEqual([
      { role: "user", content: "Hello", toolCallId: undefined },
    ]);
    expect(responseMessage).toEqual({
      role: "assistant",
      content: "Hi",
      toolCalls: undefined,
    });
  });

  it("flattens vision-style content arrays", () => {
    const { requestMessages } = parseMessages(
      {
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Describe this" },
              { type: "image_url", image_url: { url: "..." } },
            ],
          },
        ],
      },
      {},
    );
    expect(requestMessages[0].content).toBe("Describe this\n[Image]");
  });

  it("parses tool_calls on the assistant response", () => {
    const { responseMessage } = parseMessages(
      {},
      {
        choices: [
          {
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_1",
                  function: { name: "get_weather", arguments: '{"city":"Paris"}' },
                },
              ],
            },
          },
        ],
      },
    );
    expect(responseMessage?.toolCalls).toEqual([
      { id: "call_1", name: "get_weather", arguments: { city: "Paris" } },
    ]);
  });
});

describe("parseMessages - Responses API", () => {
  it("parses string input as a single user message", () => {
    const { requestMessages } = parseMessages({ input: "hi" }, {});
    expect(requestMessages).toEqual([{ role: "user", content: "hi" }]);
  });

  it("parses input[] with input_text content parts", () => {
    const { requestMessages } = parseMessages(
      {
        input: [
          { role: "system", content: "You are helpful" },
          {
            role: "user",
            content: [{ type: "input_text", text: "Hello" }],
          },
        ],
      },
      {},
    );
    expect(requestMessages).toEqual([
      { role: "system", content: "You are helpful" },
      { role: "user", content: "Hello" },
    ]);
  });

  it("extracts assistant text from response.output[] with output_text", () => {
    const { responseMessage } = parseMessages(
      {},
      {
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "All done." }],
          },
        ],
      },
    );
    expect(responseMessage).toEqual({
      role: "assistant",
      content: "All done.",
      toolCalls: undefined,
    });
  });

  it("extracts function_call items from response.output[] as tool calls", () => {
    const { responseMessage } = parseMessages(
      {},
      {
        output: [
          {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "Running tool..." }],
          },
          {
            type: "function_call",
            call_id: "call_abc",
            name: "search",
            arguments: '{"q":"cats"}',
          },
        ],
      },
    );
    expect(responseMessage?.content).toBe("Running tool...");
    expect(responseMessage?.toolCalls).toEqual([
      { id: "call_abc", name: "search", arguments: { q: "cats" } },
    ]);
  });

  it("falls back to response.output_text when output[] is empty", () => {
    const { responseMessage } = parseMessages(
      {},
      { output: [], output_text: "hello world" },
    );
    expect(responseMessage).toEqual({
      role: "assistant",
      content: "hello world",
    });
  });

  it("parses function_call_output items in input[] as tool role messages", () => {
    const { requestMessages } = parseMessages(
      {
        input: [
          { role: "user", content: "call the tool" },
          {
            type: "function_call",
            call_id: "call_1",
            name: "search",
            arguments: '{"q":"x"}',
          },
          {
            type: "function_call_output",
            call_id: "call_1",
            output: "result body",
          },
        ],
      },
      {},
    );
    expect(requestMessages[1]).toEqual({
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_1", name: "search", arguments: { q: "x" } }],
    });
    expect(requestMessages[2]).toEqual({
      role: "tool",
      content: "result body",
      toolCallId: "call_1",
    });
  });

  it("returns null response when neither choices nor output is present", () => {
    const { responseMessage } = parseMessages({}, { status: "completed" });
    expect(responseMessage).toBeNull();
  });
});

describe("parseMessages - role preservation", () => {
  it("preserves the OpenAI `developer` role (does not coerce to user)", () => {
    const { requestMessages } = parseMessages(
      {
        messages: [
          { role: "developer", content: "Internal instructions" },
          { role: "user", content: "hi" },
        ],
      },
      {},
    );
    expect(requestMessages[0].role).toBe("developer");
    expect(requestMessages[1].role).toBe("user");
  });

  it("preserves `developer` role on the Responses API input path", () => {
    const { requestMessages } = parseMessages(
      {
        input: [
          { role: "developer", content: "Internal instructions" },
          { role: "user", content: "hi" },
        ],
      },
      {},
    );
    expect(requestMessages[0].role).toBe("developer");
  });

  it("falls back to `user` only for genuinely unknown roles", () => {
    const { requestMessages } = parseMessages(
      { messages: [{ role: "moderator", content: "hi" }] },
      {},
    );
    expect(requestMessages[0].role).toBe("user");
  });
});

describe("parseMessages - Anthropic-style system field", () => {
  it("promotes top-level system string to a system message", () => {
    const { requestMessages } = parseMessages(
      {
        system: "Be terse.",
        messages: [{ role: "user", content: "hi" }],
      },
      {},
    );
    expect(requestMessages[0]).toEqual({
      role: "system",
      content: "Be terse.",
    });
    expect(requestMessages[1].role).toBe("user");
  });

  it("flattens system content-part arrays", () => {
    const { requestMessages } = parseMessages(
      {
        system: [
          { type: "text", text: "Line 1" },
          { type: "text", text: "Line 2" },
        ],
        messages: [{ role: "user", content: "hi" }],
      },
      {},
    );
    expect(requestMessages[0].content).toBe("Line 1\nLine 2");
  });
});
