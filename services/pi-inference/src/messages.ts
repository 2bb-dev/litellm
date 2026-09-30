import { z } from "zod";
import { hasApi } from "@earendil-works/pi-ai";
import type {
  Api,
  AssistantMessage,
  Context,
  Message,
  Model,
  TextContent,
  Usage,
} from "@earendil-works/pi-ai";
import {
  emptyUsage,
  failed,
  invalid,
  type EventEncoder,
  type PreparedCall,
  type Result,
  type WireEvent,
} from "./protocol.js";

const text = z.string().max(32 * 1024 * 1024);
const name = z.string().regex(/^[a-zA-Z0-9_-]{1,64}$/);
const typeName = z.string().regex(/^[a-z][a-z0-9_]{0,63}$/);
// The backend reads these blocks; every other block reaches Anthropic unchanged.
const interpretedBlocks = new Set([
  "text",
  "tool_use",
  "tool_result",
  "thinking",
  "redacted_thinking",
  "tool_addition",
  "tool_removal",
]);
const forwardedBlock = z
  .looseObject({ type: typeName })
  .refine((block) => !interpretedBlocks.has(block.type));
const textBlock = z.looseObject({ type: z.literal("text"), text });
const inputBlocks = z.union([textBlock, forwardedBlock]);
const toolResult = z.looseObject({
  type: z.literal("tool_result"),
  tool_use_id: name,
  content: z.union([text, z.array(inputBlocks).max(256)]).optional(),
  is_error: z.boolean().optional(),
});
const assistantBlock = z.union([
  textBlock,
  z.looseObject({
    type: z.literal("tool_use"),
    id: name,
    name,
    input: z.record(z.string(), z.json()),
  }),
  z.looseObject({
    type: z.literal("thinking"),
    thinking: text,
    signature: text.min(1),
  }),
  z.looseObject({ type: z.literal("redacted_thinking"), data: text.min(1) }),
  forwardedBlock,
]);
const toolChange = <T extends "tool_addition" | "tool_removal">(type: T) =>
  z.looseObject({
    type: z.literal(type),
    tool: z.looseObject({ type: z.literal("tool_reference"), name }),
  });
const systemMessage = z.looseObject({
  role: z.literal("system"),
  content: z.union([
    text,
    z
      .array(
        z.union([
          textBlock,
          toolChange("tool_addition"),
          toolChange("tool_removal"),
          forwardedBlock,
        ]),
      )
      .max(4096),
  ]),
});
const customTool = z.looseObject({
  type: z.literal("custom").optional(),
  name,
  description: text.optional(),
  input_schema: z
    .record(z.string(), z.json())
    .refine((value) => value.type === "object"),
});
// Anthropic-defined client and server tools are declared by their versioned type.
const anthropicTool = z
  .looseObject({ type: typeName, name: name.optional() })
  .refine((tool) => tool.type !== "custom");
const requestSchema = z.looseObject({
  model: z.string().min(1).max(256),
  max_tokens: z.number().int().positive(),
  stream: z.boolean().optional(),
  messages: z
    .array(
      z.discriminatedUnion("role", [
        z.looseObject({
          role: z.literal("user"),
          content: z.union([
            text.min(1),
            z
              .array(z.union([textBlock, toolResult, forwardedBlock]))
              .min(1)
              .max(4096),
          ]),
        }),
        z.looseObject({
          role: z.literal("assistant"),
          content: z.union([
            text.min(1),
            z.array(assistantBlock).min(1).max(4096),
          ]),
        }),
        systemMessage,
      ]),
    )
    .min(1)
    .max(10_000)
    // Anthropic accepts only an effort-only system message as messages[0].
    .refine(
      ([first]) =>
        first?.role !== "system" ||
        (Array.isArray(first.content) && first.content.length === 0),
    ),
  system: z.union([text, z.array(textBlock).max(256)]).optional(),
  tools: z
    .array(z.union([customTool, anthropicTool]))
    .max(1000)
    .optional(),
  thinking: z
    .discriminatedUnion("type", [
      z.looseObject({
        type: z.literal("enabled"),
        budget_tokens: z.number().int().min(1024),
      }),
      z.looseObject({ type: z.literal("adaptive") }),
      z.looseObject({ type: z.literal("disabled") }),
    ])
    .optional(),
  output_config: z.looseObject({ effort: z.string().optional() }).optional(),
  tool_choice: z
    .discriminatedUnion("type", [
      z.looseObject({ type: z.literal("auto") }),
      z.looseObject({ type: z.literal("any") }),
      z.looseObject({ type: z.literal("tool"), name }),
      z.looseObject({ type: z.literal("none") }),
    ])
    .optional(),
  temperature: z.number().optional(),
  top_p: z.number().optional(),
  top_k: z.number().optional(),
});
type NativeRequest = z.infer<typeof requestSchema>;
type Tool = NonNullable<NativeRequest["tools"]>[number];
type CustomTool = z.infer<typeof customTool>;
const isCustomTool = (tool: Tool): tool is CustomTool =>
  tool.type === undefined || tool.type === "custom";
const is =
  <T extends string>(type: T) =>
  <B extends { type: string }>(block: B): block is Extract<B, { type: T }> =>
    block.type === type;
// Pi's context never reaches Anthropic: the upstream body carries the client's blocks.
const forwardedText = (block: { type: string }): TextContent => ({
  type: "text",
  text: `[${block.type}]`,
});
const inputBlock = (block: z.infer<typeof inputBlocks>): TextContent =>
  is("text")(block) ? { type: "text", text: block.text } : forwardedText(block);

function toContext(input: NativeRequest, model: Model<Api>): Context {
  const customTools = input.tools?.filter(isCustomTool) ?? [];
  const declaredTools = new Map(customTools.map((tool) => [tool.name, tool]));
  const anthropicTools = new Set(
    input.tools?.flatMap((tool) =>
      !isCustomTool(tool) && tool.name ? [tool.name] : [],
    ),
  );
  const lateNames = new Set(
    input.messages.flatMap((message) =>
      message.role === "system" && Array.isArray(message.content)
        ? message.content
            .filter(is("tool_addition"))
            .map((block) => block.tool.name)
        : [],
    ),
  );
  const calls = input.messages.flatMap((message) =>
    message.role === "assistant" && Array.isArray(message.content)
      ? message.content
          .filter(is("tool_use"))
          .filter((call) => !anthropicTools.has(call.name))
      : [],
  );
  const messages = input.messages.flatMap((message): Message[] => {
    if (message.role === "system") {
      const blocks = Array.isArray(message.content) ? message.content : [];
      return [
        {
          role: "system",
          content:
            typeof message.content === "string"
              ? message.content
              : blocks
                  .filter(is("text"))
                  .map((block) => block.text)
                  .join("\n\n"),
          toolsAdded: blocks.filter(is("tool_addition")).flatMap((block) => {
            const tool = declaredTools.get(block.tool.name);
            return tool
              ? [
                  {
                    name: tool.name,
                    description: tool.description ?? "",
                    parameters: tool.input_schema,
                  },
                ]
              : [];
          }),
          toolsRemoved: blocks
            .filter(is("tool_removal"))
            .filter((block) => !anthropicTools.has(block.tool.name))
            .map((block) => ({ name: block.tool.name })),
          timestamp: 0,
        },
      ];
    }
    if (message.role === "assistant") {
      const content: AssistantMessage["content"] =
        typeof message.content === "string"
          ? [{ type: "text", text: message.content }]
          : message.content.map((block) => {
              if (is("text")(block)) return { type: "text", text: block.text };
              if (is("tool_use")(block) && !anthropicTools.has(block.name))
                return {
                  type: "toolCall",
                  id: block.id,
                  name: block.name,
                  arguments: block.input,
                };
              if (is("thinking")(block))
                return {
                  type: "thinking",
                  thinking: block.thinking,
                  thinkingSignature: block.signature,
                };
              if (is("redacted_thinking")(block))
                return {
                  type: "thinking",
                  thinking: "",
                  thinkingSignature: block.data,
                  redacted: true,
                };
              return forwardedText(block);
            });
      return [
        {
          role: "assistant",
          content,
          api: model.api,
          provider: model.provider,
          model: model.id,
          stopReason: "stop",
          usage: emptyUsage(),
          timestamp: 0,
        },
      ];
    }
    if (typeof message.content === "string")
      return [{ ...message, content: message.content, timestamp: 0 }];
    return message.content.map(
      (block): Message =>
        !is("tool_result")(block)
          ? { role: "user", content: [inputBlock(block)], timestamp: 0 }
          : {
              role: "toolResult",
              toolCallId: block.tool_use_id,
              toolName:
                calls.find((call) => call.id === block.tool_use_id)?.name ?? "",
              content:
                typeof block.content === "string"
                  ? [{ type: "text", text: block.content }]
                  : (block.content ?? []).map(inputBlock),
              isError: block.is_error ?? false,
              timestamp: 0,
            },
    );
  });
  return {
    messages,
    systemPrompt:
      typeof input.system === "string"
        ? input.system
        : input.system?.map((block) => block.text).join("\n\n"),
    tools:
      input.tools &&
      customTools
        .filter(
          (tool) =>
            tool.name !== "__pi_deferred_placeholder__" &&
            !lateNames.has(tool.name),
        )
        .map((tool) => ({
          name: tool.name,
          description: tool.description ?? "",
          parameters: tool.input_schema,
        })),
  };
}

const generatedSchema = z.looseObject({
  system: z.array(z.looseObject({ type: z.string() })).optional(),
  tools: z.array(z.looseObject({ name: z.string() })).optional(),
  messages: z.array(
    z.looseObject({
      content: z.union([
        z.string(),
        z.array(
          z.looseObject({
            type: z.string(),
            id: z.string().optional(),
            name: z.string().optional(),
          }),
        ),
      ]),
    }),
  ),
});
function nativePayload(
  payload: unknown,
  input: NativeRequest,
  context: Context,
): unknown {
  const parsed = generatedSchema.safeParse(payload);
  if (!parsed.success) throw new Error("Unsupported native provider payload");
  // Request semantics are the client's; Pi's payload contributes only the envelope and OAuth transforms.
  const {
    system,
    tools,
    thinking: _thinking,
    output_config: _effort,
    fallbacks: _fallbacks,
    temperature: _temperature,
    tool_choice: _toolChoice,
    metadata: _metadata,
    ...generated
  } = parsed.data;
  const names = new Map(
    generated.messages.flatMap((message) =>
      typeof message.content === "string"
        ? []
        : message.content
            .filter((block) => block.type === "tool_use")
            .map((block) => [block.id, block.name]),
    ),
  );
  const {
    model: _model,
    stream: _stream,
    messages,
    system: nativeSystem,
    tools: nativeTools,
    ...fields
  } = input;
  const prefix = (system ?? []).slice(
    0,
    Math.max(0, (system?.length ?? 0) - (context.systemPrompt ? 1 : 0)),
  );
  const choice = input.tool_choice;
  const declaredNames = [
    ...(context.tools?.map((tool) => tool.name) ?? []),
    ...context.messages.flatMap((message) =>
      message.role === "system"
        ? (message.toolsAdded?.map((tool) => tool.name) ?? [])
        : [],
    ),
  ];
  const convertedTools = tools?.filter(
    (tool) => tool.name !== "__pi_deferred_placeholder__",
  );
  const convertedByName = new Map(
    declaredNames.map((name, index) => [name, convertedTools?.[index]]),
  );
  const nativeOutputTools: Record<string, unknown>[] | undefined =
    nativeTools?.map((tool) => {
      const converted = isCustomTool(tool)
        ? convertedByName.get(tool.name)
        : undefined;
      return converted
        ? {
            ...tool,
            name: converted.name,
            ...(converted.defer_loading ? { defer_loading: true } : {}),
          }
        : tool;
    });
  const placeholder = tools?.find(
    (tool) => tool.name === "__pi_deferred_placeholder__",
  );
  if (
    placeholder &&
    nativeOutputTools &&
    !nativeOutputTools.some((tool) => tool.name === placeholder.name)
  )
    nativeOutputTools.splice(context.tools?.length ?? 0, 0, placeholder);
  return {
    ...generated,
    ...fields,
    ...(choice?.type === "tool"
      ? {
          tool_choice: {
            ...choice,
            name: convertedByName.get(choice.name)?.name ?? choice.name,
          },
        }
      : {}),
    messages: messages.map((message) =>
      message.role !== "assistant" || typeof message.content === "string"
        ? message
        : {
            ...message,
            content: message.content.map((block) =>
              is("tool_use")(block)
                ? { ...block, name: names.get(block.id) ?? block.name }
                : block,
            ),
          },
    ),
    ...(nativeSystem !== undefined || prefix.length
      ? {
          system: [
            ...prefix,
            ...(typeof nativeSystem === "string"
              ? [{ type: "text", text: nativeSystem }]
              : (nativeSystem ?? [])),
          ],
        }
      : {}),
    ...(nativeOutputTools
      ? {
          tools: nativeOutputTools,
        }
      : {}),
  };
}

function boundedJson(
  value: unknown,
  depth = 0,
  budget = { remaining: 100_000 },
): boolean {
  if (depth > 32 || --budget.remaining < 0) return false;
  if (value === null || typeof value === "string" || typeof value === "boolean")
    return true;
  if (typeof value === "number") return Number.isFinite(value);
  if (typeof value !== "object") return false;
  return Object.values(value).every((child) =>
    boundedJson(child, depth + 1, budget),
  );
}

export function prepareMessages(
  body: unknown,
  model: Model<Api>,
): Result<PreparedCall> {
  if (!hasApi(model, "anthropic-messages"))
    return invalid("Messages requires a native Anthropic Messages model");
  if (!boundedJson(body))
    return invalid("Messages request exceeds JSON bounds");
  const parsed = requestSchema.safeParse(body);
  if (!parsed.success)
    return invalid("Invalid or unsupported Messages request");
  const input = parsed.data;
  const changedTools = new Set<string>();
  for (const message of input.messages) {
    if (message.role !== "system" || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!is("tool_addition")(block) && !is("tool_removal")(block)) continue;
      if (
        block.type === "tool_addition" &&
        (changedTools.has(block.tool.name) ||
          !input.tools?.some((tool) => tool.name === block.tool.name))
      )
        return invalid("Unsupported tool addition sequence");
      changedTools.add(block.tool.name);
    }
  }
  const alwaysThinks =
    model.compat?.forceAdaptiveThinking === true &&
    model.thinkingLevelMap?.off === null;
  const thinking =
    (input.thinking && input.thinking.type !== "disabled") ||
    (alwaysThinks && input.thinking === undefined);
  const choice = input.tool_choice;
  if (input.max_tokens > model.maxTokens)
    return invalid("max_tokens exceeds model limit");
  if (
    input.thinking?.type === "enabled" &&
    input.thinking.budget_tokens >= input.max_tokens
  )
    return invalid("Thinking budget must be less than max_tokens");
  if (
    (input.thinking?.type === "enabled" ||
      input.thinking?.type === "adaptive" ||
      input.output_config?.effort) &&
    !model.reasoning
  )
    return invalid("Model does not support thinking");
  if (
    input.thinking?.type === "disabled" &&
    model.thinkingLevelMap?.off === null
  )
    return invalid("Model cannot disable thinking");
  if (input.thinking?.type === "enabled" && model.compat?.forceAdaptiveThinking)
    return invalid("Model requires adaptive thinking");
  if (
    input.temperature !== undefined &&
    (thinking || model.compat?.supportsTemperature === false)
  )
    return invalid(
      "Temperature is unsupported with this model or thinking mode",
    );
  if (
    (input.top_p !== undefined || input.top_k !== undefined) &&
    model.compat?.supportsTemperature === false
  )
    return invalid("Sampling parameters are unsupported with this model");
  if (thinking && (choice?.type === "any" || choice?.type === "tool"))
    return invalid("Thinking cannot force tool use");
  if (choice && choice.type !== "none" && !input.tools?.length)
    return invalid("tool_choice requires tools");
  if (
    choice?.type === "tool" &&
    !input.tools?.some((tool) => tool.name === choice.name)
  )
    return invalid("tool_choice names an unknown tool");
  const toolNames =
    input.tools?.flatMap((tool) =>
      tool.name ? [tool.name.toLowerCase()] : [],
    ) ?? [];
  if (new Set(toolNames).size !== toolNames.length)
    return invalid("Tool names must be unique ignoring case");
  if (
    !model.input.includes("image") &&
    input.messages.some(
      (message) =>
        message.role === "user" &&
        Array.isArray(message.content) &&
        message.content.some(
          (block) =>
            block.type === "image" ||
            (is("tool_result")(block) &&
              Array.isArray(block.content) &&
              block.content.some((item) => item.type === "image")),
        ),
    )
  )
    return invalid("Model does not support images");
  const context = toContext(input, model);
  return {
    ok: true,
    value: {
      context,
      options: {
        maxTokens: input.max_tokens,
        cacheRetention: "none",
        onPayload: (payload) => nativePayload(payload, input, context),
      },
      stream: input.stream ?? false,
    },
  };
}

const nativeUsage = (usage: Usage) => ({
  input_tokens: usage.input,
  output_tokens: usage.output,
  cache_read_input_tokens: usage.cacheRead,
  cache_creation_input_tokens: usage.cacheWrite,
  ...(usage.cacheWrite1h !== undefined
    ? {
        cache_creation: {
          ephemeral_1h_input_tokens: usage.cacheWrite1h,
          ephemeral_5m_input_tokens: usage.cacheWrite - usage.cacheWrite1h,
        },
      }
    : {}),
  ...(usage.reasoning !== undefined
    ? { output_tokens_details: { thinking_tokens: usage.reasoning } }
    : {}),
});
const stopReason = (message: AssistantMessage): string | null => {
  if (
    message.rawStopReason &&
    [
      "end_turn",
      "max_tokens",
      "tool_use",
      "stop_sequence",
      "pause_turn",
      "refusal",
    ].includes(message.rawStopReason)
  )
    return message.rawStopReason;
  return message.stopReason === "stop"
    ? "end_turn"
    : message.stopReason === "length"
      ? "max_tokens"
      : message.stopReason === "toolUse"
        ? "tool_use"
        : null;
};
type OutputBlock = AssistantMessage["content"][number];
const nativeBlock = (block: OutputBlock, start = false): unknown => {
  if (block.type === "text")
    return { type: "text", text: start ? "" : block.text };
  if (block.type === "toolCall")
    return {
      type: "tool_use",
      id: block.id,
      name: block.name,
      input: start ? {} : block.arguments,
    };
  if (block.redacted)
    return { type: "redacted_thinking", data: block.thinkingSignature ?? "" };
  return {
    type: "thinking",
    thinking: start ? "" : block.thinking,
    signature: start ? "" : (block.thinkingSignature ?? ""),
  };
};
export function messagesResponse(
  message: AssistantMessage,
  alias: string,
  id: string,
): unknown {
  if (
    failed(message) &&
    !(message.stopReason === "error" && message.rawStopReason === "refusal")
  )
    return {
      type: "error",
      error: { type: "api_error", message: "Upstream inference failed" },
    };
  return {
    id,
    type: "message",
    role: "assistant",
    model: alias,
    content: message.content.map((block) => nativeBlock(block)),
    stop_reason: stopReason(message),
    stop_sequence: null,
    usage: nativeUsage(message.usage),
  };
}
const frame = (
  event: string,
  fields: Record<string, unknown> = {},
): WireEvent => ({ event, data: { type: event, ...fields } });
const deltaFrame = (index: number, delta: unknown): WireEvent =>
  frame("content_block_delta", { index, delta });

export function createMessagesEncoder(alias: string, id: string): EventEncoder {
  const states = new Map<
    number,
    { delta: boolean; closed: boolean; text: string }
  >();
  let started = false;
  let finished = false;
  const start = (index: number, block: OutputBlock): WireEvent[] => {
    if (states.has(index)) return [];
    states.set(index, { delta: false, closed: false, text: "" });
    return [
      frame("content_block_start", {
        index,
        content_block: nativeBlock(block, true),
      }),
    ];
  };
  const end = (
    index: number,
    block: OutputBlock,
    content?: string,
  ): WireEvent[] => {
    const prefix = start(index, block);
    const state = states.get(index)!;
    if (state.closed) return [];
    state.closed = true;
    const text =
      content ??
      (block.type === "text"
        ? block.text
        : block.type === "thinking"
          ? block.thinking
          : "");
    if (
      state.delta &&
      (block.type === "text" ||
        (block.type === "thinking" && !block.redacted)) &&
      state.text !== text
    ) {
      finished = true;
      const error = {
        type: "api_error",
        message: "Unsupported upstream content stream",
      };
      return [
        { ...frame("error", { error }), error: { status: 502, ...error } },
      ];
    }
    const delta =
      block.type === "toolCall"
        ? {
            type: "input_json_delta",
            partial_json: JSON.stringify(block.arguments),
          }
        : block.type === "thinking"
          ? { type: "thinking_delta", thinking: text }
          : { type: "text_delta", text };
    return [
      ...prefix,
      ...(!state.delta &&
      (block.type === "toolCall" || text) &&
      !(block.type === "thinking" && block.redacted)
        ? [deltaFrame(index, delta)]
        : []),
      ...(block.type === "thinking" &&
      !block.redacted &&
      block.thinkingSignature
        ? [
            deltaFrame(index, {
              type: "signature_delta",
              signature: block.thinkingSignature,
            }),
          ]
        : []),
      frame("content_block_stop", { index }),
    ];
  };
  return (event) => {
    if (finished) return [];
    if (event.type === "error") {
      finished = true;
      return [];
    }
    const prefix = started
      ? []
      : [
          frame("message_start", {
            message: {
              id,
              type: "message",
              role: "assistant",
              model: alias,
              content: [],
              stop_reason: null,
              stop_sequence: null,
              usage: nativeUsage(emptyUsage()),
            },
          }),
        ];
    started = true;
    if (event.type === "start") return prefix;
    if (event.type === "done") {
      const content = event.message.content.flatMap((block, index) =>
        finished ? [] : end(index, block),
      );
      if (finished) return [...prefix, ...content];
      finished = true;
      return [
        ...prefix,
        ...content,
        frame("message_delta", {
          delta: {
            stop_reason: stopReason(event.message),
            stop_sequence: null,
          },
          usage: nativeUsage(event.message.usage),
        }),
        frame("message_stop"),
      ];
    }
    const index = event.contentIndex;
    const block = event.partial.content[index];
    if (!block) return prefix;
    if (event.type.endsWith("_start"))
      return [...prefix, ...start(index, block)];
    if ("delta" in event) {
      const opening = start(index, block);
      const state = states.get(index)!;
      if (state.closed || (block.type === "thinking" && block.redacted))
        return [...prefix, ...opening];
      state.delta = true;
      if (event.type !== "toolcall_delta") state.text += event.delta;
      const delta =
        event.type === "text_delta"
          ? { type: "text_delta", text: event.delta }
          : event.type === "thinking_delta"
            ? { type: "thinking_delta", thinking: event.delta }
            : { type: "input_json_delta", partial_json: event.delta };
      return [...prefix, ...opening, deltaFrame(index, delta)];
    }
    return [
      ...prefix,
      ...end(index, block, "content" in event ? event.content : undefined),
    ];
  };
}
