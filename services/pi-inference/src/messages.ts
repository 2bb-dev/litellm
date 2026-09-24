import { z } from "zod";
import { hasApi } from "@earendil-works/pi-ai";
import type {
  Api,
  AssistantMessage,
  Context,
  ImageContent,
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
const cache = z.strictObject({
  type: z.literal("ephemeral"),
  ttl: z.enum(["5m", "1h"]).optional(),
});
const textBlock = z.strictObject({
  type: z.literal("text"),
  text,
  cache_control: cache.optional(),
});
const imageBlock = z.strictObject({
  type: z.literal("image"),
  source: z.strictObject({
    type: z.literal("base64"),
    media_type: z.enum(["image/jpeg", "image/png", "image/gif", "image/webp"]),
    data: z
      .string()
      .min(4)
      .max(7_000_000)
      .regex(
        /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/,
      ),
  }),
  cache_control: cache.optional(),
});
const toolResult = z.strictObject({
  type: z.literal("tool_result"),
  tool_use_id: name,
  content: z
    .union([text, z.array(z.union([textBlock, imageBlock])).max(256)])
    .optional(),
  is_error: z.boolean().optional(),
  cache_control: cache.optional(),
});
const assistantBlock = z.discriminatedUnion("type", [
  textBlock,
  z.strictObject({
    type: z.literal("tool_use"),
    id: name,
    name,
    input: z.record(z.string(), z.json()),
    cache_control: cache.optional(),
  }),
  z.strictObject({
    type: z.literal("thinking"),
    thinking: text,
    signature: text.min(1),
  }),
  z.strictObject({ type: z.literal("redacted_thinking"), data: text.min(1) }),
]);
const effort = z.enum(["low", "medium", "high", "xhigh", "max"]);
const toolChange = z.strictObject({
  type: z.enum(["tool_addition", "tool_removal"]),
  tool: z.strictObject({ type: z.literal("tool_reference"), name }),
});
const systemMessage = z.strictObject({
  role: z.literal("system"),
  content: z.union([text, z.array(z.union([textBlock, toolChange])).max(4096)]),
  output_config: z.strictObject({ effort }).optional(),
  clear_at: z.literal("next_user_message").optional(),
});
const natural = z.number().int().nonnegative();
const requestSchema = z.strictObject({
  model: z.string().min(1).max(256),
  max_tokens: z.number().int().positive(),
  stream: z.boolean().optional(),
  cache_control: cache.nullable().optional(),
  messages: z
    .array(
      z.discriminatedUnion("role", [
        z.strictObject({
          role: z.literal("user"),
          content: z.union([
            text.min(1),
            z
              .array(z.union([textBlock, imageBlock, toolResult]))
              .min(1)
              .max(4096),
          ]),
        }),
        z.strictObject({
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
    .array(
      z.strictObject({
        name,
        description: text.optional(),
        input_schema: z
          .record(z.string(), z.json())
          .refine((value) => value.type === "object"),
        eager_input_streaming: z.boolean().optional(),
        strict: z.boolean().optional(),
        defer_loading: z.boolean().optional(),
        cache_control: cache.optional(),
      }),
    )
    .max(1000)
    .optional(),
  thinking: z
    .discriminatedUnion("type", [
      z.strictObject({
        type: z.literal("enabled"),
        budget_tokens: z.number().int().min(1024),
        display: z.enum(["summarized", "omitted"]).optional(),
      }),
      z.strictObject({
        type: z.literal("adaptive"),
        display: z.enum(["summarized", "omitted", "updates"]).optional(),
        block_binding: z
          .strictObject({
            prefix_mismatch_behavior: z.enum(["error", "drop_block"]),
          })
          .optional(),
      }),
      z.strictObject({ type: z.literal("disabled") }),
    ])
    .optional(),
  output_config: z
    .strictObject({
      effort: effort.optional(),
      task_budget: z
        .strictObject({
          type: z.literal("tokens"),
          total: z.number().int().positive(),
          remaining: natural.optional(),
        })
        .optional(),
      format: z
        .strictObject({
          type: z.literal("json_schema"),
          schema: z.record(z.string(), z.json()),
        })
        .optional(),
    })
    .optional(),
  tool_choice: z
    .discriminatedUnion("type", [
      z.strictObject({
        type: z.literal("auto"),
        disable_parallel_tool_use: z.boolean().optional(),
      }),
      z.strictObject({
        type: z.literal("any"),
        disable_parallel_tool_use: z.boolean().optional(),
      }),
      z.strictObject({
        type: z.literal("tool"),
        name,
        disable_parallel_tool_use: z.boolean().optional(),
      }),
      z.strictObject({ type: z.literal("none") }),
    ])
    .optional(),
  temperature: z.number().min(0).max(1).optional(),
  top_p: z.number().min(0).max(1).optional(),
  top_k: natural.optional(),
  stop_sequences: z.array(text.min(1)).max(64).optional(),
  metadata: z
    .strictObject({ user_id: z.string().max(256).optional() })
    .optional(),
});
type NativeRequest = z.infer<typeof requestSchema>;
type InputBlock = z.infer<typeof textBlock> | z.infer<typeof imageBlock>;
const inputBlock = (block: InputBlock): TextContent | ImageContent =>
  block.type === "text"
    ? { type: "text", text: block.text }
    : {
        type: "image",
        data: block.source.data,
        mimeType: block.source.media_type,
      };

function toContext(input: NativeRequest, model: Model<Api>): Context {
  const declaredTools = new Map(input.tools?.map((tool) => [tool.name, tool]));
  const lateNames = new Set(
    input.messages.flatMap((message) =>
      message.role === "system" && Array.isArray(message.content)
        ? message.content.flatMap((block) =>
            block.type === "tool_addition" ? [block.tool.name] : [],
          )
        : [],
    ),
  );
  const calls = input.messages.flatMap((message) =>
    message.role === "assistant" && Array.isArray(message.content)
      ? message.content.filter((block) => block.type === "tool_use")
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
                  .filter((block) => block.type === "text")
                  .map((block) => block.text)
                  .join("\n\n"),
          toolsAdded: blocks.flatMap((block) => {
            if (block.type !== "tool_addition") return [];
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
          toolsRemoved: blocks.flatMap((block) =>
            block.type === "tool_removal" ? [{ name: block.tool.name }] : [],
          ),
          timestamp: 0,
        },
      ];
    }
    if (message.role === "assistant") {
      const content: AssistantMessage["content"] =
        typeof message.content === "string"
          ? [{ type: "text", text: message.content }]
          : message.content.map((block) => {
              if (block.type === "text")
                return { type: "text", text: block.text };
              if (block.type === "tool_use")
                return {
                  type: "toolCall",
                  id: block.id,
                  name: block.name,
                  arguments: block.input,
                };
              if (block.type === "thinking")
                return {
                  type: "thinking",
                  thinking: block.thinking,
                  thinkingSignature: block.signature,
                };
              return {
                type: "thinking",
                thinking: "",
                thinkingSignature: block.data,
                redacted: true,
              };
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
        block.type !== "tool_result"
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
    tools: input.tools
      ?.filter(
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
      const converted = convertedByName.get(tool.name);
      return {
        ...tool,
        name: converted?.name ?? tool.name,
        ...(converted?.defer_loading ? { defer_loading: true } : {}),
      };
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
              block.type === "tool_use"
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
      if (block.type === "text") continue;
      if (
        block.type === "tool_addition" &&
        (changedTools.has(block.tool.name) ||
          !input.tools?.some((tool) => tool.name === block.tool.name))
      )
        return invalid("Unsupported tool addition sequence");
      changedTools.add(block.tool.name);
    }
  }
  if (input.stop_sequences !== undefined)
    return invalid(
      "stop_sequences is unsupported: Pi does not preserve the matched stop sequence",
    );
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
  if (
    new Set(input.tools?.map((tool) => tool.name.toLowerCase())).size !==
    (input.tools?.length ?? 0)
  )
    return invalid("Tool names must be unique ignoring case");
  const context = toContext(input, model);
  if (
    !model.input.includes("image") &&
    context.messages.some(
      (message) =>
        message.role !== "assistant" &&
        Array.isArray(message.content) &&
        message.content.some((block) => block.type === "image"),
    )
  )
    return invalid("Model does not support images");
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
      includeUsage: true,
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
