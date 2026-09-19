import { z } from "zod";
import {
  getSupportedThinkingLevels,
  Type,
  type Api,
  type AssistantMessage,
  type ImageContent,
  type Message,
  type Model,
  type TextContent,
} from "@earendil-works/pi-ai";
import {
  emptyUsage,
  invalid,
  type EventEncoder,
  type PreparedCall,
  type Result,
} from "./protocol.js";

const textPart = z
  .object({ type: z.literal("text"), text: z.string() })
  .strict();
const imagePart = z
  .object({
    type: z.literal("image_url"),
    image_url: z
      .object({
        url: z
          .string()
          .regex(
            /^data:image\/(png|jpeg|webp|gif);base64,[A-Za-z0-9+/]+={0,2}$/,
          ),
        detail: z.literal("auto").optional(),
      })
      .strict(),
  })
  .strict();
const text = z.union([z.string(), z.array(textPart)]);
const content = z.union([z.string(), z.array(z.union([textPart, imagePart]))]);
const toolCall = z
  .object({
    id: z.string().min(1),
    type: z.literal("function"),
    function: z
      .object({ name: z.string().min(1), arguments: z.string() })
      .strict(),
  })
  .strict();
const message = z.discriminatedUnion("role", [
  z.object({ role: z.literal("system"), content: text }).strict(),
  z.object({ role: z.literal("developer"), content: text }).strict(),
  z.object({ role: z.literal("user"), content }).strict(),
  z
    .object({
      role: z.literal("assistant"),
      content: text.nullable().optional(),
      tool_calls: z.array(toolCall).optional(),
      reasoning_content: z.string().optional(),
    })
    .strict(),
  z
    .object({
      role: z.literal("tool"),
      content,
      tool_call_id: z.string().min(1),
      name: z.string().optional(),
    })
    .strict(),
]);
const request = z
  .object({
    model: z.string().min(1),
    messages: z.array(message).min(1),
    stream: z.boolean().default(false),
    stream_options: z
      .object({ include_usage: z.boolean().optional() })
      .strict()
      .optional(),
    max_tokens: z.number().int().positive().optional(),
    max_completion_tokens: z.number().int().positive().optional(),
    temperature: z.number().min(0).max(2).optional(),
    reasoning_effort: z
      .enum(["none", "off", "minimal", "low", "medium", "high", "xhigh", "max"])
      .optional(),
    tool_choice: z.enum(["auto", "none"]).optional(),
    tools: z
      .array(
        z
          .object({
            type: z.literal("function"),
            function: z
              .object({
                name: z.string().min(1),
                description: z.string().optional(),
                parameters: z
                  .object({ type: z.literal("object") })
                  .catchall(z.unknown()),
                strict: z.boolean().optional(),
              })
              .strict(),
          })
          .strict(),
      )
      .optional(),
    n: z.literal(1).optional(),
    user: z.string().optional(),
    metadata: z.record(z.string(), z.unknown()).optional(),
    prompt_cache_key: z.string().min(1).max(256).optional(),
  })
  .strict();

function parts(value: z.infer<typeof content>): (TextContent | ImageContent)[] {
  if (typeof value === "string") return [{ type: "text", text: value }];
  return value.map((part) => {
    if (part.type === "text") return part;
    const separator = part.image_url.url.indexOf(",");
    return {
      type: "image",
      mimeType: part.image_url.url.slice(5, part.image_url.url.indexOf(";")),
      data: part.image_url.url.slice(separator + 1),
    };
  });
}

export function prepareChat(
  body: unknown,
  model: Model<Api>,
): Result<PreparedCall> {
  const parsed = request.safeParse(body);
  if (!parsed.success)
    return invalid("Unsupported or invalid Chat Completions request fields");
  const input = parsed.data;
  const effort =
    input.reasoning_effort === "none" ? "off" : input.reasoning_effort;
  if (effort && !getSupportedThinkingLevels(model).includes(effort))
    return invalid("Unsupported reasoning effort for this model");
  if (input.max_tokens && input.max_completion_tokens)
    return invalid("Use only one output token limit");
  const maxTokens = input.max_completion_tokens ?? input.max_tokens;
  if (maxTokens && maxTokens > model.maxTokens)
    return invalid("Output token limit exceeds model maximum");
  const firstDialogue = input.messages.findIndex(
    (item) => item.role !== "system" && item.role !== "developer",
  );
  if (
    firstDialogue < 0 ||
    input.messages
      .slice(firstDialogue)
      .some((item) => item.role === "system" || item.role === "developer")
  )
    return invalid(
      "System and developer messages must precede conversation messages",
    );
  const history: Message[] = [];
  const calls = new Map<string, string>();
  for (const item of input.messages.slice(firstDialogue)) {
    if (item.role === "user") {
      history.push({
        role: "user",
        content: parts(item.content),
        timestamp: 0,
      });
    } else if (item.role === "assistant") {
      const blocks: AssistantMessage["content"] =
        item.content == null
          ? []
          : typeof item.content === "string"
            ? [{ type: "text", text: item.content }]
            : item.content;
      if (item.reasoning_content)
        blocks.push({ type: "thinking", thinking: item.reasoning_content });
      for (const call of item.tool_calls ?? []) {
        const args = parseArguments(call.function.arguments);
        if (!args.ok) return args;
        if (calls.has(call.id)) return invalid("Duplicate tool call id");
        calls.set(call.id, call.function.name);
        blocks.push({
          type: "toolCall",
          id: call.id,
          name: call.function.name,
          arguments: args.value,
        });
      }
      history.push({
        role: "assistant",
        content: blocks,
        model: model.id,
        provider: model.provider,
        api: model.api,
        timestamp: 0,
        usage: emptyUsage(),
        stopReason: item.tool_calls?.length ? "toolUse" : "stop",
      });
    } else if (item.role === "tool") {
      const name = calls.get(item.tool_call_id);
      if (!name || (item.name && item.name !== name))
        return invalid("Tool result must match a preceding tool call");
      history.push({
        role: "toolResult",
        toolCallId: item.tool_call_id,
        toolName: name,
        content: parts(item.content),
        isError: false,
        timestamp: 0,
      });
      calls.delete(item.tool_call_id);
    }
  }
  if (calls.size)
    return invalid("Every historical tool call must have a tool result");
  if (
    !model.input.includes("image") &&
    history.some(
      (item) =>
        typeof item.content !== "string" &&
        item.content.some((part) => part.type === "image"),
    )
  )
    return invalid("Model does not support images");
  return {
    ok: true,
    value: {
      context: {
        systemPrompt: input.messages
          .slice(0, firstDialogue)
          .map((item) =>
            typeof item.content === "string"
              ? item.content
              : item.content
                  ?.map((part) => (part.type === "text" ? part.text : ""))
                  .join("\n"),
          )
          .join("\n\n"),
        messages: history,
        tools:
          input.tool_choice === "none"
            ? undefined
            : input.tools?.map(({ function: tool }) => ({
                name: tool.name,
                description: tool.description ?? "",
                parameters: Type.Unsafe(tool.parameters),
                ...(tool.strict
                  ? {
                      constrainedSampling: {
                        type: "json_schema" as const,
                        strict: "require" as const,
                      },
                    }
                  : {}),
              })),
      },
      options: {
        maxTokens,
        temperature: input.temperature,
        reasoning: effort === "off" ? undefined : effort,
        sessionId: input.prompt_cache_key,
        metadata: input.user ? { user_id: input.user } : undefined,
      },
      stream: input.stream,
      includeUsage: input.stream_options?.include_usage ?? false,
    },
  };
}

function parseArguments(value: string): Result<Record<string, unknown>> {
  try {
    const parsed = z
      .record(z.string(), z.unknown())
      .safeParse(JSON.parse(value));
    return parsed.success
      ? { ok: true, value: parsed.data }
      : invalid("Tool arguments must be a JSON object");
  } catch {
    return invalid("Tool arguments must be valid JSON");
  }
}

export function chatUsage(message: AssistantMessage) {
  const { input, output, cacheRead, cacheWrite, reasoning } = message.usage;
  const prompt = input + cacheRead + cacheWrite;
  return {
    prompt_tokens: prompt,
    completion_tokens: output,
    total_tokens: prompt + output,
    prompt_tokens_details: {
      cached_tokens: cacheRead,
      cache_creation_tokens: cacheWrite,
    },
    completion_tokens_details: {
      ...(reasoning === undefined ? {} : { reasoning_tokens: reasoning }),
    },
  };
}

function finishReason(message: AssistantMessage): string {
  if (message.rawStopReason === "refusal") return "content_filter";
  return message.stopReason === "toolUse"
    ? "tool_calls"
    : message.stopReason === "length"
      ? "length"
      : "stop";
}

export function chatResponse(
  message: AssistantMessage,
  alias: string,
  id: string,
) {
  const toolCalls = message.content
    .filter((block) => block.type === "toolCall")
    .map((block) => ({
      id: block.id,
      type: "function",
      function: {
        name: block.name,
        arguments: JSON.stringify(block.arguments),
      },
    }));
  const reasoning = message.content
    .filter((block) => block.type === "thinking" && !block.redacted)
    .map((block) => (block.type === "thinking" ? block.thinking : ""))
    .join("");
  return {
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: alias,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content:
            message.content
              .filter((block) => block.type === "text")
              .map((block) => block.text)
              .join("") || null,
          ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
          ...(reasoning ? { reasoning_content: reasoning } : {}),
        },
        finish_reason: finishReason(message),
      },
    ],
    usage: chatUsage(message),
  };
}

export function createChatEncoder(
  alias: string,
  id: string,
  includeUsage: boolean,
): EventEncoder {
  const indexes = new Map<number, number>();
  const envelope = {
    id,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: alias,
  };
  const chunk = (delta: unknown, finish: string | null = null) => ({
    data: {
      ...envelope,
      choices: [{ index: 0, delta, finish_reason: finish }],
    },
  });
  return (event) => {
    switch (event.type) {
      case "start":
        return [chunk({ role: "assistant", content: "" })];
      case "text_delta":
        return [chunk({ content: event.delta })];
      case "thinking_delta":
        return [chunk({ reasoning_content: event.delta })];
      case "toolcall_end": {
        const index = indexes.size;
        indexes.set(event.contentIndex, index);
        return [
          chunk({
            tool_calls: [
              {
                index,
                id: event.toolCall.id,
                type: "function",
                function: {
                  name: event.toolCall.name,
                  arguments: JSON.stringify(event.toolCall.arguments),
                },
              },
            ],
          }),
        ];
      }
      case "done":
        return [
          chunk({}, finishReason(event.message)),
          ...(includeUsage
            ? [
                {
                  data: {
                    ...envelope,
                    choices: [],
                    usage: chatUsage(event.message),
                  },
                },
              ]
            : []),
          { data: "[DONE]" },
        ];
      default:
        return [];
    }
  };
}
