// Adapted in part from @benvargas/pi-claude-code-use@2.2.0
// (https://github.com/ben-vargas/pi-packages, revision 4eaa1e26e44151a01c6977354e7c539322f048be)
// MIT License, Copyright (c) 2026 Ben Vargas

import {
  createAssistantMessageEventStream,
  type Api,
  type AssistantMessage,
  type AssistantMessageEvent,
  type AssistantMessageEventStream,
  type Model,
  type Provider,
  type StreamOptions,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import { emptyUsage } from "./protocol.js";

const coreToolNames = new Map([
  ["askuserquestion", "AskUserQuestion"],
  ["enterplanmode", "EnterPlanMode"],
  ["exitplanmode", "ExitPlanMode"],
  ["killshell", "KillShell"],
  ["notebookedit", "NotebookEdit"],
  ["taskoutput", "TaskOutput"],
  ["todowrite", "TodoWrite"],
  ["webfetch", "WebFetch"],
  ["websearch", "WebSearch"],
]);

const coreTools = new Set([
  "read",
  "write",
  "edit",
  "bash",
  "grep",
  "glob",
  "askuserquestion",
  "enterplanmode",
  "exitplanmode",
  "killshell",
  "notebookedit",
  "skill",
  "task",
  "taskoutput",
  "todowrite",
  "webfetch",
  "websearch",
]);

function toolAliases(context: TranscriptContext): ReadonlyMap<string, string> {
  const names = new Set(
    context.messages.flatMap((message) => {
      if (message.role === "system")
        return [
          ...(message.toolsAdded?.map((tool) => tool.name) ?? []),
          ...(message.toolsRemoved?.map((tool) => tool.name) ?? []),
        ];
      if (message.role === "assistant")
        return message.content.flatMap((block) =>
          block.type === "toolCall" ? [block.name] : [],
        );
      if (message.role === "toolResult") return [message.toolName];
      return [];
    }),
  );
  const reserved = new Set([...names].map((name) => name.toLowerCase()));
  const aliases = new Map<string, string>();
  const suffixes = new Map<string, number>();
  for (const name of names) {
    const lower = name.toLowerCase();
    if (!name || coreTools.has(lower) || lower.startsWith("mcp__")) continue;
    const segment =
      lower.replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "tool";
    const stem = `mcp__pi__${segment}`.slice(0, 128);
    let alias = stem;
    let suffix = suffixes.get(stem) ?? 1;
    while (reserved.has(alias)) {
      const tail = `_${++suffix}`;
      alias = stem.slice(0, 128 - tail.length) + tail;
    }
    suffixes.set(stem, suffix);
    reserved.add(alias);
    aliases.set(name, alias);
  }
  return aliases;
}

function renameMessage(
  message: AssistantMessage,
  aliases: ReadonlyMap<string, string>,
): AssistantMessage {
  return {
    ...message,
    content: message.content.map((block) =>
      block.type === "toolCall"
        ? { ...block, name: aliases.get(block.name) ?? block.name }
        : block,
    ),
  };
}

function renameContext(
  context: TranscriptContext,
  aliases: ReadonlyMap<string, string>,
): TranscriptContext {
  const name = (value: string) => aliases.get(value) ?? value;
  return {
    ...context,
    messages: context.messages.map((message) => {
      if (message.role === "system")
        return {
          ...message,
          ...(message.toolsAdded
            ? {
                toolsAdded: message.toolsAdded.map((tool) => ({
                  ...tool,
                  name: name(tool.name),
                })),
              }
            : {}),
          ...(message.toolsRemoved
            ? {
                toolsRemoved: message.toolsRemoved.map((tool) => ({
                  ...tool,
                  name: name(tool.name),
                })),
              }
            : {}),
        };
      if (message.role === "assistant") return renameMessage(message, aliases);
      if (message.role === "toolResult")
        return { ...message, toolName: name(message.toolName) };
      return message;
    }),
  };
}

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function renameClaudeOAuthResponse(
  value: unknown,
  context: TranscriptContext,
): unknown {
  if (!object(value)) return value;
  const aliases = toolAliases(context);
  const reverse = new Map([...aliases].map(([name, alias]) => [alias, name]));
  for (const message of context.messages) {
    if (message.role !== "system") continue;
    for (const tool of message.toolsAdded ?? []) {
      const lower = tool.name.toLowerCase();
      if (coreTools.has(lower))
        reverse.set(
          coreToolNames.get(lower) ?? lower[0]!.toUpperCase() + lower.slice(1),
          tool.name,
        );
    }
  }
  const block =
    value.type === "content_block_start" ? value.content_block : undefined;
  if (
    object(block) &&
    block.type === "tool_use" &&
    typeof block.name === "string"
  )
    return {
      ...value,
      content_block: { ...block, name: reverse.get(block.name) ?? block.name },
    };
  if (!Array.isArray(value.content)) return value;
  return {
    ...value,
    content: value.content.map((item: unknown) =>
      object(item) && item.type === "tool_use" && typeof item.name === "string"
        ? { ...item, name: reverse.get(item.name) ?? item.name }
        : item,
    ),
  };
}

function rewriteSystem(payload: unknown): unknown {
  if (!object(payload) || payload.system === undefined) return payload;
  const text = (value: string) =>
    value
      .replaceAll("pi itself", "the cli itself")
      .replaceAll("pi .md files", "cli .md files")
      .replaceAll("pi packages", "cli packages");
  const system = payload.system;
  return {
    ...payload,
    system:
      typeof system === "string"
        ? text(system)
        : Array.isArray(system)
          ? system.map((block: unknown) =>
              object(block) &&
              block.type === "text" &&
              typeof block.text === "string"
                ? { ...block, text: text(block.text) }
                : block,
            )
          : system,
  };
}

function renameToolReferences(
  payload: unknown,
  aliases: ReadonlyMap<string, string>,
): unknown {
  if (!object(payload) || !Array.isArray(payload.messages)) return payload;
  return {
    ...payload,
    messages: payload.messages.map((message: unknown) => {
      if (
        !object(message) ||
        message.role !== "system" ||
        !Array.isArray(message.content)
      )
        return message;
      return {
        ...message,
        content: message.content.map((block: unknown) => {
          if (
            !object(block) ||
            (block.type !== "tool_addition" && block.type !== "tool_removal") ||
            !object(block.tool) ||
            block.tool.type !== "tool_reference" ||
            typeof block.tool.name !== "string"
          )
            return block;
          return {
            ...block,
            tool: {
              ...block.tool,
              name: aliases.get(block.tool.name) ?? block.tool.name,
            },
          };
        }),
      };
    }),
  };
}

function renameEvent(
  event: AssistantMessageEvent,
  aliases: ReadonlyMap<string, string>,
): AssistantMessageEvent {
  if (event.type === "done")
    return { ...event, message: renameMessage(event.message, aliases) };
  if (event.type === "error")
    return { ...event, error: renameMessage(event.error, aliases) };
  if (event.type === "toolcall_end")
    return {
      ...event,
      partial: renameMessage(event.partial, aliases),
      toolCall: {
        ...event.toolCall,
        name: aliases.get(event.toolCall.name) ?? event.toolCall.name,
      },
    };
  return { ...event, partial: renameMessage(event.partial, aliases) };
}

function compatibleStream<T extends StreamOptions>(
  model: Model<Api>,
  context: TranscriptContext,
  options: T | undefined,
  invoke: (
    context: TranscriptContext,
    options?: T,
  ) => AssistantMessageEventStream,
): AssistantMessageEventStream {
  if (
    model.api !== "anthropic-messages" ||
    !options?.apiKey?.includes("sk-ant-oat") ||
    ("client" in options && options.client)
  )
    return invoke(context, options);
  const aliases = toolAliases(context);
  const reverse = new Map([...aliases].map(([name, alias]) => [alias, name]));
  const choice = "toolChoice" in options ? options.toolChoice : undefined;
  const source = invoke(renameContext(context, aliases), {
    ...options,
    ...(object(choice) &&
    choice.type === "tool" &&
    typeof choice.name === "string"
      ? {
          toolChoice: {
            ...choice,
            name: aliases.get(choice.name) ?? choice.name,
          },
        }
      : {}),
    onPayload: async (payload, resolved) => {
      const overlaid = await options.onPayload?.(payload, resolved);
      return renameToolReferences(
        rewriteSystem(overlaid === undefined ? payload : overlaid),
        aliases,
      );
    },
  });
  const stream = createAssistantMessageEventStream();
  void (async () => {
    try {
      for await (const event of source)
        stream.push(renameEvent(event, reverse));
      stream.end(renameMessage(await source.result(), reverse));
    } catch {
      const reason = options.signal?.aborted ? "aborted" : "error";
      stream.push({
        type: "error",
        reason,
        error: {
          role: "assistant",
          content: [],
          api: model.api,
          provider: model.provider,
          model: model.id,
          usage: emptyUsage(),
          timestamp: Date.now(),
          stopReason: reason,
          errorMessage: "Anthropic OAuth compatibility stream failed",
        },
      });
      stream.end();
    }
  })();
  return stream;
}

export function withClaudeOAuthCompatibility(provider: Provider): Provider {
  if (provider.id !== "anthropic") return provider;
  return {
    ...provider,
    stream: (model, context, options) =>
      compatibleStream(model, context, options, (nextContext, nextOptions) =>
        provider.stream(model, nextContext, nextOptions),
      ),
    streamSimple: (model, context, options) =>
      compatibleStream(model, context, options, (nextContext, nextOptions) =>
        provider.streamSimple(model, nextContext, nextOptions),
      ),
  };
}
