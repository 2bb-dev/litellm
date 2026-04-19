/**
 * Utility functions for parsing and formatting messages for pretty view
 */

import { ParsedMessage, ParsedMessages, RoleStyle, ToolCall } from './prettyMessagesTypes';

/**
 * Role color styles for message cards - minimal, professional design
 * Color only used for labels and left border accent
 */
export const ROLE_STYLES: Record<string, RoleStyle> = {
  system: {
    background: 'transparent',
    borderColor: '#8c8c8c',
    label: 'SYSTEM',
    labelColor: '#8c8c8c',
  },
  developer: {
    background: 'transparent',
    borderColor: '#722ed1',
    label: 'DEVELOPER',
    labelColor: '#722ed1',
  },
  user: {
    background: 'transparent',
    borderColor: '#1677ff',
    label: 'USER',
    labelColor: '#1677ff',
  },
  assistant: {
    background: 'transparent',
    borderColor: '#52c41a',
    label: 'ASSISTANT',
    labelColor: '#52c41a',
  },
  tool: {
    background: 'transparent',
    borderColor: '#fa8c16',
    label: 'TOOL RESULT',
    labelColor: '#fa8c16',
  },
};

type ParsedRole = ParsedMessage['role'];

const KNOWN_ROLES: readonly ParsedRole[] = ['system', 'developer', 'user', 'assistant', 'tool'];

const normalizeRole = (role: any): ParsedRole => {
  if (typeof role === 'string' && (KNOWN_ROLES as readonly string[]).includes(role)) {
    return role as ParsedRole;
  }
  return 'user';
};

/**
 * Parse request messages and response message from log data.
 *
 * Supports:
 *   - Chat Completions: request.messages[] + response.choices[0].message
 *   - Responses API:    request.input[] (or string) + response.output[]
 *   - Anthropic-style:  request.system (string or content-part array)
 */
export const parseMessages = (request: any, response: any): ParsedMessages => {
  const requestMessages: ParsedMessage[] = [];

  // Anthropic Messages API stores system prompt in a separate top-level field.
  if (request?.system) {
    requestMessages.push({
      role: 'system',
      content: parseMessageContent(request.system),
    });
  }

  if (request?.messages && Array.isArray(request.messages)) {
    request.messages.forEach((msg: any) => {
      requestMessages.push({
        role: normalizeRole(msg.role),
        content: parseMessageContent(msg.content),
        toolCallId: msg.tool_call_id,
      });
    });
  } else if (request?.input !== undefined) {
    // OpenAI Responses API: `input` can be a string or an array of message-like items.
    parseResponsesInput(request.input).forEach((m) => requestMessages.push(m));
  }

  const responseMessage = parseResponseMessage(response);

  return { requestMessages, responseMessage };
};

/**
 * Parse message content - handle strings and content arrays (for vision, Responses API, Anthropic, etc.)
 */
const parseMessageContent = (content: any): string => {
  if (content === null || content === undefined) {
    return '';
  }

  if (typeof content === 'string') {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === 'string') return item;
        if (!item || typeof item !== 'object') return '';
        // OpenAI chat: { type: "text", text } / { type: "image_url" }
        // OpenAI Responses API: { type: "input_text", text } / { type: "output_text", text }
        // Anthropic: { type: "text", text } / { type: "image" }
        if (item.type === 'text' || item.type === 'input_text' || item.type === 'output_text') {
          return typeof item.text === 'string' ? item.text : '';
        }
        if (item.type === 'image_url' || item.type === 'image' || item.type === 'input_image') {
          return '[Image]';
        }
        // Fallback: some providers omit `type` but still carry `text`.
        if (typeof item.text === 'string') return item.text;
        return JSON.stringify(item);
      })
      .filter((s) => s.length > 0)
      .join('\n');
  }

  return JSON.stringify(content);
};

/**
 * Parse the `input` field of a Responses API request into ParsedMessages.
 */
const parseResponsesInput = (input: any): ParsedMessage[] => {
  if (typeof input === 'string') {
    return [{ role: 'user', content: input }];
  }
  if (!Array.isArray(input)) {
    return [];
  }
  const out: ParsedMessage[] = [];
  input.forEach((item) => {
    if (!item || typeof item !== 'object') return;
    // A Responses API input item can be a message, a function_call, or a function_call_output.
    const type = item.type;
    if (type === 'function_call_output') {
      out.push({
        role: 'tool',
        content: parseMessageContent(item.output),
        toolCallId: item.call_id,
      });
      return;
    }
    if (type === 'function_call') {
      out.push({
        role: 'assistant',
        content: '',
        toolCalls: [
          {
            id: item.call_id || item.id || '',
            name: item.name || 'unknown',
            arguments: parseToolArguments(item.arguments),
          },
        ],
      });
      return;
    }
    // Default: treat as a message (type may be "message" or omitted).
    out.push({
      role: normalizeRole(item.role),
      content: parseMessageContent(item.content),
    });
  });
  return out;
};

/**
 * Extract the assistant message from a response, supporting both
 * Chat Completions and Responses API shapes.
 */
const parseResponseMessage = (response: any): ParsedMessage | null => {
  if (!response || typeof response !== 'object') return null;

  // Chat Completions: { choices: [{ message: { role, content, tool_calls } }] }
  const chatMsg = response?.choices?.[0]?.message;
  if (chatMsg) {
    return {
      role: normalizeRole(chatMsg.role || 'assistant'),
      content: chatMsg.content ?? '',
      toolCalls: parseToolCalls(chatMsg.tool_calls),
    };
  }

  // Responses API: { output: [ { type: "message", content: [{ type: "output_text", text }] }, { type: "function_call", ... } ] }
  if (Array.isArray(response.output) && response.output.length > 0) {
    const textParts: string[] = [];
    const toolCalls: ToolCall[] = [];
    response.output.forEach((item: any) => {
      if (!item || typeof item !== 'object') return;
      if (item.type === 'function_call') {
        toolCalls.push({
          id: item.call_id || item.id || '',
          name: item.name || 'unknown',
          arguments: parseToolArguments(item.arguments),
        });
        return;
      }
      // Message or reasoning item with content parts.
      const rendered = parseMessageContent(item.content);
      if (rendered) textParts.push(rendered);
    });

    const content = textParts.join('\n') || (typeof response.output_text === 'string' ? response.output_text : '');
    if (content || toolCalls.length > 0) {
      return {
        role: 'assistant',
        content,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      };
    }
  }

  // Responses API fallback: convenience field `output_text` only.
  if (typeof response.output_text === 'string' && response.output_text.length > 0) {
    return { role: 'assistant', content: response.output_text };
  }

  return null;
};

/**
 * Parse tool calls from a Chat Completions response message.
 */
const parseToolCalls = (toolCalls: any[]): ToolCall[] | undefined => {
  if (!toolCalls || !Array.isArray(toolCalls)) return undefined;

  return toolCalls.map((tc) => ({
    id: tc.id || '',
    name: tc.function?.name || 'unknown',
    arguments: parseToolArguments(tc.function?.arguments),
  }));
};

/**
 * Parse tool arguments - handle both string and object formats
 */
const parseToolArguments = (args: any): Record<string, any> => {
  if (!args) return {};

  if (typeof args === 'string') {
    try {
      return JSON.parse(args);
    } catch {
      return { raw: args };
    }
  }

  return args;
};
