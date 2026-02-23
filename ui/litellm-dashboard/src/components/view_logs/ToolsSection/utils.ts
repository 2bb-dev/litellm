/**
 * Utility functions for parsing and processing tool data from log entries
 * Supports both OpenAI and Anthropic tool formats
 */

import { LogEntry } from "../columns";
import { ParsedTool, ToolDefinition, ToolCall, AnthropicToolUse } from "./types";

/**
 * Parse raw data that might be a string or object
 */
function parseData(input: any): any {
  if (typeof input === "string") {
    try {
      return JSON.parse(input);
    } catch {
      return input;
    }
  }
  return input;
}

/**
 * Extract tools array from request data
 */
function extractToolsFromRequest(log: LogEntry): ToolDefinition[] {
  // Check proxy_server_request first (most complete), then messages
  const requestData = parseData(log.proxy_server_request || log.messages);
  
  if (!requestData) return [];
  
  // Handle array format (messages array)
  if (Array.isArray(requestData)) {
    // Tools are not typically in messages array, return empty
    return [];
  }
  
  // Handle object format (request body)
  if (typeof requestData === "object" && requestData.tools) {
    return Array.isArray(requestData.tools) ? requestData.tools : [];
  }
  
  return [];
}

/**
 * Extract tool name from a tool definition (supports both formats)
 *
 * OpenAI:    { type: "function", function: { name: "foo" } }
 * Anthropic: { name: "foo", input_schema: {...} }
 */
function getToolName(tool: ToolDefinition, index: number): string {
  // Anthropic format: name at top level
  if (tool.name) return tool.name;
  // OpenAI format: nested under function
  if (tool.function?.name) return tool.function.name;
  // Fallback
  return `Tool ${index + 1}`;
}

/**
 * Extract tool description from a tool definition (supports both formats)
 */
function getToolDescription(tool: ToolDefinition): string {
  return tool.description || tool.function?.description || "";
}

/**
 * Extract tool parameters from a tool definition (supports both formats)
 *
 * OpenAI:    function.parameters
 * Anthropic: input_schema
 */
function getToolParameters(tool: ToolDefinition): Record<string, any> {
  return tool.input_schema || tool.function?.parameters || {};
}

/**
 * Extract tool calls from response data
 * Supports both OpenAI and Anthropic response formats
 */
function extractToolCallsFromResponse(log: LogEntry): { name: string; id: string; arguments: Record<string, any> }[] {
  const responseData = parseData(log.response);
  
  if (!responseData || typeof responseData !== "object") return [];
  
  const results: { name: string; id: string; arguments: Record<string, any> }[] = [];

  // OpenAI format: response.choices[0].message.tool_calls
  const choices = responseData.choices;
  if (Array.isArray(choices) && choices.length > 0) {
    const firstChoice = choices[0];
    const message = firstChoice.message;
    if (message && Array.isArray(message.tool_calls)) {
      for (const tc of message.tool_calls as ToolCall[]) {
        const name = tc.function?.name;
        if (name) {
          results.push({
            name,
            id: tc.id,
            arguments: parseSafeJson(tc.function?.arguments || "{}"),
          });
        }
      }
    }
  }

  // Anthropic format: response.content[].type === "tool_use"
  const content = responseData.content;
  if (Array.isArray(content)) {
    for (const block of content) {
      if (block.type === "tool_use" && block.name) {
        const toolUse = block as AnthropicToolUse;
        results.push({
          name: toolUse.name,
          id: toolUse.id,
          arguments: toolUse.input || {},
        });
      }
    }
  }

  return results;
}

/**
 * Parse safe JSON with fallback
 */
function parseSafeJson(jsonString: string): Record<string, any> {
  try {
    return JSON.parse(jsonString);
  } catch {
    return {};
  }
}

/**
 * Main function to parse tools from a log entry
 * Returns an array of tools with their definition and call status
 */
export function parseToolsFromLog(log: LogEntry): ParsedTool[] {
  // Get tools from request
  const requestTools = extractToolsFromRequest(log);
  
  if (requestTools.length === 0) {
    return [];
  }
  
  // Get tool calls from response
  const toolCallResults = extractToolCallsFromResponse(log);
  const calledToolNames = new Set(toolCallResults.map((tc) => tc.name));
  
  // Map tool calls by name for quick lookup
  const toolCallMap = new Map<string, { id: string; name: string; arguments: Record<string, any> }>();
  toolCallResults.forEach((tc) => {
    toolCallMap.set(tc.name, tc);
  });
  
  // Parse each tool definition (supports both OpenAI and Anthropic formats)
  return requestTools.map((tool: ToolDefinition, index: number) => {
    const name = getToolName(tool, index);
    
    return {
      index: index + 1,
      name: name,
      description: getToolDescription(tool),
      parameters: getToolParameters(tool),
      called: calledToolNames.has(name),
      callData: toolCallMap.get(name),
      originalJson: tool as Record<string, any>,
    };
  });
}

/**
 * Check if a log entry has any tools
 */
export function hasTools(log: LogEntry): boolean {
  const requestTools = extractToolsFromRequest(log);
  return requestTools.length > 0;
}
