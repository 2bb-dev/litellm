/**
 * Type definitions for the Tools section
 * Supports both OpenAI and Anthropic tool formats
 */

// OpenAI format: { type: "function", function: { name, description, parameters } }
// Anthropic format: { name, description, input_schema }
export interface ToolDefinition {
  // OpenAI format
  type?: string;
  function?: {
    name: string;
    description?: string;
    parameters?: Record<string, any>;
  };
  // Anthropic format (top-level)
  name?: string;
  description?: string;
  input_schema?: Record<string, any>;
}

// OpenAI response format: choices[].message.tool_calls[]
export interface ToolCall {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string;
  };
}

// Anthropic response format: content[].type === "tool_use"
export interface AnthropicToolUse {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, any>;
}

export interface ParsedTool {
  index: number;
  name: string;
  description: string;
  parameters: Record<string, any>;
  called: boolean;
  originalJson: Record<string, any>;
  callData?: {
    id: string;
    name: string;
    arguments: Record<string, any>;
  };
}

export interface ParameterRow {
  key: string;
  name: string;
  type: string;
  description: string;
  required: boolean;
}
