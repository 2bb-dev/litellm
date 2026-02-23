/**
 * Tests for tool parsing utilities
 * Covers both OpenAI and Anthropic tool formats
 */

import { describe, it, expect } from "vitest";
import { parseToolsFromLog, hasTools } from "./utils";
import { LogEntry } from "../columns";

describe("ToolsSection utils", () => {
  describe("parseToolsFromLog", () => {
    it("should return empty array when no tools in request", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-1",
        messages: [],
        response: {},
      };

      const result = parseToolsFromLog(log as LogEntry);

      expect(result).toEqual([]);
    });

    it("should parse tools from proxy_server_request (OpenAI format)", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-2",
        proxy_server_request: {
          tools: [
            {
              type: "function",
              function: {
                name: "get_weather",
                description: "Get the current weather",
                parameters: {
                  type: "object",
                  properties: {
                    location: { type: "string" },
                  },
                  required: ["location"],
                },
              },
            },
          ],
        },
        response: {},
      } as any;

      const result = parseToolsFromLog(log as LogEntry);

      expect(result).toHaveLength(1);
      expect(result[0]).toMatchObject({
        index: 1,
        name: "get_weather",
        description: "Get the current weather",
        called: false,
      });
    });

    it("should parse tools from proxy_server_request (Anthropic format)", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-anthropic-1",
        proxy_server_request: {
          tools: [
            {
              name: "read_file",
              description: "Read a file from disk",
              input_schema: {
                type: "object",
                properties: {
                  path: { type: "string", description: "File path" },
                },
                required: ["path"],
              },
            },
            {
              name: "write_file",
              description: "Write content to a file",
              input_schema: {
                type: "object",
                properties: {
                  path: { type: "string" },
                  content: { type: "string" },
                },
                required: ["path", "content"],
              },
            },
          ],
        },
        response: {},
      } as any;

      const result = parseToolsFromLog(log as LogEntry);

      expect(result).toHaveLength(2);
      expect(result[0].name).toBe("read_file");
      expect(result[0].description).toBe("Read a file from disk");
      expect(result[0].parameters).toEqual({
        type: "object",
        properties: {
          path: { type: "string", description: "File path" },
        },
        required: ["path"],
      });
      expect(result[1].name).toBe("write_file");
    });

    it("should parse tools from messages object format", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-3",
        messages: {
          tools: [
            {
              type: "function",
              function: {
                name: "search_web",
                description: "Search the web",
              },
            },
          ],
        },
        response: {},
      } as any;

      const result = parseToolsFromLog(log as LogEntry);

      expect(result).toHaveLength(1);
      expect(result[0].name).toBe("search_web");
    });

    it("should mark tools as called when present in OpenAI response", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-4",
        proxy_server_request: {
          tools: [
            {
              type: "function",
              function: {
                name: "get_weather",
                description: "Get weather",
              },
            },
            {
              type: "function",
              function: {
                name: "send_email",
                description: "Send email",
              },
            },
          ],
        },
        response: {
          choices: [
            {
              message: {
                tool_calls: [
                  {
                    id: "call_123",
                    type: "function",
                    function: {
                      name: "get_weather",
                      arguments: '{"location": "San Francisco"}',
                    },
                  },
                ],
              },
            },
          ],
        },
      } as any;

      const result = parseToolsFromLog(log as LogEntry);

      expect(result).toHaveLength(2);
      expect(result[0].called).toBe(true);
      expect(result[0].callData).toBeDefined();
      expect(result[0].callData?.arguments).toEqual({
        location: "San Francisco",
      });
      expect(result[1].called).toBe(false);
      expect(result[1].callData).toBeUndefined();
    });

    it("should mark tools as called when present in Anthropic response", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-anthropic-2",
        proxy_server_request: {
          tools: [
            {
              name: "read_file",
              description: "Read a file",
              input_schema: { type: "object", properties: {} },
            },
            {
              name: "write_file",
              description: "Write a file",
              input_schema: { type: "object", properties: {} },
            },
          ],
        },
        response: {
          content: [
            {
              type: "tool_use",
              id: "toolu_123",
              name: "read_file",
              input: { path: "/tmp/test.txt" },
            },
          ],
        },
      } as any;

      const result = parseToolsFromLog(log as LogEntry);

      expect(result).toHaveLength(2);
      expect(result[0].called).toBe(true);
      expect(result[0].callData).toBeDefined();
      expect(result[0].callData?.arguments).toEqual({ path: "/tmp/test.txt" });
      expect(result[1].called).toBe(false);
    });

    it("should handle string format request and response", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-5",
        proxy_server_request: JSON.stringify({
          tools: [
            {
              type: "function",
              function: {
                name: "calculate",
              },
            },
          ],
        }),
        response: JSON.stringify({
          choices: [
            {
              message: {
                tool_calls: [
                  {
                    id: "call_456",
                    type: "function",
                    function: {
                      name: "calculate",
                      arguments: '{"x": 5}',
                    },
                  },
                ],
              },
            },
          ],
        }),
      } as any;

      const result = parseToolsFromLog(log as LogEntry);

      expect(result).toHaveLength(1);
      expect(result[0].called).toBe(true);
    });

    it("should handle tools with no description or parameters", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-6",
        proxy_server_request: {
          tools: [
            {
              type: "function",
              function: {
                name: "minimal_tool",
              },
            },
          ],
        },
        response: {},
      } as any;

      const result = parseToolsFromLog(log as LogEntry);

      expect(result).toHaveLength(1);
      expect(result[0]).toMatchObject({
        index: 1,
        name: "minimal_tool",
        description: "",
        parameters: {},
        called: false,
      });
    });

    it("should handle invalid JSON in tool call arguments gracefully", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-7",
        proxy_server_request: {
          tools: [
            {
              type: "function",
              function: {
                name: "test_tool",
              },
            },
          ],
        },
        response: {
          choices: [
            {
              message: {
                tool_calls: [
                  {
                    id: "call_789",
                    type: "function",
                    function: {
                      name: "test_tool",
                      arguments: "invalid json",
                    },
                  },
                ],
              },
            },
          ],
        },
      } as any;

      const result = parseToolsFromLog(log as LogEntry);

      expect(result).toHaveLength(1);
      expect(result[0].called).toBe(true);
      expect(result[0].callData?.arguments).toEqual({});
    });

    it("should assign correct indices to multiple tools", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-8",
        proxy_server_request: {
          tools: [
            { type: "function", function: { name: "tool1" } },
            { type: "function", function: { name: "tool2" } },
            { type: "function", function: { name: "tool3" } },
          ],
        },
        response: {},
      } as any;

      const result = parseToolsFromLog(log as LogEntry);

      expect(result).toHaveLength(3);
      expect(result[0].index).toBe(1);
      expect(result[1].index).toBe(2);
      expect(result[2].index).toBe(3);
    });

    it("should preserve originalJson for each tool", () => {
      const anthropicTool = {
        name: "read_file",
        description: "Read a file",
        input_schema: { type: "object", properties: {} },
      };
      const log: Partial<LogEntry> = {
        request_id: "test-original-json",
        proxy_server_request: {
          tools: [anthropicTool],
        },
        response: {},
      } as any;

      const result = parseToolsFromLog(log as LogEntry);

      expect(result[0].originalJson).toEqual(anthropicTool);
    });
  });

  describe("hasTools", () => {
    it("should return false when no tools in request", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-9",
        messages: [],
        response: {},
      };

      expect(hasTools(log as LogEntry)).toBe(false);
    });

    it("should return true when tools present in request", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-10",
        proxy_server_request: {
          tools: [
            {
              type: "function",
              function: {
                name: "test_tool",
              },
            },
          ],
        },
        response: {},
      } as any;

      expect(hasTools(log as LogEntry)).toBe(true);
    });

    it("should return true when Anthropic format tools present", () => {
      const log: Partial<LogEntry> = {
        request_id: "test-11",
        proxy_server_request: {
          tools: [
            {
              name: "test_tool",
              input_schema: { type: "object", properties: {} },
            },
          ],
        },
        response: {},
      } as any;

      expect(hasTools(log as LogEntry)).toBe(true);
    });
  });
});
