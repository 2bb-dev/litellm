/**
 * JSON view of tool definition
 */

import { ParsedTool } from "./types";

interface JsonToolViewProps {
  tool: ParsedTool;
}

export function JsonToolView({ tool }: JsonToolViewProps) {
  // Show the original tool definition as stored in the request
  const toolJson = tool.originalJson;

  return (
    <pre
      style={{
        margin: 0,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
        fontSize: 12,
        background: "#fafafa",
        padding: 12,
        borderRadius: 4,
        maxHeight: 300,
        overflow: "auto",
      }}
    >
      {JSON.stringify(toolJson, null, 2)}
    </pre>
  );
}
