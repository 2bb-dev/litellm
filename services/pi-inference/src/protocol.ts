import type {
  AssistantMessage,
  AssistantMessageEvent,
  Context,
  SimpleStreamOptions,
  Usage,
} from "@earendil-works/pi-ai";

export interface ApiError {
  status: number;
  type: string;
  message: string;
}

export type Result<T> = { ok: true; value: T } | { ok: false; error: ApiError };

export const invalid = (message: string): Result<never> => ({
  ok: false,
  error: { status: 400, type: "invalid_request_error", message },
});

export interface PreparedCall {
  context: Context;
  options: SimpleStreamOptions;
  stream: boolean;
  includeUsage: boolean;
}

export interface WireEvent {
  event?: string;
  data: unknown;
  error?: ApiError;
}

export type EventEncoder = (event: AssistantMessageEvent) => WireEvent[];

export const emptyUsage = (): Usage => ({
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

export function failed(message: AssistantMessage): boolean {
  return message.stopReason === "error" || message.stopReason === "aborted";
}
