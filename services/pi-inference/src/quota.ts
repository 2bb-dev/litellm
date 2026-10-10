import type { ApiError } from "./protocol.js";

export const exhaustedUntilHeader = "x-openorange-subscription-exhausted-until";
export const atCapacityHeader = "x-openorange-pi-slot-at-capacity";

export interface UnifiedLimit {
  status?: string;
  reset?: string;
  claim?: string;
}

export interface Exhaustion {
  until: number;
  claim?: string;
}

const unknownResetSeconds = 300;
const longestExhaustionSeconds = 8 * 24 * 3600;

export function unifiedLimit(headers: Headers): UnifiedLimit {
  const read = (name: string) =>
    headers.get(`anthropic-ratelimit-unified-${name}`)?.trim() || undefined;
  return {
    status: read("status"),
    reset: read("reset"),
    claim: read("representative-claim"),
  };
}

export function exhaustedUntil(
  status: number | undefined,
  limit: UnifiedLimit | undefined,
  nowSeconds: number,
): number | undefined {
  if (status !== 429 || limit?.status?.toLowerCase() !== "rejected")
    return undefined;
  const reset = Number(limit.reset ?? Number.NaN);
  if (!Number.isFinite(reset)) return nowSeconds + unknownResetSeconds;
  const until = Math.ceil(reset);
  if (until <= nowSeconds) return undefined;
  return Math.min(until, nowSeconds + longestExhaustionSeconds);
}

export class ExhaustedModels {
  readonly #byAlias = new Map<string, Exhaustion>();

  constructor(private readonly nowSeconds: () => number) {}

  get(alias: string): Exhaustion | undefined {
    const exhaustion = this.#byAlias.get(alias);
    if (exhaustion && exhaustion.until <= this.nowSeconds()) {
      this.#byAlias.delete(alias);
      return undefined;
    }
    return exhaustion;
  }

  mark(alias: string, exhaustion: Exhaustion): Exhaustion {
    const known = this.get(alias);
    const kept = known && known.until >= exhaustion.until ? known : exhaustion;
    this.#byAlias.set(alias, kept);
    return kept;
  }
}

export function exhaustionClaim(value: string | undefined): string | undefined {
  return value && /^[a-z0-9_]{1,64}$/.test(value) ? value : undefined;
}

export function exhaustionFailure(exhaustion: Exhaustion): ApiError {
  const claim = exhaustion.claim ? ` (${exhaustion.claim})` : "";
  const resets = new Date(exhaustion.until * 1000).toISOString();
  return {
    status: 429,
    type: "rate_limit_error",
    code: "subscription_exhausted",
    message: `Claude subscription usage limit reached${claim}; it resets at ${resets}`,
    exhaustedUntil: exhaustion.until,
  };
}
