import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ExhaustedModels,
  exhaustedUntil,
  exhaustionClaim,
  exhaustionFailure,
  unifiedLimit,
} from "../src/quota.js";

const now = 1_800_000_000;

test("only a 429 that Anthropic marks rejected is an exhausted subscription", () => {
  const rejected = { status: "rejected", reset: String(now + 3600) };
  assert.equal(exhaustedUntil(429, rejected, now), now + 3600);
  assert.equal(
    exhaustedUntil(429, { ...rejected, status: "Rejected" }, now),
    now + 3600,
  );
  assert.equal(
    exhaustedUntil(429, { ...rejected, reset: `${now + 10.2}` }, now),
    now + 11,
  );
  for (const status of [200, 400, 500, 529, undefined])
    assert.equal(exhaustedUntil(status, rejected, now), undefined);
  for (const limit of [
    undefined,
    {},
    { ...rejected, status: "allowed" },
    { ...rejected, status: "allowed_warning" },
  ])
    assert.equal(exhaustedUntil(429, limit, now), undefined);
});

test("a rejection's reset is bounded; one already past is a reopened window", () => {
  const rejected = (reset?: string) => ({ status: "rejected", reset });
  assert.equal(exhaustedUntil(429, rejected(), now), now + 300);
  assert.equal(exhaustedUntil(429, rejected("soon"), now), now + 300);
  assert.equal(exhaustedUntil(429, rejected(String(now)), now), undefined);
  assert.equal(exhaustedUntil(429, rejected(String(now - 5)), now), undefined);
  assert.equal(
    exhaustedUntil(429, rejected(String(now + 30 * 86_400)), now),
    now + 8 * 86_400,
  );
});

test("unified headers are read case-insensitively and trimmed", () => {
  const headers = new Headers({
    "Anthropic-Ratelimit-Unified-Status": " rejected ",
    "anthropic-ratelimit-unified-reset": "1800003600",
    "anthropic-ratelimit-unified-representative-claim": "seven_day",
  });
  assert.deepEqual(unifiedLimit(headers), {
    status: "rejected",
    reset: "1800003600",
    claim: "seven_day",
  });
  assert.deepEqual(unifiedLimit(new Headers()), {
    status: undefined,
    reset: undefined,
    claim: undefined,
  });
});

test("an exhausted model is remembered until its reset; the later reset wins", () => {
  let clock = now;
  const models = new ExhaustedModels(() => clock);
  assert.equal(models.get("opus"), undefined);
  assert.deepEqual(
    models.mark("opus", { until: now + 60, claim: "five_hour" }),
    {
      until: now + 60,
      claim: "five_hour",
    },
  );
  assert.equal(models.mark("opus", { until: now + 30 }).until, now + 60);
  assert.equal(
    models.mark("opus", { until: now + 90, claim: "seven_day" }).claim,
    "seven_day",
  );
  assert.equal(models.get("sonnet"), undefined);
  clock = now + 89;
  assert.equal(models.get("opus")?.until, now + 90);
  clock = now + 90;
  assert.equal(models.get("opus"), undefined);
});

test("the refusal names the claim and the reset, with a distinct code", () => {
  assert.equal(exhaustionClaim("five_hour"), "five_hour");
  assert.equal(exhaustionClaim("Five Hours"), undefined);
  assert.equal(exhaustionClaim("x".repeat(65)), undefined);
  assert.deepEqual(exhaustionFailure({ until: now, claim: "five_hour" }), {
    status: 429,
    type: "rate_limit_error",
    code: "subscription_exhausted",
    message:
      "Claude subscription usage limit reached (five_hour); it resets at 2027-01-15T08:00:00.000Z",
    exhaustedUntil: now,
  });
  assert.match(exhaustionFailure({ until: now }).message, /reached; it resets/);
});
