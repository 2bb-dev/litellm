class SmokeFailure extends Error {}

function check(condition, message) {
  if (!condition) throw new SmokeFailure(message);
}

function tokenCount(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function chatUsage(usage) {
  return (
    tokenCount(usage?.prompt_tokens) &&
    tokenCount(usage?.completion_tokens) &&
    tokenCount(usage?.total_tokens) &&
    usage.total_tokens === usage.prompt_tokens + usage.completion_tokens
  );
}

function hasText(value) {
  return typeof value === "string" && value.trim().length > 0;
}

async function* textChunks(response) {
  check(response.body, "Response has no body");
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let size = 0;
  for await (const chunk of response.body) {
    size += chunk.byteLength;
    check(size <= 8 * 1024 * 1024, "Response exceeds smoke size limit");
    yield decoder.decode(chunk, { stream: true });
  }
  yield decoder.decode();
}

async function jsonBody(response) {
  check(
    response.headers.get("content-type")?.includes("application/json"),
    "Expected a JSON response",
  );
  let text = "";
  for await (const chunk of textChunks(response)) text += chunk;
  return JSON.parse(text);
}

async function* sseLines(response) {
  let buffer = "";
  for await (const chunk of textChunks(response)) {
    buffer += chunk;
    check(buffer.length <= 1024 * 1024, "SSE line exceeds smoke size limit");
    let boundary;
    while ((boundary = /\r\n|\r(?!$)|\n/.exec(buffer))) {
      yield buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary[0].length);
    }
  }
  if (buffer.endsWith("\r")) yield buffer.slice(0, -1);
}

async function* sseEvents(response) {
  check(
    response.headers.get("content-type")?.includes("text/event-stream"),
    "Expected an SSE response",
  );
  let event = "";
  let data = [];
  let size = 0;
  for await (const line of sseLines(response)) {
    if (line === "") {
      if (data.length) yield { event, data: data.join("\n") };
      event = "";
      data = [];
      size = 0;
      continue;
    }
    if (line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator < 0 ? line : line.slice(0, separator);
    const value =
      separator < 0 ? "" : line.slice(separator + 1).replace(/^ /, "");
    if (field === "event") event = value;
    if (field !== "data") continue;
    size += value.length;
    check(size <= 1024 * 1024, "SSE event exceeds smoke size limit");
    data.push(value);
  }
}

async function verifyStream(response, protocol) {
  let text = false;
  let finished = false;
  let usage = false;
  let stopped = false;
  let started = false;
  for await (const frame of sseEvents(response)) {
    check(!stopped, "Received data after stream termination");
    check(frame.event !== "error", "Stream returned an error event");
    if (protocol === "chat" && frame.data === "[DONE]") {
      stopped = true;
      continue;
    }
    const item = JSON.parse(frame.data);
    check(!item.error && item.type !== "error", "Stream returned an error");
    if (protocol === "chat") {
      check(Array.isArray(item.choices), "Invalid Chat stream chunk");
      text ||= item.choices.some((choice) => hasText(choice.delta?.content));
      finished ||= item.choices.some((choice) => hasText(choice.finish_reason));
      if (item.usage) {
        check(chatUsage(item.usage), "Invalid Chat stream usage");
        usage ||= finished;
      }
      continue;
    }
    if (item.type === "message_start") {
      check(
        !started && tokenCount(item.message?.usage?.input_tokens),
        "Invalid Messages start usage",
      );
      started = true;
    }
    if (item.type === "content_block_delta") {
      text ||= item.delta?.type === "text_delta" && hasText(item.delta.text);
    }
    if (item.type === "message_delta" && hasText(item.delta?.stop_reason)) {
      check(started, "Messages stream finished before message_start");
      check(
        tokenCount(item.usage?.output_tokens),
        "Invalid Messages terminal usage",
      );
      finished = true;
      usage = true;
    }
    if (item.type === "message_stop") stopped = true;
  }
  check(text, "Stream contains no text");
  check(finished && stopped, "Stream is missing protocol termination");
  check(usage, "Stream is missing terminal usage");
}

function verifyCompletion(item, protocol) {
  check(!item.error, "Completion returned an error");
  if (protocol === "chat") {
    check(
      Array.isArray(item.choices) &&
        item.choices.some(
          (choice) =>
            hasText(choice.message?.content) && hasText(choice.finish_reason),
        ),
      "Chat completion is missing text or finish reason",
    );
    check(chatUsage(item.usage), "Invalid Chat completion usage");
    return;
  }
  check(
    item.type === "message" &&
      hasText(item.stop_reason) &&
      Array.isArray(item.content) &&
      item.content.some(
        (block) => block.type === "text" && hasText(block.text),
      ),
    "Messages completion is missing text or stop reason",
  );
  check(
    tokenCount(item.usage?.input_tokens) &&
      tokenCount(item.usage?.output_tokens),
    "Invalid Messages completion usage",
  );
}

async function main() {
  const protocol = process.env.INFERENCE_PROTOCOL ?? "chat";
  check(
    ["chat", "messages"].includes(protocol),
    "Use chat or messages protocol",
  );
  const base = new URL(
    process.env.INFERENCE_API_BASE ?? "http://127.0.0.1:4000",
  );
  check(
    ["http:", "https:"].includes(base.protocol) &&
      !base.username &&
      !base.password &&
      !base.search &&
      !base.hash &&
      ["/", "/v1", "/v1/"].includes(base.pathname),
    "INFERENCE_API_BASE must be a front LiteLLM root or /v1 URL without credentials",
  );
  check(
    base.port !== "4001" && !/^pi-inference(?:[.-]|$)/i.test(base.hostname),
    "Target the front LiteLLM proxy, not the private sidecar",
  );
  const key = process.env.INFERENCE_API_KEY;
  check(hasText(key), "INFERENCE_API_KEY must be a front LiteLLM key");
  const model =
    process.env.INFERENCE_MODEL ??
    (protocol === "chat" ? "pi/claude-chat" : "anthropic/claude-haiku-4-5/pi");
  check(hasText(model), "INFERENCE_MODEL must be a front LiteLLM alias");
  const request = async (path, body, authenticated = true) => {
    const response = await fetch(new URL(`/v1/${path}`, base), {
      method: body ? "POST" : "GET",
      redirect: "error",
      signal: AbortSignal.timeout(120_000),
      headers: {
        ...(authenticated ? { authorization: `Bearer ${key}` } : {}),
        ...(body ? { "content-type": "application/json" } : {}),
        ...(protocol === "messages"
          ? { "anthropic-version": "2023-06-01" }
          : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    if (authenticated)
      check(response.ok, `HTTP request failed (${response.status})`);
    return response;
  };
  const anonymous = await request("models", undefined, false);
  await anonymous.body?.cancel();
  check(
    [401, 403].includes(anonymous.status),
    "Front LiteLLM model discovery must require authentication",
  );
  const discovery = await jsonBody(await request("models"));
  check(
    Array.isArray(discovery.data) &&
      discovery.data.some((entry) => entry.id === model),
    "Configured alias is missing from authenticated model discovery",
  );
  console.log("PASS: authenticated front LiteLLM model discovery");
  if (process.env.INFERENCE_CONFIRM_PAID !== "1") {
    console.log("SKIP: paid calls require INFERENCE_CONFIRM_PAID=1");
    return;
  }
  const endpoint = protocol === "chat" ? "chat/completions" : "messages";
  const body = {
    model,
    messages: [{ role: "user", content: "Reply with just OK." }],
    max_tokens: 128,
  };
  verifyCompletion(
    await jsonBody(await request(endpoint, { ...body, stream: false })),
    protocol,
  );
  console.log("PASS: non-streaming text and terminal usage");
  await verifyStream(
    await request(endpoint, {
      ...body,
      stream: true,
      ...(protocol === "chat"
        ? { stream_options: { include_usage: true } }
        : {}),
    }),
    protocol,
  );
  console.log("PASS: streaming text, termination, and terminal usage");
}

main().catch((error) => {
  console.error(
    error instanceof SmokeFailure
      ? `FAIL: ${error.message}`
      : "FAIL: network, decoding, or protocol error (details suppressed)",
  );
  process.exitCode = 1;
});
