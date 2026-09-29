import assert from "node:assert/strict";
import test from "node:test";
import {
  InMemoryCredentialStore,
  type AuthContext,
} from "@earendil-works/pi-ai";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";
import { loadRuntime, type InferenceRuntime } from "../src/runtime.js";
import type { Result } from "../src/protocol.js";

const route = { alias: "chat", provider: "openai", model: "gpt-4o-mini" };
const metadata = {
  api: "openai-completions",
  contextWindow: 8192,
  maxTokens: 1024,
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
const noAuth: AuthContext = {
  env: async () => undefined,
  fileExists: async () => false,
};

function runtime(result: Result<InferenceRuntime>): InferenceRuntime {
  assert.equal(result.ok, true);
  if (!result.ok) assert.fail("Expected valid runtime");
  return result.value;
}

function invalidConfig(config: unknown): void {
  const result = loadRuntime(config, { authContext: noAuth });
  assert.equal(result.ok, false);
  if (result.ok) assert.fail("Expected invalid config");
  assert.equal(result.error.status, 400);
  assert.equal(result.error.type, "invalid_request_error");
}

test("registers every builtin runtime provider, exposes only explicit aliases, and does not resolve startup auth", () => {
  const loaded = runtime(
    loadRuntime(
      { models: [route] },
      {
        authContext: {
          env: async () => assert.fail("Startup must not resolve credentials"),
          fileExists: async () =>
            assert.fail("Startup must not inspect credential files"),
        },
      },
    ),
  );
  assert.equal(loaded.models.getProviders().length, 41);
  assert.ok(loaded.models.getProvider("radius"));
  assert.ok(loaded.models.getProvider("amazon-bedrock"));
  assert.deepEqual([...loaded.routes.keys()], ["chat"]);
  assert.equal(loaded.routes.get("chat")?.id, "gpt-4o-mini");
  assert.equal(loaded.routes.get("chat")?.provider, "openai");
  assert.equal(loaded.routes.has("gpt-4o-mini"), false);
});

test("every static builtin provider's model can be explicitly allowlisted", () => {
  const catalog = builtinModels({ authContext: noAuth });
  const routes = catalog.getProviders().flatMap((provider) => {
    const model = provider.getModels()[0];
    return model
      ? [{ alias: provider.id, provider: provider.id, model: model.id }]
      : [];
  });
  const loaded = runtime(
    loadRuntime({ models: routes }, { authContext: noAuth }),
  );
  assert.equal(loaded.routes.size, routes.length);
  for (const configured of routes) {
    assert.equal(loaded.routes.get(configured.alias)?.id, configured.model);
  }
});

test("uses injected builtin collection factory and defers environment auth until request time", async () => {
  const calls: string[] = [];
  const loaded = runtime(
    loadRuntime(
      { models: [route] },
      {
        builtinModels: (options) => {
          calls.push("catalog");
          return builtinModels(options);
        },
        authContext: {
          ...noAuth,
          env: async (name) => {
            calls.push(name);
            return name === "OPENAI_API_KEY" ? "test-env-key" : undefined;
          },
        },
      },
    ),
  );
  assert.deepEqual(calls, ["catalog"]);
  assert.equal(
    (await loaded.models.getAuth("openai"))?.auth.apiKey,
    "test-env-key",
  );
  assert.deepEqual(calls, ["catalog", "OPENAI_API_KEY"]);
});

test("stored provider credentials win over environment keys", async () => {
  const credentials = new InMemoryCredentialStore();
  await credentials.modify("openai", async () => ({
    type: "api_key",
    key: "stored-test-key",
  }));
  const loaded = runtime(
    loadRuntime(
      { models: [route] },
      {
        credentials,
        authContext: {
          ...noAuth,
          env: async () => assert.fail("Must use stored key"),
        },
      },
    ),
  );
  assert.equal(
    (await loaded.models.getAuth("openai"))?.auth.apiKey,
    "stored-test-key",
  );
});

test("endpoint overrides are per alias and do not mutate builtin catalog models", () => {
  const loaded = runtime(
    loadRuntime(
      {
        models: [
          { ...route, alias: "stub", baseUrl: "http://127.0.0.1:9999/v1" },
          route,
        ],
      },
      { authContext: noAuth },
    ),
  );
  assert.equal(loaded.routes.get("stub")?.baseUrl, "http://127.0.0.1:9999/v1");
  assert.equal(loaded.routes.get("chat")?.baseUrl, "https://api.openai.com/v1");
  assert.equal(
    loaded.models.getModel("openai", route.model)?.baseUrl,
    "https://api.openai.com/v1",
  );
});

test("complete metadata registers a new upstream model with existing provider auth and API", () => {
  const loaded = runtime(
    loadRuntime(
      {
        models: [
          {
            ...route,
            model: "private-deployment",
            metadata: { ...metadata, api: "openai-responses" },
          },
        ],
      },
      { authContext: noAuth },
    ),
  );
  assert.equal(loaded.routes.get("chat")?.id, "private-deployment");
  assert.equal(loaded.routes.get("chat")?.baseUrl, "https://api.openai.com/v1");
  assert.equal(
    loaded.models.getModel("openai", "private-deployment")?.api,
    "openai-responses",
  );
});

test("custom provider resolves configured alias to upstream model and uses provider-scoped env auth", async () => {
  const loaded = runtime(
    loadRuntime(
      {
        models: [
          {
            alias: "public-name",
            provider: "internal-proxy",
            model: "upstream-name",
            baseUrl: "http://stub.invalid/v1",
            metadata,
          },
        ],
      },
      {
        authContext: {
          ...noAuth,
          env: async (name) =>
            name === "INTERNAL_PROXY_API_KEY" ? "stub-test-key" : undefined,
        },
      },
    ),
  );
  assert.equal(loaded.models.getProviders().length, 42);
  assert.equal(
    (await loaded.models.getAuth("internal-proxy"))?.auth.apiKey,
    "stub-test-key",
  );
  const model = loaded.routes.get("public-name");
  assert.ok(model);
  const response = await loaded.models.completeSimple(
    model,
    {
      messages: [{ role: "user", content: "Hi", timestamp: 0 }],
    },
    {
      fetch: async (url, options) => {
        assert.equal(String(url), "http://stub.invalid/v1/chat/completions");
        const body: unknown = JSON.parse(String(options?.body));
        assert.ok(body && typeof body === "object" && "model" in body);
        assert.equal(body.model, "upstream-name");
        assert.equal(
          new Headers(options?.headers).get("authorization"),
          "Bearer stub-test-key",
        );
        return new Response(
          'data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}\n\ndata: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
          {
            headers: { "content-type": "text/event-stream" },
          },
        );
      },
    },
  );
  assert.equal(response.stopReason, "stop");
  assert.deepEqual(response.content, [{ type: "text", text: "hello" }]);
});

test("rejects unknown routes without full metadata and unknown providers without explicit endpoints", () => {
  invalidConfig({ models: [{ ...route, model: "missing" }] });
  invalidConfig({ models: [{ ...route, provider: "missing" }] });
  invalidConfig({ models: [{ ...route, provider: "missing", metadata }] });
  invalidConfig({
    models: [{ ...route, metadata: { ...metadata, api: "not-an-api" } }],
  });
  invalidConfig({ models: [{ ...route, model: "missing", metadata }] });
  for (const field of Object.keys(metadata)) {
    invalidConfig({
      models: [
        {
          ...route,
          metadata: Object.fromEntries(
            Object.entries(metadata).filter(([key]) => key !== field),
          ),
        },
      ],
    });
  }
});

test("rejects malformed, ambiguous, unsafe and secret-bearing configuration as values", () => {
  for (const config of [
    null,
    [],
    {},
    { models: [] },
    { models: [route, route] },
    { models: [{ ...route, alias: " " }] },
    { models: [route], apiKey: "secret-value-not-for-output" },
    { models: [{ ...route, apiKey: "secret-value-not-for-output" }] },
    { models: [{ ...route, headers: { Authorization: "secret" } }] },
    { models: [{ ...route, options: { env: { OPENAI_API_KEY: "secret" } } }] },
    { models: [{ ...route, metadata: { ...metadata, maxTokens: 9000 } }] },
    {
      models: [
        {
          ...route,
          metadata: { ...metadata, cost: { ...metadata.cost, input: -1 } },
        },
      ],
    },
    {
      models: [
        { ...route, metadata: { ...metadata, input: ["text", "text"] } },
      ],
    },
    {
      models: [
        { ...route, metadata: { ...metadata, contextWindow: Infinity } },
      ],
    },
  ])
    invalidConfig(config);
  for (const baseUrl of [
    "not a URL",
    "file:///tmp/socket",
    "ftp://example.com",
    "https://user:secret@example.com",
    "https://example.com?key=secret",
    "https://example.com/#secret",
  ]) {
    invalidConfig({ models: [{ ...route, baseUrl }] });
  }
  assert.equal(
    JSON.stringify(
      loadRuntime({ models: [route], apiKey: "secret-value-not-for-output" }),
    ).includes("secret-value-not-for-output"),
    false,
  );
});

test("Radius accepts explicit pi-messages metadata without refreshing its empty startup catalog", () => {
  const configured = {
    alias: "radius-chat",
    provider: "radius",
    model: "admin-selected-radius-model",
    baseUrl: "http://radius.internal",
    metadata: { ...metadata, api: "pi-messages" },
  };
  const loaded = runtime(
    loadRuntime(
      { models: [configured] },
      {
        authContext: {
          env: async () => assert.fail("Must not resolve startup auth"),
          fileExists: async () => assert.fail("Must not inspect auth files"),
        },
      },
    ),
  );
  assert.equal(loaded.models.getProviders().length, 41);
  assert.equal(loaded.routes.get("radius-chat")?.api, "pi-messages");
  assert.equal(loaded.routes.get("radius-chat")?.id, configured.model);
  assert.ok(loaded.models.getModel("radius", configured.model));
  invalidConfig({ models: [{ ...configured, metadata }] });
});

test("missing credentials remain unconfigured at request time rather than preventing startup", async () => {
  const loaded = runtime(
    loadRuntime({ models: [route] }, { authContext: noAuth }),
  );
  assert.equal(await loaded.models.getAuth("openai"), undefined);
  assert.equal(await loaded.models.checkAuth("openai"), undefined);
  assert.deepEqual(await loaded.models.getAvailable("openai"), []);
});
