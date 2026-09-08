import {
  createProvider,
  envApiKeyAuth,
  type Api,
  type AuthContext,
  type CredentialStore,
  type Model,
  type Models,
  type Provider,
} from "@earendil-works/pi-ai";
import { anthropicMessagesApi } from "@earendil-works/pi-ai/api/anthropic-messages.lazy";
import { azureOpenAIResponsesApi } from "@earendil-works/pi-ai/api/azure-openai-responses.lazy";
import { bedrockConverseStreamApi } from "@earendil-works/pi-ai/api/bedrock-converse-stream.lazy";
import { googleGenerativeAIApi } from "@earendil-works/pi-ai/api/google-generative-ai.lazy";
import { googleVertexApi } from "@earendil-works/pi-ai/api/google-vertex.lazy";
import { mistralConversationsApi } from "@earendil-works/pi-ai/api/mistral-conversations.lazy";
import { openAICodexResponsesApi } from "@earendil-works/pi-ai/api/openai-codex-responses.lazy";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { openAIResponsesApi } from "@earendil-works/pi-ai/api/openai-responses.lazy";
import { piMessagesApi } from "@earendil-works/pi-ai/api/pi-messages.lazy";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";
import { z } from "zod";
import { invalid, type Result } from "./protocol.js";

const apiFactories = {
  "anthropic-messages": anthropicMessagesApi,
  "azure-openai-responses": azureOpenAIResponsesApi,
  "bedrock-converse-stream": bedrockConverseStreamApi,
  "google-generative-ai": googleGenerativeAIApi,
  "google-vertex": googleVertexApi,
  "mistral-conversations": mistralConversationsApi,
  "openai-codex-responses": openAICodexResponsesApi,
  "openai-completions": openAICompletionsApi,
  "openai-responses": openAIResponsesApi,
  "pi-messages": piMessagesApi,
};

const endpointSchema = z.url().refine((value) => {
  if (!URL.canParse(value)) return false;
  const url = new URL(value);
  return (
    (url.protocol === "http:" || url.protocol === "https:") &&
    !url.username &&
    !url.password &&
    !url.search &&
    !url.hash
  );
});
const metadataSchema = z
  .strictObject({
    api: z.enum(
      Object.keys(apiFactories) as [
        keyof typeof apiFactories,
        ...(keyof typeof apiFactories)[],
      ],
    ),
    contextWindow: z.number().int().positive(),
    maxTokens: z.number().int().positive(),
    reasoning: z.boolean(),
    input: z
      .array(z.enum(["text", "image"]))
      .min(1)
      .max(2),
    cost: z.strictObject({
      input: z.number().nonnegative(),
      output: z.number().nonnegative(),
      cacheRead: z.number().nonnegative(),
      cacheWrite: z.number().nonnegative(),
    }),
  })
  .refine((value) => value.maxTokens <= value.contextWindow)
  .refine((value) => new Set(value.input).size === value.input.length);
const routeSchema = z.strictObject({
  alias: z.string().min(1).max(256).regex(/^\S+$/),
  provider: z.string().regex(/^[a-z][a-z0-9.-]*$/),
  model: z.string().min(1).max(512).regex(/^\S+$/),
  baseUrl: endpointSchema.optional(),
  metadata: metadataSchema.optional(),
});
const configSchema = z.strictObject({ models: z.array(routeSchema).min(1) });

export interface InferenceRuntime {
  models: Models;
  routes: ReadonlyMap<string, Model<Api>>;
}

export interface RuntimeOptions {
  credentials?: CredentialStore;
  authContext?: AuthContext;
  builtinModels?: typeof builtinModels;
}

type RouteConfig = z.infer<typeof routeSchema>;

function resolveRoute(entry: RouteConfig, catalog: Models): Result<Model<Api>> {
  const known = catalog.getModel(entry.provider, entry.model);
  const provider = catalog.getProvider(entry.provider);
  if (!known && !entry.metadata) {
    return invalid("Unknown provider/model requires complete model metadata");
  }
  if (!provider && !entry.baseUrl) {
    return invalid("Custom providers require an explicit baseUrl");
  }
  if (
    entry.metadata &&
    provider &&
    !provider.getModels().some((model) => model.api === entry.metadata?.api) &&
    !(entry.provider === "radius" && entry.metadata.api === "pi-messages")
  ) {
    return invalid(
      "Model API is not supported by the configured builtin provider",
    );
  }
  const baseUrl =
    entry.baseUrl ??
    known?.baseUrl ??
    provider?.baseUrl ??
    provider?.getModels().find((model) => model.api === entry.metadata?.api)
      ?.baseUrl;
  if (baseUrl === undefined)
    return invalid("Model requires an explicit baseUrl");
  if (known) {
    return { ok: true, value: { ...known, ...entry.metadata, baseUrl } };
  }
  if (!entry.metadata) return invalid("Model requires complete metadata");
  return {
    ok: true,
    value: {
      ...entry.metadata,
      id: entry.model,
      name: entry.model,
      provider: entry.provider,
      baseUrl,
    },
  };
}

function registerModels(
  provider: Provider | undefined,
  id: string,
  entries: readonly Model<Api>[],
): Provider {
  if (provider) {
    const existing = provider.getModels();
    return {
      ...provider,
      getModels: () => [
        ...existing,
        ...entries.filter(
          (entry) => !existing.some((model) => model.id === entry.id),
        ),
      ],
    };
  }
  return createProvider({
    id,
    auth: {
      apiKey: envApiKeyAuth(`${id} API key`, [
        `${id.toUpperCase().replace(/[^A-Z0-9]/g, "_")}_API_KEY`,
      ]),
    },
    models: entries,
    api: Object.fromEntries(
      Object.entries(apiFactories).map(([api, factory]) => [api, factory()]),
    ),
  });
}

export function loadRuntime(
  config: unknown,
  options: RuntimeOptions = {},
): Result<InferenceRuntime> {
  const parsed = configSchema.safeParse(config);
  if (!parsed.success) return invalid("Invalid runtime model configuration");
  if (
    new Set(parsed.data.models.map((entry) => entry.alias)).size !==
    parsed.data.models.length
  ) {
    return invalid("Model aliases must be unique");
  }
  const models = (options.builtinModels ?? builtinModels)({
    credentials: options.credentials,
    authContext: options.authContext,
  });
  const resolved = parsed.data.models.map((entry) => ({
    alias: entry.alias,
    result: resolveRoute(entry, models),
  }));
  const failure = resolved.find((entry) => !entry.result.ok);
  if (failure && !failure.result.ok) return failure.result;
  const routes = new Map(
    resolved.flatMap(({ alias, result }) =>
      result.ok ? [[alias, result.value] as const] : [],
    ),
  );
  for (const id of new Set(
    [...routes.values()].map((model) => model.provider),
  )) {
    models.setProvider(
      registerModels(
        models.getProvider(id),
        id,
        [...routes.values()].filter((model) => model.provider === id),
      ),
    );
  }
  return { ok: true, value: { models, routes } };
}
