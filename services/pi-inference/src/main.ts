import { readFile } from "node:fs/promises";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";
import { FileCredentialStore } from "./credentials.js";
import { loadRuntime } from "./runtime.js";
import { createInferenceServer } from "./server.js";

async function main(): Promise<void> {
  if (process.argv[2] === "catalog") {
    const models = builtinModels();
    console.log(
      JSON.stringify(
        {
          providers: models.getProviders().map((provider) => ({
            id: provider.id,
            name: provider.name,
            api_key: Boolean(provider.auth.apiKey),
            oauth: Boolean(provider.auth.oauth),
            models: provider.getModels().map((model) => ({
              id: model.id,
              api: model.api,
              contextWindow: model.contextWindow,
              maxTokens: model.maxTokens,
              reasoning: model.reasoning,
              input: model.input,
            })),
          })),
        },
        null,
        2,
      ),
    );
    return;
  }
  const apiKey = process.env.PI_INFERENCE_API_KEY;
  const path = process.env.PI_INFERENCE_CONFIG;
  const port = Number(process.env.PI_INFERENCE_PORT ?? 4001);
  if (
    !apiKey ||
    apiKey.length < 32 ||
    !path ||
    !Number.isInteger(port) ||
    port < 1 ||
    port > 65535
  ) {
    console.error(
      "Set PI_INFERENCE_CONFIG, a PI_INFERENCE_API_KEY of at least 32 characters, and a valid PI_INFERENCE_PORT",
    );
    process.exitCode = 1;
    return;
  }
  const config: unknown = JSON.parse(await readFile(path, "utf8"));
  const slot = process.env.PI_INFERENCE_SLOT_ID ?? "default";
  if (!/^[A-Za-z0-9_.-]{1,64}$/.test(slot)) {
    console.error(
      "PI_INFERENCE_SLOT_ID must be a short alphanumeric identifier",
    );
    process.exitCode = 1;
    return;
  }
  const credentials = process.env.PI_INFERENCE_AUTH_FILE
    ? new FileCredentialStore(process.env.PI_INFERENCE_AUTH_FILE)
    : undefined;
  try {
    if (credentials) await credentials.list();
    const runtime = loadRuntime(config, { credentials });
    if (!runtime.ok) {
      await credentials?.close();
      console.error(runtime.error.message);
      process.exitCode = 1;
      return;
    }
    const backend = createInferenceServer({
      apiKey,
      runtime: runtime.value,
      log: (record) =>
        console.log(JSON.stringify({ ...record, slot_id: slot })),
    });
    backend.server.once("close", () => {
      void credentials?.close();
    });
    backend.server.on("error", () => {
      void credentials?.close();
      console.error("Pi inference listener failed");
      process.exitCode = 1;
    });
    backend.server.listen(
      port,
      process.env.PI_INFERENCE_HOST ?? "127.0.0.1",
      () => {
        console.log(
          JSON.stringify({
            event: "pi_inference.listening",
            port,
            models: runtime.value.routes.size,
            slot_id: slot,
          }),
        );
      },
    );
    const shutdown = () => {
      backend.abortAll();
      backend.server.close();
      const timer = setTimeout(
        () => backend.server.closeAllConnections(),
        5_000,
      );
      timer.unref();
    };
    process.once("SIGTERM", shutdown);
    process.once("SIGINT", shutdown);
  } catch {
    await credentials?.close();
    console.error(
      "Pi inference startup failed; check model configuration and credential volume permissions",
    );
    process.exitCode = 1;
  }
}

void main().catch(() => {
  console.error(
    "Pi inference startup failed; check model configuration and credential volume permissions",
  );
  process.exitCode = 1;
});
