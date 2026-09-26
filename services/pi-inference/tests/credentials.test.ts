import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { EventEmitter, once } from "node:events";
import {
  chmodSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test, { type TestContext } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import {
  createModels,
  fauxProvider,
  type Credential,
  type OAuthCredential,
} from "@earendil-works/pi-ai";
import { FileCredentialStore } from "../src/credentials.js";

const expired: OAuthCredential = {
  type: "oauth",
  access: "old-test-access",
  refresh: "old-test-refresh",
  expires: 0,
  accountId: "test-account",
};
const unrelated: Credential = {
  type: "api_key",
  key: "unrelated-test-key",
  env: { CLOUDFLARE_ACCOUNT_ID: "test-account" },
};

function gate(): { promise: Promise<void>; resolve: () => void } {
  const events = new EventEmitter();
  return {
    promise: once(events, "release").then(() => undefined),
    resolve: () => {
      events.emit("release");
    },
  };
}

function tempFile(t: TestContext): {
  directory: string;
  file: string;
  open: () => FileCredentialStore;
} {
  const directory = mkdtempSync(join(tmpdir(), "pi-inference-credentials-"));
  const file = join(directory, "auth.json");
  const stores = new Set<FileCredentialStore>();
  t.after(async () => {
    await Promise.all([...stores].map((store) => store.close()));
    rmSync(directory, { recursive: true, force: true });
  });
  return {
    directory,
    file,
    open: () => {
      const store = new FileCredentialStore(file);
      stores.add(store);
      return store;
    },
  };
}

function oauth(credential: Credential | undefined): OAuthCredential {
  assert.equal(credential?.type, "oauth");
  if (credential?.type !== "oauth") assert.fail("Expected OAuth credential");
  return credential;
}

test("reads isolated Pi-compatible auth.json, preserves provider fields, and enumerates metadata only", async (t) => {
  const { file, directory, open } = tempFile(t);
  writeFileSync(
    file,
    JSON.stringify({ "openai-codex": expired, other: unrelated }),
    { mode: 0o644 },
  );
  const store = open();
  assert.deepEqual(await store.read("openai-codex"), expired);
  assert.deepEqual(await store.read("other"), unrelated);
  assert.equal(await store.read("missing"), undefined);
  assert.equal(await store.read("toString"), undefined);
  assert.deepEqual(await store.list(), [
    { providerId: "openai-codex", type: "oauth" },
    { providerId: "other", type: "api_key" },
  ]);
  assert.equal(statSync(file).mode & 0o777, 0o600);
  assert.equal(statSync(directory).mode & 0o777, 0o700);
});

test("serializes concurrent refresh-style updates across providers and atomically replaces private file", async (t) => {
  const { file, directory, open } = tempFile(t);
  const store = open();
  await store.modify("openai-codex", async () => expired);
  await store.modify("other", async () => unrelated);
  const inode = statSync(file).ino;
  const original = readFileSync(file, "utf8");
  const entered = gate();
  const release = gate();
  const refreshed = {
    ...expired,
    access: "fresh-test-access",
    refresh: "rotated-test-refresh",
    expires: Date.now() + 3_600_000,
  };
  const first = store.modify("openai-codex", async (current) => {
    assert.deepEqual(current, expired);
    entered.resolve();
    await release.promise;
    return refreshed;
  });
  await entered.promise;
  const second = store.modify("openai-codex", async (current) => {
    assert.deepEqual(current, refreshed);
    return { ...oauth(current), access: "second-test-access" };
  });
  const third = store.modify("third", async () => ({
    type: "api_key",
    key: "third-test-key",
  }));
  assert.equal(readFileSync(file, "utf8"), original);
  release.resolve();
  await Promise.all([first, second, third]);
  assert.equal(
    oauth(await store.read("openai-codex")).access,
    "second-test-access",
  );
  assert.deepEqual(await store.read("other"), unrelated);
  assert.equal((await store.list()).length, 3);
  assert.notEqual(statSync(file).ino, inode);
  assert.equal(statSync(file).mode & 0o777, 0o600);
  assert.equal(statSync(directory).mode & 0o777, 0o700);
  assert.deepEqual(readdirSync(directory).sort(), [
    "auth.json",
    "auth.json.lock",
  ]);
});

test("directory sync failure rejects a credential update rather than acknowledging durability", async (t) => {
  const { file } = tempFile(t);
  writeFileSync(file, "{}\n", { mode: 0o600 });
  const source = new URL("../src/credentials.ts", import.meta.url).href;
  const child = spawnSync(
    process.execPath,
    [
      "--import",
      "tsx",
      "--input-type=module",
      "--eval",
      `
      import fs from 'node:fs';
      import { syncBuiltinESMExports } from 'node:module';
      const original = fs.fsyncSync;
      fs.fsyncSync = (fd) => {
        if (fs.fstatSync(fd).isDirectory()) throw new Error('synthetic directory sync failure');
        return original(fd);
      };
      syncBuiltinESMExports();
      const { FileCredentialStore } = await import(${JSON.stringify(source)});
      const store = new FileCredentialStore(${JSON.stringify(file)});
      try {
        await store.modify('provider', async () => ({ type: 'api_key', key: 'synthetic-test-key' }));
        process.exitCode = 1;
      } catch (error) {
        process.exitCode = /Unable to persist credential/.test(error.message) ? 0 : 2;
      } finally { await store.close(); }
      `,
    ],
    { encoding: "utf8" },
  );
  assert.equal(child.status, 0, child.stderr);
  assert.equal(child.stdout, "");
});

test("failed, invalid and unchanged modifications leave previous credentials intact and queue usable", async (t) => {
  const { file, open } = tempFile(t);
  const store = open();
  await store.modify("provider", async () => expired);
  await store.modify("other", async () => unrelated);
  const original = readFileSync(file, "utf8");
  await assert.rejects(
    store.modify("provider", async (current) => {
      oauth(current).access = "mutated-test-access";
      throw new Error("Test refresh failed");
    }),
    /Test refresh failed/,
  );
  assert.equal(readFileSync(file, "utf8"), original);
  const unchanged = await store.modify("provider", async (current) => {
    oauth(current).access = "discarded-test-access";
    return undefined;
  });
  assert.deepEqual(unchanged, expired);
  await assert.rejects(
    store.modify("provider", async () => ({ ...expired, expires: NaN })),
    /Invalid credential/,
  );
  assert.equal(readFileSync(file, "utf8"), original);
  await store.modify("provider", async () => ({
    ...expired,
    access: "next-test-access",
  }));
  assert.equal(oauth(await store.read("provider")).access, "next-test-access");
  assert.deepEqual(await store.read("other"), unrelated);
});

test("persists and serializes deletes, survives reopening, and returns missing for absent credentials", async (t) => {
  const { open } = tempFile(t);
  const store = open();
  await store.modify("provider", async () => expired);
  await Promise.all([
    store.modify("provider", async () => ({
      ...expired,
      access: "new-test-access",
    })),
    store.delete("provider"),
    store.modify("other", async () => unrelated),
  ]);
  assert.equal(await store.read("provider"), undefined);
  await store.delete("missing");
  await store.close();
  const reopened = open();
  assert.equal(await reopened.read("provider"), undefined);
  assert.deepEqual(await reopened.read("other"), unrelated);
});

test("lifetime lock rejects another instance and another process without deleting the active lock", async (t) => {
  const { file, open } = tempFile(t);
  const store = open();
  assert.throws(() => new FileCredentialStore(file), /locked/);
  const source = new URL("../src/credentials.ts", import.meta.url).href;
  const child = spawnSync(
    process.execPath,
    [
      "--import",
      "tsx",
      "--input-type=module",
      "--eval",
      `
    import { FileCredentialStore } from ${JSON.stringify(source)};
    try { new FileCredentialStore(${JSON.stringify(file)}); process.exitCode = 1; }
    catch (error) { process.exitCode = /locked/.test(error.message) ? 0 : 2; }
  `,
    ],
    { encoding: "utf8" },
  );
  assert.equal(child.status, 0);
  assert.equal(child.stdout, "");
  assert.equal(child.stderr, "");
  assert.throws(() => new FileCredentialStore(file), /locked/);
  await store.close();
  const reopened = open();
  assert.deepEqual(await reopened.list(), []);
});

test(
  "heartbeat protects live locks while a SIGKILLed process's lock recovers after the stale window",
  { timeout: 20_000 },
  async (t) => {
    const live = tempFile(t);
    const crashed = tempFile(t);
    live.open();
    const initialHeartbeat = statSync(`${live.file}.lock`).mtimeMs;
    const source = new URL("../src/credentials.ts", import.meta.url).href;
    const child = spawn(
      process.execPath,
      [
        "--import",
        "tsx",
        "--input-type=module",
        "--eval",
        `
      import { FileCredentialStore } from ${JSON.stringify(source)};
      const store = new FileCredentialStore(${JSON.stringify(crashed.file)});
      await store.modify("provider", async () => (${JSON.stringify(expired)}));
      process.send("ready");
      setInterval(() => {}, 1000);
    `,
      ],
      { stdio: ["ignore", "pipe", "pipe", "ipc"] },
    );
    const output: string[] = [];
    assert.ok(child.stdout && child.stderr);
    child.stdout.on("data", (chunk: Buffer) => output.push(chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => output.push(chunk.toString()));
    const exited = once(child, "exit");
    try {
      await once(child, "message", { signal: AbortSignal.timeout(5_000) });
      assert.throws(() => new FileCredentialStore(crashed.file), /locked/);
      const original = readFileSync(crashed.file, "utf8");
      child.kill("SIGKILL");
      const [, signal] = await exited;
      assert.equal(signal, "SIGKILL");
      assert.ok(statSync(`${crashed.file}.lock`).isDirectory());
      assert.throws(() => new FileCredentialStore(crashed.file), /locked/);
      const staleAt = statSync(`${crashed.file}.lock`).mtimeMs + 10_000;
      await delay(Math.max(0, staleAt - Date.now()) + 150);
      assert.ok(statSync(`${live.file}.lock`).mtimeMs > initialHeartbeat);
      assert.throws(() => new FileCredentialStore(live.file), /locked/);
      const recovered = crashed.open();
      assert.deepEqual(await recovered.read("provider"), expired);
      assert.equal(readFileSync(crashed.file, "utf8"), original);
      await recovered.close();
      assert.deepEqual(await crashed.open().read("provider"), expired);
      assert.deepEqual(output, []);
    } finally {
      child.kill("SIGKILL");
      await exited;
    }
  },
);

test("cancellation does not commit and close waits for active refresh before releasing lock", async (t) => {
  const { file, open } = tempFile(t);
  const store = open();
  await store.modify("provider", async () => expired);
  const controller = new AbortController();
  const entered = gate();
  const release = gate();
  const modification = store.modify(
    "provider",
    async () => {
      entered.resolve();
      await release.promise;
      return { ...expired, access: "cancelled-test-access" };
    },
    { signal: controller.signal },
  );
  await entered.promise;
  controller.abort();
  const rejected = assert.rejects(modification, { name: "AbortError" });
  const closed = store.close();
  await assert.rejects(store.read("provider"), /closed/);
  assert.throws(() => new FileCredentialStore(file), /locked/);
  release.resolve();
  await rejected;
  await closed;
  const reopened = open();
  assert.deepEqual(await reopened.read("provider"), expired);
  const aborted = AbortSignal.abort();
  await assert.rejects(reopened.delete("provider", { signal: aborted }), {
    name: "AbortError",
  });
  assert.deepEqual(await reopened.read("provider"), expired);
});

test("rejects invalid and command-style credential data without exposing or executing secrets", async (t) => {
  const { file, directory, open } = tempFile(t);
  const marker = join(directory, "should-not-exist");
  for (const contents of [
    '{"provider":{"type":"api_key","key":"secret-not-for-output"',
    JSON.stringify({ provider: { type: "api_key", key: `!touch ${marker}` } }),
    JSON.stringify({
      provider: { type: "oauth", access: "secret-not-for-output" },
    }),
    "[]",
  ]) {
    writeFileSync(file, contents);
    assert.throws(
      () => new FileCredentialStore(file),
      (error: unknown) => {
        assert.ok(error instanceof Error);
        assert.equal(error.message.includes("secret-not-for-output"), false);
        return /credential file data/.test(error.message);
      },
    );
    assert.equal(readdirSync(directory).includes("auth.json.lock"), false);
  }
  writeFileSync(file, "{}");
  const store = open();
  await assert.rejects(
    store.modify("provider", async () => ({
      type: "api_key",
      key: `!touch ${marker}`,
    })),
    /Invalid credential/,
  );
  assert.deepEqual(await store.list(), []);
  assert.equal(readdirSync(directory).includes("should-not-exist"), false);
});

test("rejects symlinked credential files without reading or changing the target", (t) => {
  const { file, directory } = tempFile(t);
  const target = join(directory, "target.json");
  writeFileSync(target, JSON.stringify({ other: unrelated }), { mode: 0o644 });
  chmodSync(target, 0o644);
  symlinkSync(target, file);
  assert.throws(() => new FileCredentialStore(file), /credential file/);
  assert.equal(statSync(target).mode & 0o777, 0o644);
});

test("SDK OAuth resolution refreshes once under storage serialization and persists tokens without losing another provider", async (t) => {
  const { open } = tempFile(t);
  const store = open();
  await store.modify("test-oauth", async () => expired);
  await store.modify("other", async () => unrelated);
  const refreshes: OAuthCredential[] = [];
  const faux = fauxProvider({ provider: "test-oauth" });
  const models = createModels({
    credentials: store,
    authContext: {
      env: async () => assert.fail("Stored OAuth must not fall back to env"),
      fileExists: async () => false,
    },
  });
  models.setProvider({
    ...faux.provider,
    auth: {
      oauth: {
        name: "Test OAuth",
        login: async () => expired,
        refresh: async (current) => {
          refreshes.push(current);
          await new Promise<void>((resolve) => setImmediate(resolve));
          return {
            ...current,
            access: "sdk-test-access",
            refresh: "sdk-test-refresh",
            expires: Date.now() + 3_600_000,
          };
        },
        toAuth: async (credential) => ({ apiKey: credential.access }),
      },
    },
  });
  const results = await Promise.all(
    Array.from({ length: 12 }, () => models.getAuth("test-oauth")),
  );
  assert.equal(refreshes.length, 1);
  for (const result of results)
    assert.equal(result?.auth.apiKey, "sdk-test-access");
  assert.equal(
    oauth(await store.read("test-oauth")).refresh,
    "sdk-test-refresh",
  );
  assert.deepEqual(await store.read("other"), unrelated);
  await store.close();
  const reopened = open();
  assert.equal(
    oauth(await reopened.read("test-oauth")).access,
    "sdk-test-access",
  );
});

test("SDK refresh failure preserves credentials and never falls back to environment API keys", async (t) => {
  const { file, open } = tempFile(t);
  const store = open();
  await store.modify("test-oauth", async () => expired);
  await store.modify("other", async () => unrelated);
  const original = readFileSync(file, "utf8");
  const faux = fauxProvider({ provider: "test-oauth" });
  const models = createModels({ credentials: store });
  models.setProvider({
    ...faux.provider,
    auth: {
      apiKey: {
        name: "Test API key",
        resolve: async () =>
          assert.fail("Must not fall back to environment auth"),
      },
      oauth: {
        name: "Test OAuth",
        login: async () => expired,
        refresh: async () => {
          throw new Error("Test refresh rejected");
        },
        toAuth: async () => assert.fail("Must not use expired tokens"),
      },
    },
  });
  await assert.rejects(models.getAuth("test-oauth"), (error: unknown) => {
    assert.ok(error instanceof Error && "code" in error);
    assert.equal(error.code, "oauth");
    return true;
  });
  assert.equal(readFileSync(file, "utf8"), original);
  assert.deepEqual(await store.read("other"), unrelated);
});

test("persists SDK OAuth optional fields and permanent OpenRouter credentials in Pi-compatible JSON", async (t) => {
  const { file, open } = tempFile(t);
  const store = open();
  await store.modify("github-copilot", async () => ({
    ...expired,
    enterpriseUrl: undefined,
    availableModelIds: ["test-model"],
  }));
  await store.modify("openrouter", async () => ({
    type: "oauth",
    access: "permanent-test-key",
    refresh: "",
    expires: Number.MAX_SAFE_INTEGER,
  }));
  assert.deepEqual(await store.read("github-copilot"), {
    ...expired,
    availableModelIds: ["test-model"],
  });
  assert.equal(oauth(await store.read("openrouter")).refresh, "");
  assert.equal(readFileSync(file, "utf8").includes("enterpriseUrl"), false);
});
