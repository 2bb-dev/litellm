import { randomUUID } from "node:crypto";
import {
  chmodSync,
  closeSync,
  constants,
  existsSync,
  fchmodSync,
  fstatSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  realpathSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import type {
  AuthOperationOptions,
  Credential,
  CredentialInfo,
  CredentialStore,
} from "@earendil-works/pi-ai";
import { lockSync } from "proper-lockfile";
import { z } from "zod";

const providerIdSchema = z
  .string()
  .min(1)
  .refine(
    (value) => !["__proto__", "constructor", "prototype"].includes(value),
  );
const credentialSchema = z.discriminatedUnion("type", [
  z.strictObject({
    type: z.literal("api_key"),
    key: z
      .string()
      .refine((key) => !key.trimStart().startsWith("!"))
      .optional(),
    env: z.record(z.string(), z.string()).optional(),
  }),
  z
    .object({
      type: z.literal("oauth"),
      refresh: z.string(),
      access: z.string().min(1),
      expires: z.number().nonnegative(),
    })
    .catchall(z.json().optional()),
]);
const documentSchema = z.record(providerIdSchema, credentialSchema);
type CredentialDocument = Record<string, Credential>;

function validateDocument(value: unknown): CredentialDocument {
  const parsed = documentSchema.safeParse(value);
  if (!parsed.success) throw new Error("Invalid credential file data");
  return parsed.data;
}

function validateProviderId(providerId: string): void {
  if (!providerIdSchema.safeParse(providerId).success) {
    throw new Error("Invalid credential provider id");
  }
}

export class FileCredentialStore implements CredentialStore {
  readonly #file: string;
  readonly #release: () => void;
  #queue: Promise<void> = Promise.resolve();
  #closing?: Promise<void>;

  constructor(file: string) {
    if (
      !file.trim() ||
      file.endsWith("/") ||
      (existsSync(file) && !lstatSync(file).isFile())
    ) {
      throw new Error("Invalid credential file path");
    }
    const parent = dirname(resolve(file));
    if (parent === dirname(parent)) {
      throw new Error("Credential file requires an isolated parent directory");
    }
    mkdirSync(parent, { recursive: true, mode: 0o700 });
    chmodSync(parent, 0o700);
    this.#file = join(realpathSync(parent), basename(file));
    try {
      this.#release = lockSync(this.#file, {
        realpath: false,
        stale: 10_000,
        update: 5_000,
        onCompromised: () => {
          throw new Error(
            "Credential volume lock compromised; stopping to prevent concurrent refresh",
          );
        },
      });
    } catch {
      throw new Error(
        "Credential volume is locked or unavailable; run one service process per volume",
      );
    }
    try {
      this.#readDocument();
      if (!existsSync(this.#file)) this.#writeDocument({});
    } catch (error) {
      this.#release();
      throw error;
    }
  }

  #readDocument(): CredentialDocument {
    try {
      const fd = openSync(
        this.#file,
        constants.O_RDONLY | constants.O_NOFOLLOW,
      );
      try {
        const stat = fstatSync(fd);
        if (!stat.isFile() || stat.nlink !== 1)
          throw new Error("Unsafe credential file");
        fchmodSync(fd, 0o600);
        const content: unknown = JSON.parse(readFileSync(fd, "utf8"));
        return validateDocument(content);
      } finally {
        closeSync(fd);
      }
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT")
        return {};
      throw new Error("Unable to read valid credential file data");
    }
  }

  #writeDocument(document: CredentialDocument): void {
    const content = `${JSON.stringify(validateDocument(document), null, 2)}\n`;
    const temp = `${this.#file}.${randomUUID()}.tmp`;
    try {
      const fd = openSync(temp, "wx", 0o600);
      try {
        fchmodSync(fd, 0o600);
        writeFileSync(fd, content, "utf8");
        fsyncSync(fd);
      } finally {
        closeSync(fd);
      }
      renameSync(temp, this.#file);
    } catch {
      throw new Error("Unable to persist credential file data");
    } finally {
      if (existsSync(temp)) unlinkSync(temp);
    }
  }

  #enqueue<T>(
    task: () => Promise<T>,
    options?: AuthOperationOptions,
  ): Promise<T> {
    if (this.#closing)
      return Promise.reject(new Error("Credential store is closed"));
    const queued = this.#queue.then(() => {
      options?.signal?.throwIfAborted();
      return task();
    });
    this.#queue = queued.then(
      () => undefined,
      () => undefined,
    );
    return queued;
  }

  read(
    providerId: string,
    options?: AuthOperationOptions,
  ): Promise<Credential | undefined> {
    return this.#enqueue(async () => {
      validateProviderId(providerId);
      const document = this.#readDocument();
      return Object.hasOwn(document, providerId)
        ? document[providerId]
        : undefined;
    }, options);
  }

  list(options?: AuthOperationOptions): Promise<readonly CredentialInfo[]> {
    return this.#enqueue(
      async () =>
        Object.entries(this.#readDocument()).map(
          ([providerId, credential]) => ({ providerId, type: credential.type }),
        ),
      options,
    );
  }

  modify(
    providerId: string,
    fn: (current: Credential | undefined) => Promise<Credential | undefined>,
    options?: AuthOperationOptions,
  ): Promise<Credential | undefined> {
    return this.#enqueue(async () => {
      validateProviderId(providerId);
      const document = this.#readDocument();
      const current = Object.hasOwn(document, providerId)
        ? document[providerId]
        : undefined;
      const next = await fn(structuredClone(current));
      options?.signal?.throwIfAborted();
      if (next === undefined) return current;
      const updated = validateDocument({ ...document, [providerId]: next });
      this.#writeDocument(updated);
      return updated[providerId];
    }, options);
  }

  delete(providerId: string, options?: AuthOperationOptions): Promise<void> {
    return this.#enqueue(async () => {
      validateProviderId(providerId);
      const document = this.#readDocument();
      if (!Object.hasOwn(document, providerId)) return;
      this.#writeDocument(
        Object.fromEntries(
          Object.entries(document).filter(([id]) => id !== providerId),
        ),
      );
    }, options);
  }

  close(): Promise<void> {
    this.#closing ??= this.#queue.then(() => this.#release());
    return this.#closing;
  }
}
