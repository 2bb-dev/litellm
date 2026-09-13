ALTER TABLE "LiteLLM_AccountingRequest" ADD COLUMN "user_id" TEXT;

-- Attribution for historical requests cannot be reconstructed from current keys.
-- Mixed key-v2/user-v3 runtimes are unsupported, including management traffic.
INSERT INTO "LiteLLM_AccountingProtocol" (id, version) VALUES ('user-v3', 3);
