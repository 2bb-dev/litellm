ALTER TABLE "LiteLLM_AccountingRequest" ADD COLUMN "team_id" TEXT;

-- Captured Team attribution cannot be reconstructed from mutable keys.
-- Requires offline cutover and new generation UUIDs; old writers/resetters must stop.
INSERT INTO "LiteLLM_AccountingProtocol" (id, version) VALUES ('team-v4', 4);
