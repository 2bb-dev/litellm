ALTER TABLE "LiteLLM_AccountingComponent"
    ADD COLUMN "liability" DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN "dispatched" BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN "nonexecution" BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE "LiteLLM_AccountingComponent" ADD CONSTRAINT "accounting_liability_finite"
    CHECK (liability >= 0 AND liability < 'Infinity'::float8);

-- Historical key-v1 generations keep their protocol and execution owners.
INSERT INTO "LiteLLM_AccountingProtocol" (id, version) VALUES ('key-v2', 2);
