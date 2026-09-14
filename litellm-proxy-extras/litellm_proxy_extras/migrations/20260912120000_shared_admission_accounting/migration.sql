-- CreateTable
CREATE TABLE "LiteLLM_AccountingProtocol" (
    "id" TEXT NOT NULL,
    "version" INTEGER NOT NULL,

    CONSTRAINT "LiteLLM_AccountingProtocol_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_AccountingGeneration" (
    "id" TEXT NOT NULL,
    "version" INTEGER NOT NULL,
    "accepting" BOOLEAN NOT NULL DEFAULT true,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_AccountingGeneration_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_AccountingRequest" (
    "id" TEXT NOT NULL,
    "generation_id" TEXT NOT NULL,
    "key_token" TEXT,
    "dispatched" BOOLEAN NOT NULL DEFAULT false,
    "input_cost" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "sealed" BOOLEAN NOT NULL DEFAULT false,
    "failed" BOOLEAN NOT NULL DEFAULT false,
    "closed" BOOLEAN NOT NULL DEFAULT false,
    "expected" TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_AccountingRequest_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_AccountingComponent" (
    "id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "actual" BOOLEAN NOT NULL DEFAULT false,
    "total" DOUBLE PRECISION NOT NULL DEFAULT 0,

    CONSTRAINT "LiteLLM_AccountingComponent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_AccountingScope" (
    "id" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "entity_id" TEXT NOT NULL,
    "epoch" TEXT NOT NULL,

    CONSTRAINT "LiteLLM_AccountingScope_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "LiteLLM_AccountingHold" (
    "request_id" TEXT NOT NULL,
    "scope_id" TEXT NOT NULL,
    "amount" DOUBLE PRECISION NOT NULL,
    "disposition" TEXT NOT NULL DEFAULT 'held',

    CONSTRAINT "LiteLLM_AccountingHold_pkey" PRIMARY KEY ("request_id","scope_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_AccountingReceipt" (
    "id" TEXT NOT NULL,
    "component_id" TEXT NOT NULL,
    "basis" TEXT NOT NULL,
    "payload_hash" TEXT NOT NULL,
    "total" DOUBLE PRECISION NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_AccountingReceipt_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_AccountingRequest_generation_id_closed_idx" ON "LiteLLM_AccountingRequest"("generation_id", "closed");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AccountingScope_kind_entity_id_epoch_key" ON "LiteLLM_AccountingScope"("kind", "entity_id", "epoch");

-- CreateIndex
CREATE INDEX "LiteLLM_AccountingHold_scope_id_disposition_idx" ON "LiteLLM_AccountingHold"("scope_id", "disposition");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_AccountingReceipt_component_id_basis_key" ON "LiteLLM_AccountingReceipt"("component_id", "basis");

-- AddForeignKey
ALTER TABLE "LiteLLM_AccountingRequest" ADD CONSTRAINT "LiteLLM_AccountingRequest_generation_id_fkey" FOREIGN KEY ("generation_id") REFERENCES "LiteLLM_AccountingGeneration"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_AccountingComponent" ADD CONSTRAINT "LiteLLM_AccountingComponent_request_id_fkey" FOREIGN KEY ("request_id") REFERENCES "LiteLLM_AccountingRequest"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_AccountingHold" ADD CONSTRAINT "LiteLLM_AccountingHold_request_id_fkey" FOREIGN KEY ("request_id") REFERENCES "LiteLLM_AccountingRequest"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_AccountingHold" ADD CONSTRAINT "LiteLLM_AccountingHold_scope_id_fkey" FOREIGN KEY ("scope_id") REFERENCES "LiteLLM_AccountingScope"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_AccountingReceipt" ADD CONSTRAINT "LiteLLM_AccountingReceipt_component_id_fkey" FOREIGN KEY ("component_id") REFERENCES "LiteLLM_AccountingComponent"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

ALTER TABLE "LiteLLM_AccountingHold" ADD CONSTRAINT "accounting_hold_finite" CHECK (amount >= 0 AND amount < 'Infinity'::float8);
ALTER TABLE "LiteLLM_AccountingComponent" ADD CONSTRAINT "accounting_total_finite" CHECK (total >= 0 AND total < 'Infinity'::float8);
ALTER TABLE "LiteLLM_AccountingReceipt" ADD CONSTRAINT "accounting_receipt_finite" CHECK (total >= 0 AND total < 'Infinity'::float8);
ALTER TABLE "LiteLLM_AccountingReceipt" ADD CONSTRAINT "accounting_receipt_basis" CHECK (basis IN ('actual', 'estimate', 'seal'));
INSERT INTO "LiteLLM_AccountingProtocol" (id, version) VALUES ('key-v1', 1);
