-- AlterTable: add the opt-in strict reservation policy to virtual keys
ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "budget_enforcement" TEXT;
ALTER TABLE "LiteLLM_DeletedVerificationToken" ADD COLUMN IF NOT EXISTS "budget_enforcement" TEXT;
