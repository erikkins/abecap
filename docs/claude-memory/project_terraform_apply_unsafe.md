---
name: project_terraform_apply_unsafe
description: "DO NOT `terraform apply` — worker/api Lambda state is drifted; apply would revert prod env+image = outage"
metadata: 
  node_type: memory
  type: project
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

**`terraform apply` is currently UNSAFE and would cause a production outage.** (Discovered Aug 7 2026 during EventBridge-rule reconcile.)

A targeted `terraform plan` showed `aws_lambda_function.worker` badly drifted from live — apply would REVERT:
- `image_uri` → an older ECR image (undoing CI/CD deploys)
- `SIGNAL_UNIVERSE_SIZE`: 100 → 200 (reintroduces the universe-drift 0-signal-adjacent bug)
- `STRIPE_PRICE_ID` / `STRIPE_PRICE_ID_ANNUAL` → old wrong prices (the pricing bug)
- `META_IG_APP_ID` → blanked; IG/Threads/Meta tokens reverted

**Why:** live infra moves via CI/CD (container image deploys) + CLI env edits, while terraform's view stays frozen. This is the same class as the CLAUDE.md warning "NEVER `aws lambda update-function-configuration --environment`" — terraform apply would do exactly that wholesale.

**How to apply:** Safe to `terraform import` (state-only, non-destructive) and `terraform plan` (read-only). NEVER `terraform apply` (even targeted on Lambda) until `main.tf`'s worker AND api Lambda `image_uri` + all `environment.variables` are reconciled to match live (pull live env via `aws lambda get-function-configuration`, update main.tf, verify plan shows no Lambda change). Adding NEW standalone resources (e.g., new EventBridge rules) is fine to import, but do not apply if the plan also touches the Lambdas. CI/CD deploys do NOT use terraform (container path) — so this drift doesn't block deploys, only `terraform apply`.
