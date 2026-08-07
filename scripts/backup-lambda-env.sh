#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DR: snapshot both Lambdas' live env vars to a private, encrypted S3 location.
# Env is managed out-of-band (CI image deploys, token-refresh cron, CLI edits) and
# terraform ignores it (lifecycle.ignore_changes), so THIS snapshot is the recovery
# artifact if a Lambda is ever recreated/wiped. Pairs with restore-lambda-env.sh.
#
# Secrets are NEVER printed or committed — written to a temp file (auto-deleted) and
# uploaded with SSE. Run: AWS_PROFILE=rigacap ./scripts/backup-lambda-env.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROFILE="${AWS_PROFILE:-rigacap}"
REGION="${AWS_REGION:-us-east-1}"
BUCKET="rigacap-prod-price-data-149218244179"
PREFIX="dr/lambda-env"
FUNCTIONS=("rigacap-prod-worker" "rigacap-prod-api")

STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
chmod 700 "$TMP"

echo "DR env backup @ ${STAMP} (profile=${PROFILE})"
for FN in "${FUNCTIONS[@]}"; do
  aws lambda get-function-configuration --profile "$PROFILE" --region "$REGION" \
    --function-name "$FN" --query "Environment.Variables" --output json > "$TMP/$FN.json"

  KEYS=$(python3 -c "import json;print(len(json.load(open('$TMP/$FN.json'))))")
  if [ "$KEYS" -lt 30 ]; then
    echo "  ⚠️  $FN only $KEYS keys — refusing to upload a suspiciously small snapshot" >&2
    exit 1
  fi

  # timestamped (history) + latest (restore convenience), both server-side encrypted
  aws s3 cp "$TMP/$FN.json" "s3://$BUCKET/$PREFIX/$FN-$STAMP.json" \
    --sse AES256 --profile "$PROFILE" --region "$REGION" --only-show-errors
  aws s3 cp "$TMP/$FN.json" "s3://$BUCKET/$PREFIX/$FN-latest.json" \
    --sse AES256 --profile "$PROFILE" --region "$REGION" --only-show-errors

  echo "  ✅ $FN: $KEYS keys → s3://$BUCKET/$PREFIX/$FN-$STAMP.json (+ -latest.json)"
done
echo "done — secrets uploaded SSE-encrypted, nothing printed or committed."
