#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DR: restore a Lambda's env vars from an S3 snapshot (see backup-lambda-env.sh).
#
# ⚠️  This uses `aws lambda update-function-configuration --environment`, which
# REPLACES ALL env vars — the exact command CLAUDE.md warns against for casual use.
# It is safe HERE only because the snapshot is a COMPLETE env dump, so we write the
# full set back. Guarded by an interactive confirmation + a completeness check.
#
# Usage:
#   AWS_PROFILE=rigacap ./scripts/restore-lambda-env.sh rigacap-prod-worker
#   AWS_PROFILE=rigacap ./scripts/restore-lambda-env.sh rigacap-prod-api dr/lambda-env/rigacap-prod-api-2026-08-07T....json
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

FN="${1:?usage: restore-lambda-env.sh <function-name> [s3-key]}"
PROFILE="${AWS_PROFILE:-rigacap}"
REGION="${AWS_REGION:-us-east-1}"
BUCKET="rigacap-prod-price-data-149218244179"
KEY="${2:-dr/lambda-env/${FN}-latest.json}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
chmod 700 "$TMP"

aws s3 cp "s3://$BUCKET/$KEY" "$TMP/env.json" \
  --profile "$PROFILE" --region "$REGION" --only-show-errors

KEYS=$(python3 -c "import json;print(len(json.load(open('$TMP/env.json'))))")
if [ "$KEYS" -lt 30 ]; then
  echo "⚠️  snapshot has only $KEYS keys — refusing to restore a partial env" >&2
  exit 1
fi

echo "About to REPLACE all env vars on '$FN' with $KEYS vars from:"
echo "  s3://$BUCKET/$KEY"
read -r -p "Type the function name to confirm: " CONF
[ "$CONF" = "$FN" ] || { echo "aborted."; exit 1; }

python3 -c "import json;d=json.load(open('$TMP/env.json'));open('$TMP/payload.json','w').write(json.dumps({'Variables':d}))"
aws lambda update-function-configuration --profile "$PROFILE" --region "$REGION" \
  --function-name "$FN" --environment "file://$TMP/payload.json" >/dev/null

echo "✅ restored $KEYS env vars to $FN"
