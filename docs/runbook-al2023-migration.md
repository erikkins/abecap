# Runbook: Lambda AL2023 Migration

**Goal:** Move Worker + API Lambdas from the current AL2-backed runtime to AL2023 for native DuckDB `httpfs` support, 10-25% faster Python execution, and modernized glibc.

**Expected duration:** 2-4 hours including validation.
**Rollback time:** <60 seconds (previous image tag stays in ECR).

---

## Pre-flight audit (run BEFORE changing anything)

### 1. Confirm current base image version

```bash
# Local docker pull to inspect what we're actually running
docker pull public.ecr.aws/lambda/python:3.11
docker run --rm public.ecr.aws/lambda/python:3.11 bash -c "
  cat /etc/os-release | head -3
  python --version
  ldd --version | head -1
"
```

**Expected finding:** current base is AL2 (Amazon Linux 2), glibc 2.26. This is why DuckDB's httpfs extension (needs glibc 2.28+) fails to load — confirmed during Layer 2 build on Apr 15 2026.

### 2. Confirm AL2023 image availability

AL2023 is the default base for Python 3.12+ Lambda images:

- `public.ecr.aws/lambda/python:3.12` → **AL2023** (glibc 2.34)
- `public.ecr.aws/lambda/python:3.11` → AL2 (older)
- `public.ecr.aws/lambda/python:3.13` → AL2023

**Choice: migrate to Python 3.12 on AL2023.**

### 3. Dependency audit for Python 3.12

Review `backend/requirements.txt` — all must support 3.12:

| Package | Version | Python 3.12 support |
|---|---|---|
| fastapi 0.109.0 | ✅ supports 3.12 |
| pydantic 2.5.0 | ✅ |
| sqlalchemy 2.0.25 | ✅ |
| asyncpg 0.29.0 | ✅ |
| greenlet 3.0.3 | ✅ |
| pandas 2.1.4 | ✅ |
| numpy 1.26.3 | ✅ |
| pyarrow 15.0.0 | ✅ |
| duckdb 1.4.4 | ✅ |
| optuna ≥3.5.0 | ✅ |
| matplotlib ≥3.8.0 | ✅ |
| yfinance ≥1.1.0 | ✅ |
| alpaca-py ≥0.30.0 | ✅ |
| boto3 1.34.0 | ✅ |
| mangum 0.17.0 | ✅ |
| httpx 0.26.0 | ✅ |
| aiohttp 3.9.1 | ✅ |

All green. No dep blockers.

### 4. Python 3.11 → 3.12 breaking-change audit

Key items to grep for in our codebase:

```bash
# Deprecated/removed in 3.12
grep -rnE "asyncio.get_event_loop\(\)" backend/app backend/main.py | wc -l
# get_event_loop() is DEPRECATED in 3.12 — Python will warn but not error
# Not breaking, but flag for cleanup

grep -rnE "distutils" backend/app backend/main.py
# distutils removed in 3.12. If anything imports it, fix before migrating.

grep -rnE "collections\.(Callable|Iterable|Mapping|Set)" backend/app backend/main.py
# Removed in 3.10 — verify no surviving refs

grep -rnE "imp\." backend/app backend/main.py
# imp module removed in 3.12. Use importlib instead.
```

Known issue in our code: we call `asyncio.get_event_loop()` in several Lambda handler branches. In 3.12 this raises `DeprecationWarning` but still works. Plan to migrate to `asyncio.new_event_loop()` + `asyncio.set_event_loop()` pattern post-3.12 cutover.

---

## Migration steps

### Step 1: Update Dockerfile

```dockerfile
# backend/Dockerfile.lambda
FROM public.ecr.aws/lambda/python:3.12    # was 3.11

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=contourpy,matplotlib -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy application code
COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY main.py ${LAMBDA_TASK_ROOT}/

CMD ["main.handler"]
```

### Step 2: Local build + smoke test

```bash
cd backend
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -f Dockerfile.lambda -t rigacap-prod-api:al2023-test --load .

# Smoke test: verify python version, glibc, duckdb httpfs
docker run --rm --entrypoint bash rigacap-prod-api:al2023-test -c "
  python --version
  ldd --version | head -1
  python -c 'import duckdb; c = duckdb.connect(); c.execute(\"INSTALL httpfs; LOAD httpfs;\"); print(\"httpfs OK\")'
  python -c 'from app.core.database import Base'
  python -c 'from app.services.scanner import scanner_service'
  python -c 'from app.services.symbol_metadata_service import symbol_metadata_service'
"
```

Expected output: `Python 3.12.x`, `GLIBC 2.34+`, `httpfs OK`, all imports succeed.

### Step 3: Push to ECR with staging tag

```bash
# Tag with al2023-canary to distinguish from production tag
aws ecr get-login-password --region us-east-1 --profile rigacap | \
  docker login --username AWS --password-stdin 149218244179.dkr.ecr.us-east-1.amazonaws.com

docker tag rigacap-prod-api:al2023-test \
  149218244179.dkr.ecr.us-east-1.amazonaws.com/rigacap-prod-api:al2023-canary

docker push 149218244179.dkr.ecr.us-east-1.amazonaws.com/rigacap-prod-api:al2023-canary
```

### Step 4: Canary deploy to WORKER only

API Lambda handles live user traffic — defer it. Start with Worker which handles scheduled jobs we can validate in isolation.

```bash
# Save current working image URI for rollback
CURRENT=$(aws lambda get-function --function-name rigacap-prod-worker \
  --region us-east-1 --profile rigacap --query 'Code.ImageUri' --output text)
echo "ROLLBACK: $CURRENT"

# Deploy canary to worker
aws lambda update-function-code \
  --function-name rigacap-prod-worker \
  --image-uri 149218244179.dkr.ecr.us-east-1.amazonaws.com/rigacap-prod-api:al2023-canary \
  --region us-east-1 --profile rigacap

# Wait for deployment
aws lambda wait function-updated \
  --function-name rigacap-prod-worker \
  --region us-east-1 --profile rigacap
```

### Step 5: Smoke test on Lambda

```bash
# Basic handler invocation — proves import + handler entry
aws lambda invoke --function-name rigacap-prod-worker --region us-east-1 --profile rigacap \
  --payload '{"alpaca_asset_probe": {"symbol": "AAPL"}}' \
  /tmp/al2023-test1.json
cat /tmp/al2023-test1.json

# DuckDB httpfs test — the feature we're upgrading FOR
# (Note: will require updating query_parquet to remove /tmp download workaround)
aws lambda invoke --function-name rigacap-prod-worker --region us-east-1 --profile rigacap \
  --payload '{"parquet_diagnose": {"_": 1}}' \
  --cli-read-timeout 300 /tmp/al2023-test2.json
cat /tmp/al2023-test2.json
```

### Step 6: Monitor CloudWatch for 30-60 min

```bash
# Tail logs during scheduled invocations (scan, hygiene, etc.)
aws logs tail /aws/lambda/rigacap-prod-worker --region us-east-1 --profile rigacap --follow
```

Watch for:
- ImportError / ModuleNotFoundError (dep issue)
- DeprecationWarning for get_event_loop (expected, non-breaking)
- Memory pressure changes (should be similar or better)
- Cold start duration (should be faster)

### Step 7: Roll out to API Lambda

Once Worker validated for at least one scheduled cycle (daily_scan runs clean):

```bash
aws lambda update-function-code \
  --function-name rigacap-prod-api \
  --image-uri 149218244179.dkr.ecr.us-east-1.amazonaws.com/rigacap-prod-api:al2023-canary \
  --region us-east-1 --profile rigacap
```

### Step 8: Post-migration cleanup

After 48 hours of stable operation:
- Remove `/tmp` download workaround from `data_export_service.query_parquet()` — use native httpfs
- Commit Dockerfile change to git (if not already)
- Update CI/CD build step if needed

---

## Rollback

```bash
# Replace $ROLLBACK with the tag you saved in Step 4
aws lambda update-function-code \
  --function-name rigacap-prod-worker \
  --image-uri $ROLLBACK \
  --region us-east-1 --profile rigacap

aws lambda update-function-code \
  --function-name rigacap-prod-api \
  --image-uri $ROLLBACK \
  --region us-east-1 --profile rigacap
```

Rollback is a 30-60 second operation. Any failure during Step 5-7 → roll back immediately, investigate offline.

---

## Known risks + mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Dep incompatibility | Low (all deps support 3.12) | Local smoke test catches before deploy |
| asyncio.get_event_loop() deprecation warnings | High (we have many calls) | Non-breaking in 3.12; plan cleanup post-migration |
| Slightly higher memory from 3.12 | Low | Monitor RSS in CloudWatch; adjust memory if needed |
| CI/CD expects specific image tag format | Low | Build uses same `rigacap-prod-api:` repo |
| Import-time failure not caught by local smoke test | Medium | Canary to Worker first (not API) so live users aren't affected |

---

## Bundle with this same-day (optional, ~60 min additional)

Since we're in deploy mode, consider bundling these related fixes:

### MMC-class false-positive fix in `verify_asset_ids`

Alpaca's Trading API returns 404 for some real symbols (MMC, possibly dozens more) that exist in their Data API. Our nightly hygiene marks them `missing_in_alpaca` which is misleading.

**Two-part fix:**
1. **Bulk pre-fetch** — call `TradingClient.get_all_assets()` once at start of `verify_asset_ids`, build a symbol→asset map, lookup against that instead of 4500 individual `get_asset()` calls. ~30 min.
2. **Data-API fallback** — if a symbol isn't in the Trading asset map, check if our pickle has recent bars for it. If yes → mark as `unverifiable` (don't alarm). Only flag `missing_in_alpaca` if BOTH APIs lack it. ~30 min.

See `~/.claude/projects/.../memory/feedback_alpaca_asset_api_inconsistency.md` for full context.

**Why bundle:** AL2023 deploy requires active Lambda monitoring anyway. Small, isolated code change in `symbol_metadata_service.py` is low-risk to add during the same validation window. Confirms parallelized bulk fetch still works under new runtime.

### Remove `/tmp` download workaround in `query_parquet`

Current `data_export.py::query_parquet` downloads the S3 parquet to `/tmp` before DuckDB queries it, because AL2's glibc 2.26 can't load DuckDB's httpfs extension. Post-AL2023 (glibc 2.34), native httpfs works.

**Fix:** replace the /tmp download block with native httpfs setup:
```python
conn.execute("INSTALL httpfs; LOAD httpfs;")
conn.execute(f"SET s3_region = '{region}';")
conn.execute("CREATE SECRET aws_creds (TYPE S3, PROVIDER CREDENTIAL_CHAIN);")
conn.execute(f"CREATE VIEW prices AS SELECT * FROM 's3://{S3_BUCKET}/prices/all_data.parquet';")
```

~15 min. Drops ~20 seconds off every `parquet_diagnose` call.

## Don't do this during

- Active TPE runs (computing on local pickle anyway, but avoid confusion)
- The 4:30 PM ET daily scan window (4:00-5:00 PM ET)
- The 6:30 PM ET hygiene window (6:15-7:00 PM ET)
- Right before weekends (validation during live trading days is easier)

Best windows: mid-morning (9-11 AM ET) or late evening (after hygiene completes, ~8 PM ET).
