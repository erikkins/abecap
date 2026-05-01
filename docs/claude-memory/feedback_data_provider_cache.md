---
name: Never re-fetch data from external providers
description: Any data pulled from Alpaca, yfinance, or future paid providers must be cached durably (S3 parquet) and re-read from cache forever after. Never re-fetch unless the source data has actually changed.
type: feedback
originSessionId: 1081317d-f863-470f-ab07-65ddc614e856
---
When fetching data from any external provider (Alpaca, yfinance, FinancialModelingPrep, any paid API), persist it to durable storage on first fetch. All subsequent runs must read from cache. Never re-pull data that's already been pulled.

**Why:** Erik's principle (Apr 30 2026) — paid data has cost, rate limits, and provider risk. We pay for Alpaca Pro. If we ever lose the relationship, we still have our historical record. Re-running a research project should be near-free in API calls. Re-fetching is a smell.

**How to apply:**

For any new research/validation project that pulls data:

1. **Before fetching**, check the cache. Cache key = (provider, symbol, date-range, granularity).
2. **If cache hit**, read from S3/parquet/local. No API call.
3. **If partial hit**, fetch only the missing range, append to cache.
4. **After fetching**, write to durable store. Default location: `s3://rigacap-prod-price-data-149218244179/<dataset>/...`
5. **Idempotent fetch logic** lives in a shared cache module; research scripts never call providers directly.

**What's already covered:**
- Daily bars (pickle + parquet shadow-write, append-only since Apr 1)
- Backup snapshots (weekly auto-archive)

**What needs the discipline applied if/when we fetch it:**
- Alpaca minute bars (for early-bird buy validation — when we run it)
- Any future intraday data (Polygon, FinancialModelingPrep, etc.)
- Corporate actions data (already cached via Apr 15 hygiene work)
- News data (if we ever pull historical)

**The "data may have changed" exception** — there are legitimate reasons to re-fetch:
- Split / dividend adjustment retroactively applied (we'd want to refresh adjusted prices)
- Vendor correction (rare; usually announced)
- Symbol corporate action (ticker change, merger) — invalidate cache for that symbol+date

These should be the only re-fetch triggers. Each should be logged with reason.

**Cost-cutting bonus:** for the intraday SELL validation, daily H/L (already in pickle) is sufficient — no Alpaca minute-bar fetch needed. Saved that way by accident; would have wasted a chunk of API calls otherwise.
