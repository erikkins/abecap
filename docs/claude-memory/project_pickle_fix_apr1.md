---
name: Pickle fix deployed Apr 1 2026
description: Pickle export was silently failing since Mar 28 — fixed and deployed, need to verify at 4:45 PM EDT
type: project
---

**Pickle export fix deployed Apr 1 2026 (commit e1ccb8b)**

**What broke:** `fetch_incremental` stripped indicator columns (dwap, ma_50, etc.) when appending new rows, shrinking pickle from 256→119 MB. Size guardrail correctly blocked it, but pipeline log reported "ok" anyway. Pickle on S3 stuck at Mar 28.

**Fix:** Keep `existing_df` columns intact during concat. Also fixed pipeline log to check `export_result.success`.

**Why:** The guardrail was right — the old code was actually destroying pre-computed indicators needed for WF sims and cold starts.

**How to apply:** Check at ~4:45 PM EDT Apr 1 that:
1. Pipeline log shows Pickle Export "ok" (not "warning")
2. S3 pickle `prices/all_data.pkl.gz` has a fresh timestamp
3. Pickle size is ~256 MB (not 119 MB)
4. Daily emails go out at 6 PM normally

**Verify command:**
```bash
aws s3api head-object --bucket rigacap-prod-price-data-149218244179 --key prices/all_data.pkl.gz --profile rigacap --region us-east-1
```
