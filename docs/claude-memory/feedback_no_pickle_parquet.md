---
name: feedback_no_pickle_parquet
description: "STOP saying 'pickle' — the live system reads PARQUET (PITFWU). Erik corrected this emphatically."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 264056a8-f1e5-489c-9140-1fb57bda9825
---

The live data read path is **PARQUET, not pickle.** `PRICE_SOURCE=parquet` + `PITFWU_READ=true` — scoped partial reads (top-universe + indices), per-symbol PITFWU store. Erik corrected this EMPHATICALLY (Aug 24, all-caps ×2: "WE ARE NOT USING PICKLE ANYMORE!!! PITFWU and the live system uses PARQUET!!!").

**Why:** CLAUDE.md still says the worker "loads full 2+ GB pickle" and the API Lambda "skips pickle loading" — that language is STALE (pre-parquet-flip). I parroted it and said "the API Lambda doesn't load the pickle." The parquet flip is LIVE (see [[project_oom_scan_zero_jun15]] + the PITFWU loop-closed note in MEMORY.md).

**How to apply:** Never reason/speak in terms of "the pickle." Say parquet / PITFWU. When reasoning about what the API Lambda has in memory, the architectural point still holds (API serves from S3 dashboard.json + DB; heavy market-data loads happen in the worker/scan) — but call the data source parquet. Better still: for DB-derivable work, read DB snapshots directly and stay price-source-agnostic.