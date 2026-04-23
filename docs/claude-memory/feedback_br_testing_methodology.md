---
name: BR testing methodology — always overlay on CB baseline params
description: When testing Bear Ripper impact, always use the exact CB-only precomputed params as the base and add BR on top. Never use generic fixed params.
type: feedback
originSessionId: 39ce1e26-1ab7-4fbd-8e9a-6c892d933b00
---
When comparing BR vs no-BR, always use the exact precomputed params from the CB-only baseline run (stored in `/tmp/cb_params_N_YYYY-MM-DD/`). Add BR fields to `active_params` and re-run.

**Why:** Using generic fixed params changes TWO variables (ensemble params + BR), making it impossible to isolate BR's contribution. The +204% baseline came from per-period TPE-optimized params — those are the "real" ensemble. BR should be tested as an overlay, not as part of a different param set.

**How to apply:** For any BR comparison:
1. Start with the CB-only precomputed params that produced the known baseline
2. Merge BR fields (`bear_ripper_enabled`, `max_positions`, `position_size_pct`, `trailing_stop_pct`) into `active_params`
3. Run with `--precomputed-params` pointing to the merged dir
4. Compare result directly against the known baseline number

In production: the optimizer re-tunes every 2 weeks. BR is just one more lever the optimizer can enable/disable per period.
