---
name: WF cache table size check — Oct 2026
description: Daily/nightly WF cache rows now append-only (no DELETE). Revisit Oct 2026 to confirm growth isn't problematic.
type: project
originSessionId: 1081317d-f863-470f-ab07-65ddc614e856
---
**Decision (Apr 28, 2026):** Refactored `_run_daily_walk_forward`, `_run_nightly_walk_forward`, and the `nightly_wf_job` Lambda handler to append-only. Removed all `DELETE FROM walk_forward_simulations WHERE is_*_cache=True` patterns.

**Why:** They kept hitting FK violations on the `walk_forward_period_results.simulation_id` FK (no ON DELETE CASCADE in schema). All readers already use `ORDER BY simulation_date DESC LIMIT 1`, so deleting was redundant defensive code that fought the schema.

**Why this memory exists:** Growth is now ~1 daily cache row + ~70 period_results children + ~1 nightly cache row + ~7 period_results children = ~80 rows/trading-day across both tables. ~250 trading days/year = ~20K rows/year. Trivial today, but worth a sanity check.

**How to apply:** When working in this project around **October 2026**, run:

```sql
SELECT
  (SELECT count(*) FROM walk_forward_simulations WHERE is_daily_cache = true) AS daily_cache_rows,
  (SELECT count(*) FROM walk_forward_simulations WHERE is_nightly_missed_opps = true) AS nightly_cache_rows,
  (SELECT count(*) FROM walk_forward_period_results
    WHERE simulation_id IN (SELECT id FROM walk_forward_simulations WHERE is_daily_cache OR is_nightly_missed_opps)) AS child_rows,
  pg_size_pretty(pg_total_relation_size('walk_forward_simulations')) AS sims_size,
  pg_size_pretty(pg_total_relation_size('walk_forward_period_results')) AS period_results_size;
```

If `child_rows > 50,000` or table sizes look uncomfortable: add a periodic prune that keeps the latest N (e.g., 90) cache rows per type and deletes children-then-parents for the rest. Don't preemptively add the prune — it may never be needed.
