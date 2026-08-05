# Preserver Productionization — Shadow Wiring + Migration Design (FOR REVIEW)

*Phase 2 of the 2-tier product. The prod port is done and proven faithful (detectors =
signal-exact vs research; `preserver_portfolio.replay_sleeve` = penny-exact vs
`shapes_portfolio.simulate`). This doc specifies the **first steps that touch live infra**
— a new table, a DB migration, and a shadow hook in the daily scan — for sign-off before
anything is applied. Nothing here is live yet.*

## 0. Safety principles (from CLAUDE.md)
- **Never touch the t30v path.** Preserver is a *parallel* table + builder; the live
  `ensemble_signals` / dashboard / `process_entries` flow is untouched.
- **Migration-first.** Create the tables via migration SQL, verify columns exist, *then*
  deploy the SQLAlchemy models + wiring in a second commit. Never model+migration together.
- **Shadow before serve.** The daily-scan hook only *records*; nothing is served to any user
  until a shadow period validates the live equity lands in the research range.
- **Fully isolated hook.** The shadow step is wrapped so its failure can *never* abort or
  alter the live daily scan (try/except, logged, non-fatal).
- **Off-hours** for the migration; not during the 4 PM ET scan window.

## 1. The book-transition rule (design decision — RECOMMENDED: hold-to-exit + layer)
When the regime flips (e.g., rotating_bull → weak_bear), the active source changes
(t30v → oversold). What happens to held positions?

- **Option A — hard rotate:** liquidate the old book, buy the new. Clean, but churns the
  whole book on every regime flip → high turnover, transaction costs, tax events.
- **✅ Option B — hold-to-exit + layer (RECOMMENDED):** keep existing positions until their
  natural exit (per-sleeve `hold` / t30v's own exits); *new* entries come only from the
  current regime's book. The book rotates gradually as old positions expire and
  new-regime names enter.

**Why B:** realistic, low-turnover, tax-friendlier, and it *smoothly approximates* the
research routing — during a flip to capitulation you hold t30v names until they expire while
layering in the (rare) oversold names, ending mostly in the regime-appropriate book.

**Honest caveat (important):** the research validated the Preserver via **return-stream
routing** (three books run continuously; realize the active one's daily return). A *real
single-capital-pool* Preserver with hold-to-exit is a **new construction** — so its equity
will be *close to* but **not penny-identical** to the research allocator curve. The
penny-exact proof covers the *sleeve mechanics*; the full single-pool Preserver's job in
shadow is to land in the **validated range** (≈19% / 1.33 / −13.5% daily 2021–26), not to
match to the cent. If we ever want an exact-match construction, that's Option A + always-on
parallel books — heavier, and not recommended.

## 2. Storage schema (new tables — additive, migration-first)

**`preserver_signals`** — the daily routed BUY candidates (mirrors `ensemble_signals`
shape, plus source/regime):

| column | type | notes |
|---|---|---|
| id | PK | |
| signal_date | Date, idx | |
| symbol | String(10), idx | |
| price | Float | |
| source | String(20) | `t30v` / `pullback_ma` / `oversold_bounce` |
| regime | String(20) | the 7-regime label that day |
| dollar_volume | Float | selection key |
| hold_days | Int | sleeve hold (informational for the book) |
| status | String(20) default 'active', idx | active/invalidated |
| created_at | DateTime | |
| — unique `(signal_date, symbol)` | | |

**`preserver_book_snapshots`** — daily snapshot of the shadow held book + equity (lets us
track live Preserver equity vs the research range over the shadow period):

| column | type | notes |
|---|---|---|
| id | PK | |
| snapshot_date | Date, unique idx | |
| regime | String(20) | |
| active_source | String(20) | which book drove entries today |
| equity | Float | mark-to-market book value |
| positions_json | JSON | [{symbol, source, shares, entry_price, exit_date}] |
| created_at | DateTime | |

Migration = two `CREATE TABLE` statements (idempotent `IF NOT EXISTS`), runnable via the
existing `{"run_migration": true}` worker path.

## 3. Daily-scan wiring point (shadow, isolated)
In `backend/main.py` `_run_daily_scan`, **after** `compute_shared_dashboard_data` returns
(regime + t30v `buy_signals` + `scanner_service.data_cache` all ready), add an isolated block:

```python
# --- SHADOW: Preserver tier (additive, NOT served; must never break the live scan) ---
try:
    regime = data['regime_forecast']['current_regime']
    preserver_service.run_shadow_day(
        db, today_et, regime,
        t30v_signals=data['buy_signals'],
        data_cache=scanner_service.data_cache,
    )
except Exception as e:
    logger.warning(f"[PRESERVER-SHADOW] non-fatal: {e}")  # never re-raise
```

`run_shadow_day` (new `preserver_service.py`): `route(regime)` → build today's entry
candidates (`build_daily_signals`) → advance the held book one day (exits by hold + fill
free slots from candidates under rule B) → persist `preserver_signals` + a
`preserver_book_snapshots` row. Reuses the already-validated `preserver_sleeves` /
`preserver_portfolio` logic; the t30v book source in rotating regimes reuses the live
`buy_signals` / existing model-portfolio positions (no recompute).

## 4. Rollout sequence
1. **Migration** (off-hours): create the two tables via `run_migration`; verify columns.
2. **Deploy models + `preserver_service` + the isolated hook** (shadow only).
3. **Shadow period (~2–4 weeks):** daily snapshots accumulate; confirm the live Preserver
   equity/return/DD tracks the research range (≈19% / 1.33 / −13.5% recent). Spot-check that
   rotating-bull days == t30v book, and sleeve days enter the expected names.
4. Only then: tier field on users (migration-first) → tier-aware serving → public 2-tier launch.

## 5. Open items / risks
- **Book-transition = Option B** pending your ✅.
- t30v-book-in-rotating: cleanest is to *reference the live t30v model-portfolio positions*
  rather than re-simulate — confirm we can read them read-only in the shadow.
- Shadow equity won't penny-match research (single-pool vs return-stream) — success = lands
  in the validated *range*, not exact.
- Regime label source: use the same `data['regime_forecast']['current_regime']` the live
  scan already computes (hysteresis-stable), so shadow and any future serve agree.

---

# Phase 2.5 — Plumbing Build (scope; decisions locked Jul 20 2026)

*Goal: promote the shadow into three real, comparable books (Core / Preserver / Maximizer)
each with its own fill/transaction log, an admin side-by-side compare view, and tier-aware
serving so the Maximizer add-on can actually be sold. Shadow ran Jul 8–17; findings below
drive this scope.*

## A. What the shadow proved (verified Jul 20)
- **Core** = the live model portfolio (`model_positions` / `model_portfolio_snapshots` /
  `model_portfolio_state`) with real entries/exits. Ground truth; already has real fills.
- **Preserver book is pinned at exactly $100k** every shadow day. Root cause: the tier books
  only manage the *defensive-sleeve* overlay and **deliberately exclude the t30v leg** ("added
  separately in prod from the live model portfolio" — never built). In rotating_bull the route
  is `t30v`, which is not a Preserver sleeve source → zero entries → flat equity.
- **Maximizer book moves** (100000 → 99172 over Jul 8–17) because its rotating_bull route is
  `breakout`, a real sleeve the shadow tracks. Only **6 signals** because the breakout detector
  is selective (close above prior 50-day high on a volume spike).

## B. Corrected per-tier routing (the load-bearing table)
| Regime | Preserver | Maximizer |
|---|---|---|
| calm_bull (strong/weak bull) | pullback_ma | pullback_ma *(same)* |
| capitulation (panic/recovery/weak_bear) | oversold_bounce | oversold_bounce *(same)* |
| **rotating_bull** | **t30v (Core)** | **breakout (vol-scaled)** |
| range_bound / unknown | t30v (Core) | t30v (Core) |

Source: `preserver_sleeves.route` (:35), `maximizer_sleeves.route` (:47). Maximizer = Preserver
routing with ONE swap: rotating_bull → breakout.

## C. Locked decisions
1. **t30v leg = MIRROR the live Core book** (read-only reference to real model-portfolio
   positions), not an independent re-simulation → one authoritative Core; tiers diverge only
   where their sleeves/brake engage. Applies **fully to Preserver** (rotating_bull + range_bound)
   and **only to range_bound for Maximizer** (rotating_bull is its own breakout book).
2. **Start basis = align to Core inception (~Jun 14) + backfill** Preserver/Maximizer so all
   three share one $100k origin. Discards the misaligned Jul-8 shadow start.

## D. Workstreams
- **WS1 — Real tier books.** Fold the mirrored Core t30v leg into Preserver/Maximizer equity +
  positions during t30v-routed regimes (per B). Preserver = the big lift (leg entirely missing);
  Maximizer already works in rotating_bull, needs the Core mirror for range_bound only. Backfill
  from Core inception.
- **WS2 — Per-tier fill log (STRs).** New `tier_fills` table; emit a row on every entry/exit in
  `advance_day`. Preserver/Maximizer tier_fills already include the mirrored-Core leg fills;
  Core's own fills are surfaced read-only from the existing model-portfolio trade records
  (UNION in the admin view — do NOT write into the live t30v path).
- **WS3 — Tier-aware serving.** Derive tier at serve-time from the Stripe subscription items
  (base only → Preserver; base + Maximizer add-on → Maximizer; Core = internal/admin only) —
  reuses `billing._maxpp_price_ids()`, **no new user column needed**. Dashboard/signals endpoints
  branch on tier. Behind a flag. THIS is the gate that makes Maximizer sellable.
- **WS4 — Admin 3-book compare view.** Core vs Preserver vs Maximizer side-by-side (equity
  curves, current positions, metrics) + the per-tier fills log. Read-only; safe to ship early.
- **WS5 — Subscriber UI per-tier behavior.** The tiers *look* different on-screen, not just a
  different list: Preserver shows defensive-sleeve rotations (pullback/oversold) with their own
  holds; Maximizer shows breakout entries + vol-brake sizing and more turnover; in rotating_bull
  Preserver == Core (UI must be honest, not imply extra activity). Exit params displayed per tier
  (ties to [[feedback_wf_prod_parity]] — displayed exit MUST equal the tier's live rule).
  Preserver users get a Maximizer upsell nudge.

## E. Build sequence (migration-first / off-hours per §0)
1. **Migration** (first step): create `tier_fills` (additive, idempotent) via
   `{"run_migration": true}`; verify. Off-hours, NOT near the 4 PM ET scan. See
   `backend/migrations/tier_fills.sql`.
2. Deploy the `TierFill` model + fill-emit logic in WS1/WS2 (still records-only; serving
   untouched). Backfill from Core inception.
3. **Validate** the three curves: rotating_bull → Preserver ≈ Core; Maximizer runs breakout;
   sleeve regimes enter the expected names; fills reconcile against snapshots.
4. **WS4 admin compare view** (read-only) — eyeball the three books before any subscriber change.
5. **WS3 tier-aware serving** LAST, behind a flag → then wire the LandingV2 Maximizer CTA (WS5)
   so the add-on is sellable + rendered correctly.

## F. Open items
- Confirm Core's live inception date + starting capital for the backfill anchor (memory says
  re-dated ~Jun 14 @ $100k — verify against `model_portfolio_state`).
- Backfill needs historical daily prices for the sleeve/breakout marks back to inception —
  reuse the parquet/PITFWU read path (same data the scan uses).
- Existing paid subs are all on the base plan → Preserver tier; Core is never sold.
