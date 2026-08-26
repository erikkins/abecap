# Portfolio Overlay — Tier 1 (public "where do your stocks sit") — Build Spec

**Goal:** A frictionless, no-signup landing hook. User pastes tickers → we show an **aggregate count** of how their names intersect RigaCap's universe + historical signals. Softens the first click (the SIS funnel leak), and captures the tickers as a lead + demand signal. Compliance: **aggregate/impersonal only** at Tier 1 (see `compliance-portfolio-overlay-brief.md`); public display gated on counsel sign-off.

**Golden rule:** NEVER throw away data — every submission is saved (even anonymous, even 0 matches).

---

## Build order (deploy is down; sequence when Actions is back)
1. **[compliance-neutral, build first] Migration + save path.** Create `portfolio_submissions`; add the endpoint that normalizes → computes counts → **saves** → returns counts. Existing/saving is neutral; just don't surface the widget yet.
2. **[after counsel] Mount the landing widget** (Tier 1 display) behind a feature flag.
3. **[Tier 2, later]** auth-gated per-ticker view + retrospective "what our rules did."

Follow the migration-first rule: run the `CREATE TABLE` via `run_migration` and verify **before** deploying the model/endpoint code.

---

## 1. Data model — `portfolio_submissions`
```sql
CREATE TABLE IF NOT EXISTS portfolio_submissions (
    id            BIGSERIAL PRIMARY KEY,
    session_id    VARCHAR(64),                 -- anonymous per-tab id from the frontend
    email         VARCHAR(255),                -- nullable; stitched if given later
    user_id       UUID REFERENCES users(id),   -- nullable; set if logged in
    tickers       JSONB NOT NULL,              -- normalized uppercase list as submitted
    ticker_count      INT NOT NULL,
    in_universe_count INT NOT NULL,
    signaled_count    INT NOT NULL,
    source        VARCHAR(32) NOT NULL DEFAULT 'landing_widget',  -- landing_widget|onboarding|app
    path          VARCHAR(255),
    utm_source    VARCHAR(128), utm_campaign VARCHAR(128), gclid VARCHAR(255),
    user_agent    VARCHAR(512),
    ip            VARCHAR(64),
    converted     BOOLEAN DEFAULT FALSE,        -- session later created an account
    created_at    TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ps_session ON portfolio_submissions(session_id);
CREATE INDEX IF NOT EXISTS idx_ps_email   ON portfolio_submissions(email);
CREATE INDEX IF NOT EXISTS idx_ps_created ON portfolio_submissions(created_at DESC);
```

## 2. Lookup sets (cached server-side; refresh daily)
- `UNIVERSE_SET` — current tracked universe (top-600) from `signals/universe-history/{latest}.json` (or `data_export`/scanner universe). ~600 symbols. **Primary hook** (always a strong number; most large/mid-caps hit).
- `SIGNALED_SET_5Y` — distinct symbols RigaCap **signaled a buy on over the trailing 5 years** (Erik: 5yr, not 21 — modern era). **Not in the DB** — the live signal tables only span ~4 months (ensemble 2026-04-13→, 111 syms; maximizer/preserver weeks). So it's a **precomputed artifact** (see §2a). Endpoint loads the artifact ∪ the live signal tables (for freshness). Cache in the API Lambda (module-level, TTL ~1h) so a submission is a set-intersection, not a DB scan.

### 2a. Precompute job — `build_signaled_symbols_5y` (worker)
Runs a **5-year ensemble backtest** over the current universe → collects the **distinct symbols the model ENTERED** → writes `signals/signaled_symbols_5y.json` to S3. Refresh **monthly** (cheap, and the set drifts slowly). Definition of "signaled" = *names the published model actually bought over the trailing 5-yr walk-forward* (~150–250 symbols) — factual/reproducible ("RigaCap traded this"), the most defensible framing. Option to broaden to "ever qualified as a candidate" (larger set) if we want more overlap — decide before widget copy.

## 3. Public endpoint — `POST /api/public/portfolio-check` (no auth)
Request:
```json
{ "tickers": ["AAPL","NVDA","PLTR"], "session_id": "abc123", "email": null,
  "source": "landing_widget", "path": "/should-i-sell",
  "utm_source": "...", "utm_campaign": "...", "gclid": "...", "turnstile_token": "..." }
```
Behavior:
1. Normalize: uppercase, trim, dedupe, cap **50**, validate `^[A-Z.\-]{1,6}$`; drop invalid.
2. `in_universe_count = |tickers ∩ UNIVERSE_SET|`; `signaled_count = |tickers ∩ EVER_SIGNALED_SET|`.
3. **SAVE** the row (always — anonymous ok, 0-match ok). Enroll email in newsletter pipeline if present.
4. Return **counts only** (Tier 1):
```json
{ "ticker_count": 3, "in_universe_count": 2, "signaled_count": 1 }
```
- **No per-ticker data at Tier 1** (compliance). Per-ticker is Tier 2 (auth-gated endpoint / unlocked field).
- Abuse: 50-ticker cap, IP rate-limit, optional Turnstile (reuse existing). PII = optional email + tickers only.

## 4. Landing widget (Tier 1 UI) — flag-gated until counsel clears
Placement: a band near the fold on `/should-i-sell`, `/momentum`, and `/` (the softer first click).
```
"See where your stocks sit in RigaCap."
[ AAPL, NVDA, PLTR, …            ] [ Check ]
→ "6 of your 10 are in our universe. 4 have triggered a RigaCap signal."
[ See which ones — and what our rules did → Start free (no card) ]
"Information only — not individualized advice."  (placeholder; counsel to finalize)
```
- Generate/reuse `session_id` in sessionStorage; POST; render counts.
- **0/0 graceful pivot:** "Your names are off our radar — here's what we *do* track →" (don't dead-end).
- Funnel events: `logPublicEvent('portfolio_check', {in_universe_count, signaled_count})` + `portfolio_check_cta`.
- On later signup in the same session → `converted=true` + stitch email to prior anonymous rows (by session_id).

## 5. Data → value (why we save everything)
- **Leads:** email + real holdings = high-intent; re-marketing hook ("2 of your stocks signaled this week →").
- **Demand intelligence:** admin view of most-submitted tickers → informs universe priorities, marketing, content.
- **Funnel analytics:** check→signup lift; convert rate by source/door.

## 6. Tier 2 (later, auth-gated) — for reference
Per-ticker: in-universe? / has-ever-signaled? + retrospective "what our rules did with these over our record." Impersonal + historical. Separate authed endpoint. Then evaluate SnapTrade read-only (Phase 2) to auto-pull holdings.

### Phase 2 — SnapTrade economics + compliance (Erik, Aug 26)
- **Dev mode: $0, ≤5 users, full RT + trading** → free PILOT to prove the read-only pull before committing.
- **Pay-as-you-go "Daily Data" tier: $1/user/mo + $0.05/sync, NO trading, not RT** → the production tier, and an exact fit (we're EOD/daily anyway; RT wasted; we don't want trading). Only billed for users who actually connect a brokerage (a subset).
- **Design principle — sync sparingly, overlay continuously.** Holdings change rarely; our signals are computed on OUR side (no sync to re-overlay). So sync holdings only on connect / login / weekly / a user-tapped "refresh my holdings," and re-overlay our daily signals at $0. → ~4 syncs/mo = **$1 + ~$0.20 ≈ $1.20/user/mo** (~1% of $129). A user-triggered refresh button puts the $0.05 in their hands, fired only when they've actually changed something.
- **Compliance win:** the no-trading tier makes the integration **structurally read-only** (can't place a trade by construction) — a stronger story for counsel than a policy promise. Relay to attorney: "the tier we'd use is data-read only, zero trading capability."

## Decisions (Erik, Aug 26)
- ✅ **Tiering:** count = **free** · email unlocks the **names** · paid = **full** signals.
- ✅ **Signaled window:** **5 years** (21 too far back). → precompute `SIGNALED_SET_5Y` from the 5-yr WF (§2a).
- ✅ **Counsel:** letter sent (`compliance-portfolio-overlay-brief.md`); public widget gated on their reply.
- ↔ **Open (minor, before widget copy):** define "signaled" = *entered* (~150-250, factual, rec) vs *ever-qualified-candidate* (bigger set). Default to *entered* unless we want a bigger overlap number. Disclaimer wording from counsel.
