-- tier_fills — per-tier transaction/fill log (STRs) for Core / Preserver / Maximizer.
-- Phase 2.5 plumbing. ADDITIVE — creates ONE new table; touches nothing existing and does
-- NOT alter the live t30v / model-portfolio path. Run OFF-HOURS (not near the 4 PM ET scan)
-- via the worker `{"run_migration": true}` path, verify columns, THEN deploy the SQLAlchemy
-- TierFill model + the fill-emit logic in a second commit (migration-first per CLAUDE.md).
-- Idempotent (IF NOT EXISTS) so re-running is safe.

CREATE TABLE IF NOT EXISTS tier_fills (
    id             SERIAL PRIMARY KEY,
    tier           VARCHAR(12) NOT NULL,          -- core | preserver | maximizer
    fill_date      DATE        NOT NULL,
    symbol         VARCHAR(10) NOT NULL,
    side           VARCHAR(4)  NOT NULL,          -- buy | sell
    shares         DOUBLE PRECISION NOT NULL,
    price          DOUBLE PRECISION NOT NULL,     -- fill price (today's close in the book)
    gross          DOUBLE PRECISION,              -- shares * price
    cost           DOUBLE PRECISION,              -- transaction cost applied to this fill
    source         VARCHAR(20),                   -- t30v | breakout | pullback_ma | oversold_bounce
    regime         VARCHAR(20),                   -- 7-regime label that day
    reason         VARCHAR(24),                   -- entry | hold_exit | stop | regime_exit | rebalance
    days_held      INTEGER,                       -- sells only: how long the lot was held
    realized_pnl   DOUBLE PRECISION,              -- sells only: realized $ P&L for the lot
    vol_scale      DOUBLE PRECISION,              -- maximizer breakout entries: vol-brake factor applied (else NULL/1.0)
    created_at     TIMESTAMP   NOT NULL DEFAULT NOW()
);

-- One row per tier/day/symbol/side (upsert-friendly; a book takes at most one buy and one
-- sell of a given name on a given day).
CREATE UNIQUE INDEX IF NOT EXISTS uq_tier_fills_tier_date_symbol_side
    ON tier_fills (tier, fill_date, symbol, side);
CREATE INDEX IF NOT EXISTS ix_tier_fills_tier_date ON tier_fills (tier, fill_date);
CREATE INDEX IF NOT EXISTS ix_tier_fills_symbol    ON tier_fills (symbol);
CREATE INDEX IF NOT EXISTS ix_tier_fills_tier      ON tier_fills (tier);
