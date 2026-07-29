"""Tier-aware serving (WS3) — ADDITIVE, reversible, live-Core-path untouched.

Serving model (Option-B, locked Jul 24 2026):
  - EVERYONE is served the PRESERVER base: the live t30v buy_signals (same names as Core)
    plus a capitulation exposure note. Nobody is served raw "Core" labeling — Core stays
    the internal model book.
  - MAXIMIZER-entitled users (subscription.has_maxpp_addon [paid] OR subscription.compmax
    [admin comp]) see the BREAKOUT BOOK in rotating_bull INSTEAD of the t30v list: the held
    breakout positions (day X/29 hold countdown) + today's fresh breakout buys. In every
    other regime Maximizer == Preserver (that's Option B — "really is Preserver aside from
    rotating bull"), so they get the Preserver base too.

Gated by the TIER_SERVING env flag (default OFF) so rollout is reversible; when off, callers
serve the legacy Core payload unchanged. This module never mutates the user's own positions
panel or the S3 dashboard cache — it only reshapes the served buy_signals list + adds tier
metadata. Breakout book/signals come from the shadow tables already populated daily behind
MAXIMIZER_SHADOW (maximizer_book_snapshots, maximizer_signals).
"""
from __future__ import annotations

import os
from datetime import date as _date, timedelta as _timedelta
from typing import Dict, List, Optional

from sqlalchemy import select

# Regimes where the Preserver overlay raises cash (exposure -> ~25%). Mirrors
# preserver_service CAP / maximizer_blend_certify CAPIT.
CAPITULATION = {"panic_crash", "recovery", "weak_bear"}
ROTATING = "rotating_bull"
BREAKOUT_HOLD = 29  # trading days (maximizer_sleeves.BREAKOUT["hold"])
CAP0 = 100_000.0    # book inception capital (matches maximizer/preserver book CAP0)


def tier_serving_enabled() -> bool:
    """Global kill-switch. Off => callers serve the legacy Core payload unchanged."""
    return os.getenv("TIER_SERVING", "").lower() in ("1", "true", "yes")


def resolve_tier(subscription, is_admin: bool = False, preview_tier: Optional[str] = None) -> str:
    """'preserver' | 'maximizer'. Entitlement = paid add-on OR admin comp. Admins may
    force a view with ?preview_tier= for dark-launch verification."""
    if preview_tier in ("preserver", "maximizer"):
        return preview_tier
    if subscription is not None and (
        getattr(subscription, "has_maxpp_addon", False) or getattr(subscription, "compmax", False)
    ):
        return "maximizer"
    return "preserver"


def _approx_exit_date(days_left: int) -> str:
    """Approx calendar exit date from trading days remaining (~5 trading days / 7 calendar)."""
    return (_date.today() + _timedelta(days=int(round(max(0, days_left) * 7 / 5)))).isoformat()


def _current_price(sym: str, data_cache: dict, fallback: float) -> float:
    df = data_cache.get(sym)
    try:
        if df is not None and len(df):
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    return fallback


async def _load_prices(symbols: List[str], data_cache: dict) -> None:
    """Lazily hydrate data_cache for symbols the API Lambda hasn't loaded (mirrors
    _get_positions_with_guidance) so held-book marks aren't stuck at entry price."""
    missing = [s for s in symbols if s and s not in data_cache]
    if not missing:
        return
    try:
        from app.services.data_export import data_export_service
        data_cache.update(data_export_service.import_symbols(missing))
    except Exception as e:  # non-fatal: cards fall back to entry price
        print(f"⚠️ tier_serving: price hydrate failed: {e}")


async def build_maximizer_breakout_view(db, data_cache: dict) -> List[dict]:
    """The breakout BOOK as buy_signal-shaped cards: HELD positions (day X/29 countdown)
    first, then today's fresh breakout buys. Reads the shadow tables only."""
    from app.core.database import MaximizerBookSnapshot, MaximizerSignal

    # Latest book snapshot -> held breakout positions.
    snap = (await db.execute(
        select(MaximizerBookSnapshot).order_by(MaximizerBookSnapshot.snapshot_date.desc()).limit(1)
    )).scalars().first()

    held: List[dict] = []
    held_syms = set()
    if snap and isinstance(snap.positions_json, dict):
        held_syms = {p.get("symbol") for p in snap.positions_json.get("positions", []) if p.get("symbol")}
        await _load_prices(list(held_syms), data_cache)
        for p in snap.positions_json.get("positions", []):
            sym = p.get("symbol")
            if not sym:
                continue
            entry = float(p.get("entry") or 0) or 0.0
            days_held = int(p.get("days_held") or 0)
            hold = int(p.get("hold") or BREAKOUT_HOLD)
            days_left = max(0, hold - days_held)
            cur = _current_price(sym, data_cache, entry)
            pnl_pct = ((cur / entry - 1) * 100) if entry else 0.0
            held.append({
                "symbol": sym,
                "price": round(cur, 2),
                "entry_price": round(entry, 2),
                "source": "breakout",
                "exit_rule": "hold",
                "hold_days": hold,
                "days_held": days_held,
                "days_left": days_left,
                "exit_date_approx": _approx_exit_date(days_left),
                "pnl_pct": round(pnl_pct, 1),
                "status": "holding",
                "is_fresh": False,
                "in_user_position": False,
                # display-compat defaults for the buy-card renderer
                "ensemble_score": 0,
                "signal_strength_label": "Holding",
            })
        held.sort(key=lambda c: c["days_left"])  # nearest to exit first

    # Today's fresh breakout buys (latest signal_date, source=breakout, active) that aren't
    # already held.
    latest = (await db.execute(
        select(MaximizerSignal.signal_date)
        .where(MaximizerSignal.source == "breakout")
        .order_by(MaximizerSignal.signal_date.desc()).limit(1)
    )).scalar_one_or_none()
    fresh: List[dict] = []
    if latest is not None:
        rows = (await db.execute(
            select(MaximizerSignal).where(
                MaximizerSignal.signal_date == latest,
                MaximizerSignal.source == "breakout",
                MaximizerSignal.status == "active",
            )
        )).scalars().all()
        for r in rows:
            if r.symbol in held_syms:
                continue
            hold = int(r.hold_days or BREAKOUT_HOLD)
            fresh.append({
                "symbol": r.symbol,
                "price": round(float(r.price or 0), 2),
                "source": "breakout",
                "exit_rule": "hold",
                "hold_days": hold,
                "days_held": 0,
                "days_left": hold,
                "exit_date_approx": _approx_exit_date(hold),
                "dollar_volume": float(r.dollar_volume or 0),
                "status": "new",
                "is_fresh": True,          # act today — breakout signal is a same-day cross
                "in_user_position": False,
                "ensemble_score": 0,
                "signal_strength_label": "New breakout",
            })
        fresh.sort(key=lambda c: -c.get("dollar_volume", 0))

    return fresh + held  # new buys first, then the held book


async def build_maximizer_missed(db, limit: int = 5) -> List[dict]:
    """Maximizer missed-opps = profitable CLOSED breakout trades from the book's real fill log
    (tier_fills), i.e. breakout winners a subscriber could have mirrored. Real trades, not a
    re-backtest — so the numbers are penny-honest. Empty until breakout exits accumulate."""
    from app.core.database import TierFill
    rows = (await db.execute(
        select(TierFill).where(
            TierFill.tier == "maximizer", TierFill.side == "sell",
            TierFill.reason == "hold_exit", TierFill.realized_pnl > 0,
        ).order_by(TierFill.fill_date.desc()).limit(limit)
    )).scalars().all()
    out = []
    for r in rows:
        entry_px = (r.price - (r.realized_pnl / r.shares)) if r.shares else r.price
        ret_pct = ((r.price / entry_px - 1) * 100) if entry_px else 0.0
        out.append({
            "symbol": r.symbol,
            "entry_price": round(entry_px, 2),
            "sell_price": round(r.price, 2),
            "sell_date": r.fill_date.isoformat() if r.fill_date else None,
            "would_be_return": round(ret_pct, 1),
            "would_be_pnl": round(r.realized_pnl, 2),
            "days_held": r.days_held,
            "source": "breakout",
        })
    return out


async def build_tier_book(db, tier: str, capital: float, data_cache: dict,
                          trailing_stop_pct: float = 30.0) -> Optional[dict]:
    """Capital-scaled MIRROR of a tier's model book. Scales the book's positions to the user's
    capital (implied_shares = book_shares × capital/book_value) so their portfolio auto-mirrors
    the book with zero per-trade entry. Maximizer = breakout book (day-X/29 exits); Preserver =
    live t30v model book (30% trailing). Returns None if the book has no data."""
    from app.core.database import MaximizerBookSnapshot, ModelPosition, ModelPortfolioSnapshot

    capital = float(capital or 100000.0)
    holdings: List[dict] = []
    invested = 0.0
    book_cash = 0.0
    as_of = None
    regime = None
    book_equity = None
    preserver_exposure = None  # set in the preserver branch; drives the cash-raise overlay

    if tier == "maximizer":
        snap = (await db.execute(
            select(MaximizerBookSnapshot).order_by(MaximizerBookSnapshot.snapshot_date.desc()).limit(1)
        )).scalars().first()
        if not snap or not isinstance(snap.positions_json, dict):
            return None
        pj = snap.positions_json
        positions = pj.get("positions", []) or []
        book_cash = float(pj.get("bk_cash") or 0.0)
        as_of, regime = snap.snapshot_date, snap.regime
        book_equity = float(snap.equity) if snap.equity else None
        await _load_prices([p.get("symbol") for p in positions if p.get("symbol")], data_cache)
        for p in positions:
            sym = p.get("symbol")
            if not sym:
                continue
            entry = float(p.get("entry") or 0) or 0.0
            cur = _current_price(sym, data_cache, entry)
            shares = float(p.get("shares") or 0)
            val = shares * cur
            invested += val
            days_held = int(p.get("days_held") or 0)
            hold = int(p.get("hold") or BREAKOUT_HOLD)
            days_left = max(0, hold - days_held)
            holdings.append({
                "symbol": sym, "shares": shares, "price": round(cur, 2),
                "entry_price": round(entry, 2), "value": val, "source": "breakout",
                "exit_rule": "hold", "days_held": days_held, "hold_days": hold,
                "days_left": days_left, "exit_date_approx": _approx_exit_date(days_left),
                "pnl_pct": round((cur / entry - 1) * 100, 1) if entry else 0.0,
            })
    else:  # preserver (and any non-maximizer) = live t30v model book
        rows = (await db.execute(
            select(ModelPosition).where(
                ModelPosition.portfolio_type == "live", ModelPosition.status == "open")
        )).scalars().all()
        snap = (await db.execute(
            select(ModelPortfolioSnapshot).where(ModelPortfolioSnapshot.portfolio_type == "live")
            .order_by(ModelPortfolioSnapshot.snapshot_date.desc()).limit(1)
        )).scalars().first()
        if not rows and not snap:
            return None
        book_cash = float(snap.cash) if snap and snap.cash is not None else 0.0
        as_of = snap.snapshot_date if snap else None
        book_equity = float(snap.total_value) if snap and snap.total_value else None
        # Preserver is the t30v names at the book's REGIME EXPOSURE (cash-raise in capitulation).
        # Read the Preserver book's exposure (1.0 normally, 0.25 in capitulation) + its own
        # exposure-adjusted equity — so the book view is genuinely Preserver, not raw Core.
        from app.core.database import PreserverBookSnapshot
        _psnap = (await db.execute(
            select(PreserverBookSnapshot).order_by(PreserverBookSnapshot.snapshot_date.desc()).limit(1)
        )).scalars().first()
        if _psnap and isinstance(_psnap.positions_json, dict):
            try:
                preserver_exposure = float(_psnap.positions_json.get("exposure", 1.0))
            except (TypeError, ValueError):
                preserver_exposure = 1.0
            if _psnap.equity:
                book_equity = float(_psnap.equity)
            if _psnap.snapshot_date:
                as_of = _psnap.snapshot_date
        await _load_prices([r.symbol for r in rows], data_cache)
        for r in rows:
            entry = float(r.entry_price or 0) or 0.0
            cur = _current_price(r.symbol, data_cache, entry)
            val = float(r.shares or 0) * cur
            invested += val
            hwm = max(entry, float(r.highest_price or 0) or entry, cur)
            stop = hwm * (1 - trailing_stop_pct / 100.0)
            holdings.append({
                "symbol": r.symbol, "shares": float(r.shares or 0), "price": round(cur, 2),
                "entry_price": round(entry, 2), "value": val, "source": "preserver",
                "exit_rule": "trailing", "trailing_stop_pct": trailing_stop_pct,
                "trailing_stop_level": round(stop, 2),
                "pnl_pct": round((cur / entry - 1) * 100, 1) if entry else 0.0,
            })

    book_value = invested + book_cash
    if book_value <= 0:
        return None

    if preserver_exposure is not None:
        # PRESERVER: hold `exposure` of capital across the t30v names (proportional to their
        # book weights), the rest in cash. exposure=1.0 in normal regimes (== fully mirrored
        # Core); exposure=0.25 in capitulation (75% cash raised). This is what makes the
        # served book genuinely Preserver rather than raw t30v.
        exp = max(0.0, min(1.0, preserver_exposure))
        invested_cap = capital * exp
        cash_cap = capital * (1 - exp)
        for h in holdings:
            w = (h["value"] / invested) if invested else 0.0   # name's weight within the book
            h["implied_value"] = round(invested_cap * w, 2)
            h["implied_shares"] = round((invested_cap * w) / h["price"], 2) if h["price"] else 0.0
            h["weight_pct"] = round(w * exp * 100, 1)           # weight of TOTAL capital
        holdings.sort(key=lambda h: -h["implied_value"])
        return {
            "tier": tier,
            "capital": round(capital, 2),
            "holdings": holdings,
            "invested_value": round(invested_cap, 2),
            "cash_value": round(cash_cap, 2),
            "cash_pct": round((1 - exp) * 100, 1),
            "invested_pct": round(exp * 100, 1),
            "exposure": round(exp, 2),
            "as_of": as_of.isoformat() if hasattr(as_of, "isoformat") else (str(as_of) if as_of else None),
            "regime": regime,
            "book_return_pct": round((book_equity / CAP0 - 1) * 100, 1) if book_equity else None,
        }

    scale = capital / book_value
    for h in holdings:
        h["implied_shares"] = round(h["shares"] * scale, 2)
        h["implied_value"] = round(h["value"] * scale, 2)
        h["weight_pct"] = round(h["value"] / book_value * 100, 1)
    holdings.sort(key=lambda h: -h["implied_value"])
    return {
        "tier": tier,
        "capital": round(capital, 2),
        "holdings": holdings,
        "invested_value": round(invested * scale, 2),
        "cash_value": round(book_cash * scale, 2),
        "cash_pct": round(book_cash / book_value * 100, 1),
        "invested_pct": round(invested / book_value * 100, 1),
        "as_of": as_of.isoformat() if hasattr(as_of, "isoformat") else (str(as_of) if as_of else None),
        "regime": regime,
        # Since-inception return of the model book (live shadow window — label as such in UI).
        "book_return_pct": round((book_equity / CAP0 - 1) * 100, 1) if book_equity else None,
    }


async def apply_tier_serving(
    db, cached: dict, tier: str, data_cache: dict, buy_signals: List[dict],
) -> dict:
    """Return the tier overlay to merge into the dashboard payload:
      {buy_signals, tier, signal_source, exit_rule, tier_note, missed_opportunities?}.
    Preserver: passthrough t30v names + capitulation exposure note.
    Maximizer + rotating_bull: swap to the breakout book. Else: Preserver base.
    Maximizer always gets breakout-based missed-opps (real closed breakout winners).
    Caller only invokes this when tier_serving_enabled().
    """
    regime = (cached.get("regime_forecast") or {}).get("current_regime") or ""

    # Stamp the Preserver base list (used by Preserver, and by Maximizer when out of rotating).
    preserver_signals = []
    for s in buy_signals:
        card = dict(s)
        card["source"] = "preserver"
        card["exit_rule"] = "trailing"
        preserver_signals.append(card)

    if tier == "maximizer":
        missed = await build_maximizer_missed(db)
        # Prefer the AI-generated daily briefing cached in the latest book snapshot (worker
        # writes it, anti-repetition); fall back to the templated strings below if absent.
        ai_brief = None
        try:
            from app.core.database import MaximizerBookSnapshot
            _bsnap = (await db.execute(
                select(MaximizerBookSnapshot).order_by(MaximizerBookSnapshot.snapshot_date.desc()).limit(1)
            )).scalars().first()
            if _bsnap and isinstance(_bsnap.positions_json, dict):
                ai_brief = _bsnap.positions_json.get("briefing")
        except Exception:
            ai_brief = None
        if regime == ROTATING:
            breakout = await build_maximizer_breakout_view(db, data_cache)
            # Faithful only if the shadow book has data; otherwise fall back to the Preserver
            # base rather than showing an empty screen.
            if breakout:
                held = sum(1 for c in breakout if c.get("status") == "holding")
                fresh = sum(1 for c in breakout if c.get("status") == "new")
                return {
                    "buy_signals": breakout,
                    "tier": "maximizer",
                    "signal_source": "breakout",
                    "exit_rule": "hold",
                    "missed_opportunities": missed,
                    "market_context": ai_brief or (
                        f"Rotating-bull momentum is broad — your Maximizer book is riding "
                        f"{held} breakout name{'s' if held != 1 else ''} into their hold windows"
                        f"{f' and added {fresh} today' if fresh else ''}. Aggressive by design; "
                        f"each name sells on time at day 29, no trailing stop. Expect chop."
                    ),
                    "tier_note": (
                        "Rotating-bull regime: your Maximizer book is hunting breakouts. Each "
                        "name is a same-day entry held ~29 trading days, then sold on time (no "
                        "trailing stop). Held names show their day X/29 countdown."
                    ),
                }
        # Maximizer OUTSIDE rotating_bull: breakout hunting is off. Serve the Preserver base for
        # new buys; existing breakout positions wind down on their own 29-day clocks (positions
        # panel). No wholesale swap.
        return {
            "buy_signals": preserver_signals,
            "tier": "maximizer",
            "signal_source": "preserver",
            "exit_rule": "trailing",
            "missed_opportunities": missed,
            "market_context": ai_brief or (
                "Out of rotating-bull — the Maximizer book has paused breakout hunting and is "
                "in Preserver mode. Any breakout names you hold wind down on their exit dates; "
                "new buys follow the Preserver book (30% trailing)."
            ),
            "tier_note": (
                "Not a rotating-bull regime: breakout hunting is paused. Hold your existing "
                "breakout names to their day-29 exits (see your positions); new buys follow "
                "Preserver until momentum broadens again."
            ),
        }
    note = None
    if regime in CAPITULATION:
        note = (
            "Capitulation regime: the Preserver overlay is defensive — raise cash toward "
            "~25% exposure and let the trailing stops do their job. New buys are paused "
            "until momentum turns."
        )
    out = {
        "buy_signals": preserver_signals,
        "tier": tier,
        "signal_source": "preserver",
        "exit_rule": "trailing",
        "tier_note": note,
    }
    if tier == "maximizer":
        # A Maximizer user outside rotating_bull still gets breakout-based missed-opps (the
        # breakout winners accumulate across rotating spells, worth surfacing any regime).
        out["missed_opportunities"] = await build_maximizer_missed(db)
    else:
        # Preserver users see the breakout winners as an UPSELL ("what Maximizer caught")
        # — a separate block, their own trailing-stop missed-opps stay intact.
        out["upsell_missed"] = await build_maximizer_missed(db)
    return out
