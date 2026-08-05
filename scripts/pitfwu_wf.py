"""PITFWU walk-forward runner (v1) — the first survivorship-free, split-clean WF.

Composes the immutable raw layer (S3 parquet bars + corp-actions) through the
pitfwu_veneer into a point-in-time data_cache + survivorship-free universe, then
runs the production ensemble backtester unchanged (so strategy logic == prod).

v1 scope (documented, not bugs):
  - split adjustment is as-of window END (correct for returns; min-price leak
    dormant for top-100 windows). v2 = per-day re-derivation.
  - price-return (no dividends). ~0.5%/yr drag for dividend payers.
  - renames: rename stubs dropped (FB); a rename mid-hold closes at rename px
    (== continuation px, negligible P&L impact).

Usage: AWS_PROFILE=rigacap python scripts/pitfwu_wf.py 2022-01-03 2026-05-29
"""
import os
import sys
import pandas as pd
from datetime import datetime

R = "/Users/erikkins/CODE/stocker-app"
sys.path.insert(0, os.path.join(R, "backend"))
sys.path.insert(0, os.path.join(R, "scripts"))
for _line in open(os.path.join(R, ".env")):
    if _line.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _line.strip().split("=", 1)[1]
        break
os.environ.setdefault("LAMBDA_ROLE", "worker")

import pitfwu_veneer as v  # noqa: E402
from app.services.backtester import BacktesterService  # noqa: E402
from app.services.scanner import scanner_service, _EXCLUDED_SET  # noqa: E402

BASKET = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"]


def _indicators(df):
    df = df.copy()
    pv = df["close"] * df["volume"]
    df["dwap"] = pv.rolling(200, min_periods=50).sum() / df["volume"].rolling(200, min_periods=50).sum()
    df["ma_50"] = df["close"].rolling(50, min_periods=1).mean()
    df["ma_200"] = df["close"].rolling(200, min_periods=1).mean()
    df["vol_avg"] = df["volume"].rolling(20, min_periods=1).mean()
    df["high_52w"] = df["close"].rolling(252, min_periods=1).max()
    return df


def _vix():
    import yfinance as yf
    x = yf.download("^VIX", start="2021-06-01", end="2026-06-05", progress=False, auto_adjust=False)
    x.columns = [(c[0] if isinstance(c, tuple) else c).lower() for c in x.columns]
    x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
    return x


def _cap_universe_by_sector(uni, sector_cap):
    """Cap the candidate universe to max N per GICS sector, keeping list order
    (dollar-volume rank). Mirrors walk_forward_service._apply_sector_cap — the exact
    method behind the May-24 cap=6 validation. Unknown-sector names always pass.
    Sector metadata from /tmp/sectors_cache.json (seed via `aws s3 cp
    s3://<prod-bucket>/universe/sectors_cache.json /tmp/sectors_cache.json`)."""
    if not sector_cap or sector_cap <= 0:
        return uni
    import json
    try:
        cache = json.load(open("/tmp/sectors_cache.json"))
    except Exception:
        print("  [sector-cap] /tmp/sectors_cache.json missing — CAP IS A NO-OP")
        return uni
    counts, out = {}, []
    for s in uni:
        info = cache.get(s)
        sec = (info.get("sector") if isinstance(info, dict) else "") or ""
        if not sec:
            out.append(s); continue
        if counts.get(sec, 0) < sector_cap:
            out.append(s); counts[sec] = counts.get(sec, 0) + 1
    print(f"  [sector-cap={sector_cap}] universe {len(uni)} -> {len(out)}")
    return out


def _load_sector_map():
    """{symbol: sector} from /tmp/sectors_cache.json (seed from the prod bucket)."""
    import json
    try:
        cache = json.load(open("/tmp/sectors_cache.json"))
    except Exception:
        print("  [sectors] /tmp/sectors_cache.json missing — throttles will NO-OP")
        return {}
    return {s: ((i.get("sector") if isinstance(i, dict) else "") or "") for s, i in cache.items()}


def run(start, end, vb_on=True, n=100, trail=0.12, max_pos=6, size=0.15,
        sector_cap=0, max_sector_entries=0, max_entries_per_rebalance=0):
    panel, ca = v.load_panel(), v.load_corp_actions()
    uni = [s for s in v.universe_asof(start, 400, panel) if s not in _EXCLUDED_SET and not s.startswith("^")][:n]
    uni = _cap_universe_by_sector(uni, sector_cap)
    cache = {}
    for s in dict.fromkeys(uni + BASKET + ["SPY"]):
        try:
            cache[s] = _indicators(v.split_adjusted(s, asof=end, ca=ca))
        except Exception as e:
            print(f"  skip {s}: {e}")
    cache["^VIX"] = _vix()
    scanner_service.data_cache = cache
    scanner_service.universe = list(cache.keys())

    bt = BacktesterService()
    bt.trailing_stop_pct = trail; bt.dd_tighten_threshold_pct = 0
    bt.dwap_threshold_pct = 0.05; bt.near_50d_high_pct = 3.0
    bt.max_positions = max_pos; bt.position_size_pct = size; bt.min_price = 15.0
    if max_sector_entries or max_entries_per_rebalance:
        bt.symbol_sectors = _load_sector_map()
        bt.max_sector_entries_per_rebalance = max_sector_entries
        bt.max_entries_per_rebalance = max_entries_per_rebalance
        print(f"  [throttle] max_sector_entries/rebal={max_sector_entries} "
              f"max_entries/rebal={max_entries_per_rebalance} sectors={len(bt.symbol_sectors)}")
    if vb_on:
        bt.cb_pause_basket_enabled = True; bt.cb_pause_basket_symbols = BASKET
        bt.cb_pause_basket_position_size_pct = 10.0; bt.cb_pause_basket_trail_pct = 8.0
        bt.cb_pause_basket_vix_trigger = 30.0
    r = bt.run_backtest(start_date=start, end_date=end, strategy_type="ensemble", force_close_at_end=True)
    yrs = (end - start).days / 365.25
    ann = ((1 + r.total_return_pct / 100) ** (1 / yrs) - 1) * 100
    cal = ann / r.max_drawdown_pct if r.max_drawdown_pct > 0 else 0
    return {"ann": ann, "sharpe": r.sharpe_ratio, "mdd": r.max_drawdown_pct,
            "calmar": cal, "total": r.total_return_pct, "trades": len(r.trades),
            "equity_curve": getattr(r, "equity_curve", None)}


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "2022-01-03"
    b = sys.argv[2] if len(sys.argv) > 2 else "2026-05-29"
    s, e = datetime.fromisoformat(a), datetime.fromisoformat(b)
    print(f"PITFWU walk-forward {a} -> {b} (survivorship-free + split-clean)")
    for label, vb in [("M3 (base+VB)", True), ("BASE (no VB)", False)]:
        m = run(s, e, vb)
        print(f"  {label:14} ann={m['ann']:6.2f}%  sharpe={m['sharpe']:.2f}  "
              f"mdd={m['mdd']:6.2f}%  calmar={m['calmar']:.2f}  trades={m['trades']}")
