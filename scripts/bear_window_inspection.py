"""
Manual inspection of bear-window behavior on the clean 11y pickle.

Goal: for each "shock event" (proxied by SPY 5-day drawdown ≥ 5% AND VIX spike),
tabulate what was tradable in T-14..T+14 surrounding window:
  - SPY drawdown depth and recovery shape
  - VIX peak / decay
  - Top RS leaders during the drawdown (held up best vs SPY)
  - Top bounce names from event-trough to T+14 (best snapback)
  - Mega-cap behavior (AAPL/MSFT/GOOGL/AMZN/META/NVDA/TSLA) as a defensive baseline

Looking for: repeated patterns across events that justify a sub-strategy.
Looking against: every pattern being unique (overfit risk too high).

Output: a per-event summary table + a cross-event aggregate.
No code changes — read-only inspection.
"""
from __future__ import annotations

import gzip
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PICKLE_PATH = Path(__file__).parent.parent / "backend" / "data" / "all_data_11y.pkl.gz"
SPY = "SPY"
VIX = "^VIX"
MEGACAPS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
SHOCK_DRAWDOWN_PCT = 5.0   # SPY 5-day drawdown threshold
SHOCK_VIX_LEVEL = 25.0     # VIX level threshold (any of the 5 days)
WINDOW_DAYS = 14           # ± window around event trough
MIN_VOLUME = 500_000       # filter out illiquid names


def load_pickle() -> dict:
    print(f"Loading pickle: {PICKLE_PATH}")
    with gzip.open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"  {len(data)} symbols loaded")
    return data


def detect_shock_events(spy: pd.DataFrame, vix: pd.DataFrame | None) -> list[pd.Timestamp]:
    """Find SPY 5-day drawdowns ≥ SHOCK_DRAWDOWN_PCT, deduplicated to a single
    'event date' = local trough within a 30-day window. VIX confirmation if present."""
    s = spy["close"].copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    # 5-day rolling max → drawdown = (close / rolling_max) - 1
    rolling_max = s.rolling(5, min_periods=1).max()
    dd = (s / rolling_max - 1) * 100  # negative pct
    shock_days = dd[dd <= -SHOCK_DRAWDOWN_PCT].index

    if vix is not None and len(vix):
        v = vix["close"].copy()
        v.index = pd.to_datetime(v.index)
        v = v.sort_index()
        # require VIX > threshold within ±2 days of shock day
        confirmed = []
        for d in shock_days:
            window = v.loc[d - pd.Timedelta(days=2): d + pd.Timedelta(days=2)]
            if len(window) and window.max() >= SHOCK_VIX_LEVEL:
                confirmed.append(d)
        shock_days = pd.DatetimeIndex(confirmed)

    # Deduplicate: collapse runs within 30 days into single event = local SPY trough
    if len(shock_days) == 0:
        return []
    events = []
    cur_window_start = shock_days[0]
    cur_window_end = shock_days[0]
    for d in shock_days[1:]:
        if (d - cur_window_end).days <= 30:
            cur_window_end = d
        else:
            # close out current event = SPY trough in [cur_window_start - 5d, cur_window_end + 5d]
            tr_lo = cur_window_start - pd.Timedelta(days=5)
            tr_hi = cur_window_end + pd.Timedelta(days=5)
            seg = s.loc[tr_lo:tr_hi]
            if len(seg):
                events.append(seg.idxmin())
            cur_window_start = d
            cur_window_end = d
    # last
    tr_lo = cur_window_start - pd.Timedelta(days=5)
    tr_hi = cur_window_end + pd.Timedelta(days=5)
    seg = s.loc[tr_lo:tr_hi]
    if len(seg):
        events.append(seg.idxmin())
    return events


def event_window(df: pd.DataFrame, event_dt: pd.Timestamp, days: int = WINDOW_DAYS) -> pd.DataFrame:
    """Slice df to event_dt ± days (inclusive of event_dt). df.index is DatetimeIndex."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.loc[event_dt - pd.Timedelta(days=days): event_dt + pd.Timedelta(days=days)]


def pct_change_window(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> float | None:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.loc[start_dt:end_dt]
    if len(df) < 2:
        return None
    p_start = df["close"].iloc[0]
    p_end = df["close"].iloc[-1]
    if not (p_start and np.isfinite(p_start)):
        return None
    return (p_end / p_start - 1) * 100


def avg_volume_window(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> float:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.loc[start_dt:end_dt]
    if "volume" not in df.columns or len(df) == 0:
        return 0.0
    return float(df["volume"].mean() or 0)


def analyze_event(data: dict, event_dt: pd.Timestamp, label: str) -> dict:
    """Per-event analysis. Returns a structured summary."""
    spy_df = data.get(SPY)
    vix_df = data.get(VIX)

    spy_window = event_window(spy_df, event_dt)
    vix_window = event_window(vix_df, event_dt) if vix_df is not None else None

    spy_high = spy_window["close"].max()
    spy_low = spy_window["close"].min()
    spy_drawdown_pct = (spy_low / spy_high - 1) * 100

    spy_pre = pct_change_window(spy_df, event_dt - pd.Timedelta(days=14), event_dt)
    spy_post = pct_change_window(spy_df, event_dt, event_dt + pd.Timedelta(days=14))

    vix_peak = vix_window["close"].max() if vix_window is not None and len(vix_window) else None
    vix_at_event = (
        vix_df.loc[vix_df.index.asof(event_dt), "close"]
        if vix_df is not None and event_dt >= vix_df.index.min()
        else None
    )

    # Per-symbol: pct change pre (T-14 → event), post (event → T+14)
    rs_during, rs_recovery = [], []
    skipped = 0
    for sym, df in data.items():
        if sym in (SPY, VIX) or sym.startswith("^"):
            continue
        try:
            avg_vol = avg_volume_window(df, event_dt - pd.Timedelta(days=14), event_dt + pd.Timedelta(days=14))
            if avg_vol < MIN_VOLUME:
                continue
            pre_change = pct_change_window(df, event_dt - pd.Timedelta(days=14), event_dt)
            post_change = pct_change_window(df, event_dt, event_dt + pd.Timedelta(days=14))
            if pre_change is None or post_change is None:
                skipped += 1
                continue
            # RS during drawdown: symbol_pre - SPY_pre  (positive = held up better)
            rs_during.append((sym, pre_change - (spy_pre or 0), pre_change))
            # Bounce post: post_change ranked
            rs_recovery.append((sym, post_change, post_change - (spy_post or 0)))
        except Exception:
            skipped += 1
            continue

    rs_during.sort(key=lambda x: -x[1])
    rs_recovery.sort(key=lambda x: -x[1])

    # Mega-cap performance
    megacap_pre = []
    megacap_post = []
    for mc in MEGACAPS:
        if mc not in data:
            continue
        pre = pct_change_window(data[mc], event_dt - pd.Timedelta(days=14), event_dt)
        post = pct_change_window(data[mc], event_dt, event_dt + pd.Timedelta(days=14))
        if pre is not None:
            megacap_pre.append((mc, pre))
        if post is not None:
            megacap_post.append((mc, post))

    return {
        "label": label,
        "event_date": event_dt.strftime("%Y-%m-%d"),
        "spy_window_drawdown_pct": round(spy_drawdown_pct, 2),
        "spy_pre_pct (T-14→T)": round(spy_pre, 2) if spy_pre is not None else None,
        "spy_post_pct (T→T+14)": round(spy_post, 2) if spy_post is not None else None,
        "vix_at_event": round(float(vix_at_event), 2) if vix_at_event is not None else None,
        "vix_peak_in_window": round(float(vix_peak), 2) if vix_peak is not None else None,
        "rs_leaders_during_drawdown (top 10 by pre - SPY_pre)": [
            (s, round(rs, 2), round(p, 2)) for (s, rs, p) in rs_during[:10]
        ],
        "rs_recovery_top10 (best post bounce)": [
            (s, round(p, 2), round(rs, 2)) for (s, p, rs) in rs_recovery[:10]
        ],
        "megacap_pre (T-14→T)": [(s, round(v, 2)) for (s, v) in megacap_pre],
        "megacap_post (T→T+14)": [(s, round(v, 2)) for (s, v) in megacap_post],
        "symbols_skipped": skipped,
    }


def main():
    data = load_pickle()
    if SPY not in data:
        print(f"ERROR: {SPY} not in pickle", file=sys.stderr)
        sys.exit(1)
    if VIX not in data:
        print(f"WARNING: {VIX} not in pickle, proceeding without VIX confirmation")

    spy = data[SPY]
    vix = data.get(VIX)

    events = detect_shock_events(spy, vix)
    print(f"\nDetected {len(events)} shock events:")
    for e in events:
        print(f"  {e.strftime('%Y-%m-%d')}")

    print("\n" + "=" * 100)
    summaries = []
    for i, ev in enumerate(events, 1):
        label = f"event_{i}"
        print(f"\n--- {label}: {ev.strftime('%Y-%m-%d')} ---")
        s = analyze_event(data, ev, label)
        summaries.append(s)
        for k, v in s.items():
            if isinstance(v, list):
                print(f"  {k}:")
                for item in v:
                    print(f"    {item}")
            else:
                print(f"  {k}: {v}")

    # Cross-event aggregate: which symbols repeatedly show up as RS leaders?
    print("\n" + "=" * 100)
    print("CROSS-EVENT AGGREGATE")
    print("=" * 100)
    rs_counter = {}
    bounce_counter = {}
    for s in summaries:
        for sym, _rs, _p in s["rs_leaders_during_drawdown (top 10 by pre - SPY_pre)"]:
            rs_counter[sym] = rs_counter.get(sym, 0) + 1
        for sym, _p, _rs in s["rs_recovery_top10 (best post bounce)"]:
            bounce_counter[sym] = bounce_counter.get(sym, 0) + 1

    rs_sorted = sorted(rs_counter.items(), key=lambda x: -x[1])
    bounce_sorted = sorted(bounce_counter.items(), key=lambda x: -x[1])

    print(f"\nSymbols appearing in ≥2 events as 'RS leaders during drawdown':")
    for sym, cnt in rs_sorted:
        if cnt >= 2:
            print(f"  {sym}: {cnt} events")

    print(f"\nSymbols appearing in ≥2 events as 'top bouncers post-event':")
    for sym, cnt in bounce_sorted:
        if cnt >= 2:
            print(f"  {sym}: {cnt} events")

    # Mega-cap baseline
    print(f"\nMega-cap behavior averaged across events:")
    mc_pre_avg = {mc: [] for mc in MEGACAPS}
    mc_post_avg = {mc: [] for mc in MEGACAPS}
    for s in summaries:
        for sym, v in s["megacap_pre (T-14→T)"]:
            mc_pre_avg[sym].append(v)
        for sym, v in s["megacap_post (T→T+14)"]:
            mc_post_avg[sym].append(v)
    print(f"  {'symbol':<8s} {'avg_pre%':>10s} {'avg_post%':>10s} {'n':>4s}")
    for mc in MEGACAPS:
        pre = mc_pre_avg[mc]
        post = mc_post_avg[mc]
        if pre:
            print(f"  {mc:<8s} {sum(pre)/len(pre):>10.2f} {sum(post)/len(post):>10.2f} {len(pre):>4d}")

    # SPY baseline for comparison
    spy_pre_vals = [s["spy_pre_pct (T-14→T)"] for s in summaries if s["spy_pre_pct (T-14→T)"] is not None]
    spy_post_vals = [s["spy_post_pct (T→T+14)"] for s in summaries if s["spy_post_pct (T→T+14)"] is not None]
    if spy_pre_vals and spy_post_vals:
        print(f"  {'SPY':<8s} {sum(spy_pre_vals)/len(spy_pre_vals):>10.2f} {sum(spy_post_vals)/len(spy_post_vals):>10.2f} {len(spy_pre_vals):>4d}")


if __name__ == "__main__":
    main()
