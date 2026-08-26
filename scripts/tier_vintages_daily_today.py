"""Phase 1 — DAILY, shippable tier vintages on the recent (CLEAN) windows.

The long-history cut was biweekly (understates DD) + EXT (survivorship-biased). Here:
  - DAILY resolution (true drawdowns, esp. Maximizer++'s momentum crash).
  - Recent windows only (2021-26, last-2yr) = CLEAN survivorship-free data, no EXT caveat.
  - Lead marketing with these (recent past ~ near future).
t30v daily = single-backtest proxy at t30v config (walk-forward emits only biweekly; the
proxy is slightly conservative on DD). Maximizer++ breakout leg wears the vol-scaling brake.
RESEARCH ONLY, local.
"""
import os, sys
R = "/Users/erikkins/CODE/stocker-app"
sys.path.insert(0, os.path.join(R, "backend")); sys.path.insert(0, os.path.join(R, "scripts"))
for _l in open(os.path.join(R, ".env")):
    if _l.startswith("DATABASE_URL="):
        os.environ["DATABASE_URL"] = _l.strip().split("=", 1)[1]; break
os.environ.setdefault("LAMBDA_ROLE", "worker")

import numpy as np
import pandas as pd
from shape_tpe import load_data
from stack_sleeves import sleeve_curve, PULLBACK, OVERSOLD
from regime_allocator_v2 import BREAKOUT
from regime_research import regime_series
from shapes_portfolio import perf, CAP0
import pitfwu_wf as pwf

CALM_BULL = {"strong_bull", "weak_bull"}
CAPITULATION = {"panic_crash", "recovery", "weak_bear"}
ROTATING = {"rotating_bull"}
WINDOWS = [("LAST 2YR (2024-05->2026-05)", pd.Timestamp("2024-07-30"), pd.Timestamp("2026-07-30")),
           ("2021-2026 (incl 2022 + momo-crash)", pd.Timestamp("2021-01-01"), pd.Timestamp("2026-07-30"))]


def t30v_daily(start, end):
    m = pwf.run(start.to_pydatetime(), end.to_pydatetime(), trail=0.30, max_pos=20, size=0.045)
    ec = m.get("equity_curve") or []
    return pd.Series([x["equity"] for x in ec], index=pd.to_datetime([x["date"] for x in ec])).sort_index()


def vol_scale(ret, target=0.20):
    rv = (ret.rolling(20).std() * np.sqrt(252)).shift(1)   # daily annualized, lagged
    return (target / rv).clip(upper=1.0).fillna(1.0)


if __name__ == "__main__":
    for label, start, end in WINDOWS:
        print(f"\n================ {label} — DAILY, clean data ================", flush=True)
        data = load_data(start, end)
        pb = sleeve_curve(data, start, end, PULLBACK, "pullback_ma")
        ob = sleeve_curve(data, start, end, OVERSOLD, "oversold_bounce")
        bk = sleeve_curve(data, start, end, BREAKOUT, "breakout")
        t = t30v_daily(start, end)
        grid = t.index
        rt = t.pct_change()
        rp = pb.reindex(grid, method="ffill").pct_change()
        ro = ob.reindex(grid, method="ffill").pct_change()
        rb = bk.reindex(grid, method="ffill").pct_change()
        reg = regime_series(start, end).reindex(grid, method="ffill").fillna("none")
        calm = reg.isin(CALM_BULL).to_numpy(); cap = reg.isin(CAPITULATION).to_numpy(); rot = reg.isin(ROTATING).to_numpy()
        df = pd.DataFrame({"t": rt, "p": rp, "o": ro, "b": rb}).fillna(0.0)
        b_scaled = df["b"] * vol_scale(df["b"])
        preserver = pd.Series(np.where(calm, df["p"], np.where(cap, df["o"], df["t"])), index=grid)
        maxpp = pd.Series(np.where(calm, df["p"], np.where(cap, df["o"], np.where(rot, b_scaled, df["t"]))), index=grid)
        print(f"  {'tier':<26} {'Annualized':>11} {'Sharpe':>7} {'MaxDD':>8}", flush=True)
        for name, r in [("Core (t30v)", df["t"]), ("Preserver (research)", preserver), ("Maximizer (research)", maxpp)]:
            c, s, m = perf(CAP0 * (1 + r).cumprod(), ppy=252)
            print(f"  {name:<26} {c:>10.1f}% {s:>7.2f} {m:>7.1f}%", flush=True)
        # --- Option A: HARD-ROTATE the whole book on each regime flip = research return-stream
        #     MINUS a round-trip rotation cost on flip days. Shows how much of the marketed edge
        #     a full rotate recovers (vs rule-B production which loses it) and the turnover it costs.
        COST_RT = 0.003  # 0.15% each way on a full-book rotate
        psrc = pd.Series(np.where(calm, "p", np.where(cap, "o", "t")), index=grid)
        msrc = pd.Series(np.where(calm, "p", np.where(cap, "o", np.where(rot, "b", "t"))), index=grid)
        yrs = (grid[-1] - grid[0]).days / 365.25
        for name, r, src in [("Preserver (Option A)", preserver, psrc), ("Maximizer (Option A)", maxpp, msrc)]:
            flip = (src != src.shift(1)).fillna(False)
            ra = r.copy(); ra[flip] = ra[flip] - COST_RT
            c, s, m = perf(CAP0 * (1 + ra).cumprod(), ppy=252)
            nflip = int(flip.sum())
            print(f"  {name:<26} {c:>10.1f}% {s:>7.2f} {m:>7.1f}%   flips={nflip} ({nflip/yrs:.1f}/yr, full-book rotates)", flush=True)
        # --- Confirmation-filtered Option A: only rotate after the routed source persists N days
        #     (short spells ignored -> stay put). Cuts whipsaw turnover; costs an N-day lag at real
        #     regime changes. Shows the deliverable trade: does it hold the edge at fewer rotates?
        ret_map_p = {"p": df["p"], "o": df["o"], "t": df["t"]}
        ret_map_m = {"p": df["p"], "o": df["o"], "t": df["t"], "b": b_scaled}
        def confirm(src, N):
            out = []; cur = src.iloc[0]; pend = None; pc = 0
            for s in src:
                if s == cur:
                    pend = None; pc = 0
                else:
                    pc = pc + 1 if s == pend else 1; pend = s
                    if pc >= N:
                        cur = s; pend = None; pc = 0
                out.append(cur)
            return pd.Series(out, index=src.index)
        for tlabel, src0, rmap in [("Preserver", psrc, ret_map_p), ("Maximizer", msrc, ret_map_m)]:
            for N in (3, 5, 8):
                cs = confirm(src0, N)
                r = pd.Series([rmap[x].loc[d] for d, x in cs.items()], index=cs.index)
                flip = (cs != cs.shift(1)).fillna(False)
                r2 = r.copy(); r2[flip] = r2[flip] - COST_RT
                c, s, m = perf(CAP0 * (1 + r2).cumprod(), ppy=252)
                nf = int(flip.sum())
                print(f"  {tlabel+' confirm-'+str(N)+'d':<26} {c:>10.1f}% {s:>7.2f} {m:>7.1f}%   flips={nf} ({nf/yrs:.1f}/yr)", flush=True)
        # --- OVERLAY/HEDGE: keep the t30v book (NO full liquidation); only RAISE CASH in
        #     capitulation (exposure -> E). Return = exposure * t30v_ret; turnover = a partial
        #     trade of size |Δexposure| on capitulation enter/exit only. Low turnover, tax-light.
        cap_mask = pd.Series(cap, index=grid)
        for E in (0.25, 0.5, 0.75):
            exp = pd.Series(np.where(cap_mask.to_numpy(), E, 1.0), index=grid)
            dexp = exp.diff().abs().fillna(0.0)
            r_ov = exp * df["t"] - dexp * 0.0015          # one-way cost on the traded fraction
            c, s, m = perf(CAP0 * (1 + r_ov).cumprod(), ppy=252)
            ntr = int((dexp > 1e-9).sum())
            print(f"  {'Preserver overlay E='+str(E):<26} {c:>10.1f}% {s:>7.2f} {m:>7.1f}%   cash-raises={ntr} ({ntr/yrs:.1f}/yr, partial)", flush=True)
