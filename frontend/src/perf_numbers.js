// SINGLE SOURCE OF TRUTH for every public performance number.
// Construction: OVERLAY (what we actually ship) — Preserver = t30v engine + capitulation
// cash-raise; Maximizer = that base + a real held breakout book in rotating-bull. NOT the
// idealized sleeve backtest. Survivorship-free; point-in-time since 2016 (pre-2016 = disclosed
// supporting evidence, not proof). Vintage 2026-07-31 (see docs/numbers-citations-registry.md §1).
//
// Rules for using these on public surfaces:
//   - NEVER expose internal names: t30v / Core / Ensemble / sleeve internals / DWAP.
//   - Returns are the HERO; sharpe/calmar/maxdd are supporting.
//   - Consumer pages: plain language ("worst drop"); adviser/methodology may use Sharpe/Calmar.
//   - Recent performance = the ROLLING-window average (drift-proof), NOT a single trailing window.
//   - plan.* (conservative planning assumptions) belong in DEPTH (methodology/advisers/FAQ), never the hero.

export const PERF = {
  vintage: '2026-07-31',
  hi_confidence_window: '2016 onward (survivorship-free, point-in-time)',

  preserver: {
    label: 'RigaCap Preserver',
    // Rolling-window averages over 2021–26 (the modern market) — the "typical" experience.
    typical_12mo: 10.7,          // avg return across every rolling 1-yr window
    typical_12mo_positive_pct: 77,
    typical_24mo: 20.8,
    yr5:  { cagr: 13.0, sharpe: 1.28, calmar: 1.01, maxdd: -12.9 },  // 2021–26 path
    yr21: { cagr: 7.7,  sharpe: 0.87, calmar: 0.56, maxdd: -13.7 },  // 2007–26 foundation
    plan_low: 11, plan_high: 13, // conservative planning assumption (DEPTH ONLY)
  },

  maximizer: {
    label: 'RigaCap Maximizer',
    typical_12mo: 26.5,
    typical_12mo_positive_pct: 77,
    typical_24mo: 57.5,
    yr5:  { cagr: 31.4, sharpe: 1.51, calmar: 2.10, maxdd: -14.9 },
    yr21: { cagr: 13.5, sharpe: 0.93, calmar: 0.65, maxdd: -20.8 },
    plan_low: 13, plan_high: 17,
    // Honest caveat for depth pages: the breakout leg is the newest component (finalized 2026),
    // so its incremental edge over Preserver carries more selection risk — plan conservatively.
    breakout_selection_caveat: true,
  },

  benchmarks: {
    spy_5yr:      { cagr: 14.2, sharpe: 0.87, calmar: 0.56, maxdd: -25.4 },
    spy_21yr:     { cagr: 9.8,  maxdd: -55 },
    raw_mom_21yr: { cagr: 13.2, sharpe: 0.69, maxdd: -57 },
  },

  // Supporting stats — RE-DERIVED on the overlay construction (2007–26). The downside story
  // survives the re-baseline: being ~75% cash in capitulation protects, so 2008 is still ~flat.
  supporting: {
    yr2008:  { preserver: 0.1,  maximizer: 0.1,  spy: -37.7 },
    yr2020:  { preserver: 9.2,  maximizer: 34.9, spy: 15.2 },
    yr2022:  { preserver: -11.2, maximizer: -6.4, spy: -19.9 },
    down_month_capture: { preserver: -1.05, maximizer: -0.97, spy: -3.85 }, // avg return in S&P down months
    up_month_capture:   { preserver: 1.59,  maximizer: 2.33 },
    monthly_corr_spy:   { preserver: 0.55, maximizer: 0.39 },
    longest_underwater_yrs: { preserver: 2.2, maximizer: 3.4, spy: 5.4 },
    beats_spy_pct: { preserver: { y1: 36, y3: 17, y5: 14 }, maximizer: { y1: 50, y3: 45, y5: 31 } },
  },
};

export default PERF;
