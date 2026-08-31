import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import MarketMeasuredSignup from './components/MarketMeasuredSignup';
import TopNav from './components/TopNav';
import D from './data/sectorObservatory.json';

// The Sector Observatory — ported from design/tools/sector-observatory.html.
// Real PITFWU data (10 yrs, 601 liquid names, 11 sectors). Descriptive research, not a signal.
const REG = {
  strong_bull: { c: '#B4622F', l: 'Strong Bull' }, rotating_bull: { c: '#7A2430', l: 'Rotating Bull' },
  weak_bull: { c: '#C9A46A', l: 'Weak Bull' }, range_bound: { c: '#A79E8C', l: 'Range-bound' },
  recovery: { c: '#77835A', l: 'Recovery' }, weak_bear: { c: '#5E6B78', l: 'Weak Bear' },
  panic_crash: { c: '#2A2320', l: 'Panic / Crash' },
};
const REG_ORDER = ['strong_bull', 'rotating_bull', 'weak_bull', 'range_bound', 'recovery', 'weak_bear', 'panic_crash'];
const MONO = "ui-monospace,'SF Mono','IBM Plex Mono',Menlo,monospace";

const SECT = D.sectors, M = D.months, S = D.summary, NREG = D.months.length;

// Draws the rotation-river heatmap on a canvas; returns a cleanup fn. Mirrors the source tool,
// with the year-axis min-gap fix so the partial first year (Dec 2016) doesn't garble into 2017.
function drawHeatmap(cv, tip) {
  const ctx = cv.getContext('2d');
  const share = S.rotation_cadence.leader_share;
  const order = SECT.map((s, i) => ({ s, i, sh: share[s] || 0 })).sort((a, b) => b.sh - a.sh);
  const rankColor = (r) => {
    if (r == null) return '#EFE9DB';
    const t = (r - 1) / (SECT.length - 1);
    const a = [107, 31, 42], b = [237, 230, 214];
    const m = [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * Math.pow(t, 0.85)));
    return 'rgb(' + m[0] + ',' + m[1] + ',' + m[2] + ')';
  };
  const LEFT = 132, TOPBAND = 18, GAP = 10, ROWH = 20, AXIS = 26, ROWS = SECT.length;
  const H = TOPBAND + GAP + ROWS * ROWH + AXIS;
  let cw = 0;
  function draw() {
    const cssW = cv.clientWidth; if (!cssW) return;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    cv.width = cssW * dpr; cv.height = H * dpr; cv.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, H);
    const plotW = cssW - LEFT - 6; cw = plotW / NREG;
    // regime ribbon — weekly resolution (catches fast crashes the monthly cells miss).
    const WK = D.regimeWeekly, nwk = WK.length, wkw = plotW / nwk;
    for (let j = 0; j < nwk; j++) {
      const rg = WK[j].r; ctx.fillStyle = (REG[rg] && REG[rg].c) || '#CFC7B5';
      ctx.fillRect(LEFT + j * wkw, 0, Math.ceil(wkw) + 0.8, TOPBAND);
    }
    ctx.fillStyle = '#8A8172'; ctx.font = '10px ' + MONO; ctx.textBaseline = 'middle';
    ctx.fillText('REGIME', LEFT - 8 - ctx.measureText('REGIME').width, TOPBAND / 2);
    // sector rows
    for (let ri = 0; ri < ROWS; ri++) {
      const o = order[ri], y = TOPBAND + GAP + ri * ROWH;
      for (let j = 0; j < NREG; j++) {
        ctx.fillStyle = rankColor(M[j].ranks[o.i]);
        ctx.fillRect(LEFT + j * cw, y, Math.ceil(cw) + 0.4, ROWH - 2);
      }
      ctx.fillStyle = '#3A342C'; ctx.font = '12px system-ui,sans-serif'; ctx.textAlign = 'right';
      ctx.fillText(o.s, LEFT - 8, y + (ROWH - 2) / 2);
      ctx.textAlign = 'left';
    }
    // year axis — collect year boundaries, then label with a min-gap so the partial first
    // year (data starts Dec 2016, a single month) doesn't garble into 2017's label.
    ctx.fillStyle = '#8A8172'; ctx.font = '10px ' + MONO; ctx.textAlign = 'center';
    const ybounds = []; let lastY = null;
    for (let j = 0; j < NREG; j++) { const yr = M[j].d.slice(0, 4); if (yr !== lastY) { lastY = yr; ybounds.push({ yr, x: LEFT + j * cw + cw / 2 }); } }
    for (let b = 0; b < ybounds.length; b++) {
      const { yr, x } = ybounds[b];
      ctx.strokeStyle = '#E0D8C6'; ctx.beginPath(); ctx.moveTo(x, TOPBAND + GAP - 3); ctx.lineTo(x, TOPBAND + GAP + ROWS * ROWH); ctx.stroke();
      if (b + 1 < ybounds.length && ybounds[b + 1].x - x < 30) continue;   // skip the partial first year
      ctx.fillText(yr, x, TOPBAND + GAP + ROWS * ROWH + 12);
    }
    ctx.textAlign = 'left';
  }
  const idxAt = (clientX) => {
    const r = cv.getBoundingClientRect(); const x = clientX - r.left;
    if (x < LEFT) return -1; const j = Math.floor((x - LEFT) / cw); return (j >= 0 && j < NREG) ? j : -1;
  };
  const onMove = (e) => {
    const j = idxAt(e.clientX);
    if (j < 0) { tip.style.opacity = 0; return; }
    const m = M[j];
    const top3 = order.map((o) => ({ s: o.s, r: m.ranks[o.i], ret: m.ret[o.i] }))
      .filter((z) => z.r != null).sort((a, b) => a.r - b.r).slice(0, 3);
    const rg = REG[m.r] || { c: '#888', l: m.r || '—' };
    tip.innerHTML = '<div class="th">' + m.d + '</div>' +
      '<span class="rg" style="background:' + rg.c + ';color:#fff">' + rg.l + '</span>' +
      top3.map((z) => '<div class="rw"><span data-r="' + z.r + '">' + z.s + '</span><span>' +
        (z.ret >= 0 ? '+' : '') + (z.ret * 100).toFixed(1) + '%</span></div>').join('');
    tip.style.opacity = 1;
    const tw = 236, x = Math.min(e.clientX + 14, window.innerWidth - tw - 8), y = Math.min(e.clientY + 14, window.innerHeight - 90);
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  };
  const onLeave = () => { tip.style.opacity = 0; };
  cv.addEventListener('mousemove', onMove);
  cv.addEventListener('mouseleave', onLeave);
  window.addEventListener('resize', draw);
  draw();
  return () => {
    cv.removeEventListener('mousemove', onMove);
    cv.removeEventListener('mouseleave', onLeave);
    window.removeEventListener('resize', draw);
  };
}

const SectionLabel = ({ children }) => (
  <p className="mb-1" style={{ fontFamily: MONO, fontSize: '11.5px', letterSpacing: '.2em', textTransform: 'uppercase', color: '#8A8172' }}>{children}</p>
);

export default function BlogSectorObservatoryPage() {
  const canvasRef = useRef(null);
  const tipRef = useRef(null);

  useEffect(() => { document.title = 'The Sector Observatory — RigaCap'; }, []);
  useEffect(() => {
    if (!canvasRef.current || !tipRef.current) return;
    return drawHeatmap(canvasRef.current, tipRef.current);
  }, []);

  const p = S.params;
  const regsPresent = [...new Set(D.regimeWeekly.map((w) => w.r).filter(Boolean))]
    .sort((a, b) => REG_ORDER.indexOf(a) - REG_ORDER.indexOf(b));

  // persistence bars
  const per = S.persistence_spearman, pKeys = ['1m', '3m', '6m', '12m'];
  const maxAbs = Math.max(...pKeys.map((k) => Math.abs(per[k] || 0)), 0.6);

  // cadence
  const cad = S.rotation_cadence;

  // regime table
  const regimeRows = Object.entries(S.regime_leaders).sort((a, b) => b[1].n - a[1].n);

  // secular drift
  const dr = Object.entries(S.secular_drift_rank_slope).sort((a, b) => a[1] - b[1]);
  const driftUp = dr.filter((x) => x[1] < 0);
  const driftDn = dr.filter((x) => x[1] >= 0).reverse();
  const fmtDrift = (v) => (v < 0 ? '▲ ' : '▼ ') + Math.abs(v * 100).toFixed(1);

  return (
    <div className="min-h-screen bg-paper text-ink antialiased">
      <TopNav />

      <article className="max-w-[1000px] mx-auto px-4 sm:px-8 pt-14 pb-20">
        {/* Hero */}
        <p className="font-semibold" style={{ fontFamily: MONO, fontSize: '11.5px', letterSpacing: '.22em', textTransform: 'uppercase', color: '#7A2430' }}>
          RigaCap Research · Sector Observatory
        </p>
        <h1 className="font-display font-medium text-ink mt-2 mb-2 tracking-[-0.01em] leading-[1.02]" style={{ fontSize: 'clamp(34px,6vw,58px)', maxWidth: '16ch', textWrap: 'balance', fontVariationSettings: '"opsz" 144' }}>
          What&rsquo;s leading under the hood
        </h1>
        <p className="font-display italic text-ink-mute mb-6" style={{ fontSize: 'clamp(18px,2.4vw,23px)', maxWidth: '40ch', lineHeight: 1.4 }}>
          Ten years of sector leadership, ranked month by month and colored by the market regime it happened in.
        </p>
        <div className="flex flex-wrap gap-x-7 gap-y-1 border-y border-rule py-3 mb-9" style={{ fontFamily: MONO, fontSize: '12px', color: '#8A8172' }}>
          <span><b className="text-ink font-semibold">{p.first}</b> → <b className="text-ink font-semibold">{p.last}</b></span>
          <span><b className="text-ink font-semibold">{p.dates}</b> months</span>
          <span><b className="text-ink font-semibold">{SECT.length}</b> sectors · <b className="text-ink font-semibold">601</b> liquid names</span>
          <span>strength = <b className="text-ink font-semibold">{p.rs_lookback}-day</b> relative return</span>
        </div>

        {/* Rotation river */}
        <SectionLabel>The rotation river</SectionLabel>
        <h2 className="font-display font-medium text-ink mb-1.5 tracking-[-0.01em]" style={{ fontSize: 'clamp(23px,3vw,30px)', textWrap: 'balance' }}>Leadership never sits still</h2>
        <p className="text-ink-mute mb-5" style={{ maxWidth: '62ch' }}>
          Each column is one month; each row a sector. The darker the cell, the stronger that sector&rsquo;s
          3-month relative strength that month (rank 1 = deepest claret). The band on top is the market regime,
          sampled weekly &mdash; fine enough to catch a fast crash like COVID&rsquo;s. Read left to right and you watch leadership migrate.
        </p>
        <div className="rounded-[10px] border border-rule px-3.5 pt-4 pb-2 mb-3.5 overflow-x-auto" style={{ background: '#FBF8F0' }}>
          <canvas ref={canvasRef} height="300" style={{ display: 'block', width: '100%', minWidth: '660px' }} />
        </div>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3 mt-3 mb-2" style={{ fontFamily: MONO, fontSize: '11px', color: '#3A342C' }}>
          <span className="flex items-center gap-2 flex-wrap">
            <span>strongest</span>
            <span style={{ display: 'inline-block', width: '112px', height: '11px', borderRadius: '2px', background: 'linear-gradient(90deg,#6B1F2A,#B4626A,#D8C7B4,#EDE6D6)' }} />
            <span>weakest</span>
          </span>
          <span className="flex items-center gap-x-4 gap-y-1 flex-wrap">
            {regsPresent.map((r) => (
              <span key={r} className="flex items-center gap-1.5">
                <span style={{ display: 'inline-block', width: '11px', height: '11px', borderRadius: '2px', background: REG[r].c }} />
                {REG[r].l}
              </span>
            ))}
          </span>
        </div>

        {/* Findings grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-[18px] mt-9">
          {/* Persistence */}
          <div className="rounded-[10px] border border-rule px-[22px] py-5" style={{ background: '#FBF8F0' }}>
            <SectionLabel>Persistence</SectionLabel>
            <h3 className="font-display text-[20px] font-medium text-ink mb-1">Momentum lasts a month, not a quarter</h3>
            <div className="flex items-end gap-4 h-[118px] mt-3.5 mb-1.5 pt-2">
              {pKeys.map((k) => {
                const v = per[k] || 0, h = Math.max(2, Math.abs(v) / maxAbs * 100);
                const pos = v >= 0.03, near = Math.abs(v) < 0.03;
                return (
                  <div key={k} className="flex-1 flex flex-col items-center gap-1.5 h-full justify-end">
                    <div style={{ fontFamily: MONO, fontSize: '12px', fontVariantNumeric: 'tabular-nums', color: '#141210' }}>{(v >= 0 ? '+' : '') + v.toFixed(2)}</div>
                    {near
                      ? <div title="≈ zero" style={{ width: '100%', maxWidth: '46px', height: '2px', background: '#8A8172' }} />
                      : <div style={{ width: '100%', maxWidth: '46px', height: h + '%', borderRadius: '3px 3px 0 0', background: pos ? '#7A2430' : '#8A8172' }} />}
                    <div style={{ fontFamily: MONO, fontSize: '11px', color: '#8A8172' }}>{k}</div>
                  </div>
                );
              })}
            </div>
            <p className="text-[14.5px] text-ink-mute mt-1.5">
              Rank-autocorrelation of sector strength at each horizon. A strong <b className="text-ink">+0.56 at one month</b> collapses
              to noise by three &mdash; today&rsquo;s leader is a coin flip a quarter out.
            </p>
          </div>

          {/* Cadence */}
          <div className="rounded-[10px] border border-rule px-[22px] py-5" style={{ background: '#FBF8F0' }}>
            <SectionLabel>Cadence</SectionLabel>
            <h3 className="font-display text-[20px] font-medium text-ink mb-1">The crown changes hands fast</h3>
            <div className="font-display font-medium text-claret leading-none" style={{ fontSize: '44px', fontVariantNumeric: 'tabular-nums' }}>
              {cad.avg_months_leader_holds.toFixed(1)}
              <small style={{ fontFamily: MONO, fontSize: '13px', color: '#8A8172', fontWeight: 400, marginLeft: '6px' }}>months on top</small>
            </div>
            <p className="text-[14.5px] text-ink-mute mt-1.5">
              The #1 sector holds the lead for about six weeks before the crown moves. All <b className="text-ink">{cad.distinct_leaders}</b> sectors have led at some point.
            </p>
          </div>

          {/* Regime leaders */}
          <div className="rounded-[10px] border border-rule px-[22px] py-5" style={{ background: '#FBF8F0' }}>
            <SectionLabel>Under the covers · by regime</SectionLabel>
            <h3 className="font-display text-[20px] font-medium text-ink mb-2">Who leads when</h3>
            <table className="w-full text-[14px]" style={{ borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {['Regime', 'Months', 'Top leaders'].map((h, i) => (
                    <th key={h} className={i === 1 ? 'text-right' : 'text-left'} style={{ fontFamily: MONO, fontSize: '10.5px', letterSpacing: '.12em', textTransform: 'uppercase', color: '#8A8172', fontWeight: 600, padding: '7px 8px', borderBottom: '1px solid #E0D8C6' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {regimeRows.map(([reg, o]) => (
                  <tr key={reg}>
                    <td style={{ padding: '7px 8px', borderBottom: '1px solid #E0D8C6' }}>
                      <span style={{ display: 'inline-block', width: '11px', height: '11px', borderRadius: '2px', background: (REG[reg] && REG[reg].c) || '#999', marginRight: '7px', verticalAlign: '-1px' }} />
                      {(REG[reg] && REG[reg].l) || reg}
                    </td>
                    <td className="text-right" style={{ fontFamily: MONO, color: '#8A8172', fontVariantNumeric: 'tabular-nums', padding: '7px 8px', borderBottom: '1px solid #E0D8C6' }}>{o.n}</td>
                    <td style={{ padding: '7px 8px', borderBottom: '1px solid #E0D8C6' }}>
                      {o.top.map((t) => (
                        <span key={t[0]} style={{ fontFamily: MONO, fontSize: '11.5px', padding: '2px 7px', borderRadius: '99px', marginRight: '5px', background: '#EFE9DB', color: '#141210', display: 'inline-block', marginBottom: '3px' }}>{t[0]}</span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Secular drift */}
          <div className="rounded-[10px] border border-rule px-[22px] py-5" style={{ background: '#FBF8F0' }}>
            <SectionLabel>Secular drift</SectionLabel>
            <h3 className="font-display text-[20px] font-medium text-ink mb-2">The slow tide, 2016&ndash;2026</h3>
            <div className="grid grid-cols-2 gap-[22px]">
              {[{ cap: 'Climbing ▲', color: '#4B6B45', rows: driftUp }, { cap: 'Fading ▼', color: '#7A2430', rows: driftDn }].map((col) => (
                <div key={col.cap}>
                  <div style={{ fontFamily: MONO, fontSize: '11px', textTransform: 'uppercase', letterSpacing: '.12em', color: col.color }}>{col.cap}</div>
                  <ul className="list-none mt-2 p-0">
                    {col.rows.map((x) => (
                      <li key={x[0]} className="flex justify-between text-[14px] py-[5px]" style={{ borderBottom: '1px dotted #E0D8C6' }}>
                        <span>{x[0]}</span>
                        <span style={{ fontFamily: MONO, fontVariantNumeric: 'tabular-nums', color: '#8A8172' }}>{fmtDrift(x[1])}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Verdict */}
        <div className="mt-10 pl-[22px] py-1.5" style={{ borderLeft: '3px solid #7A2430' }}>
          <SectionLabel>The verdict</SectionLabel>
          <h2 className="font-display font-medium text-ink mb-2.5 tracking-[-0.01em]" style={{ fontSize: 'clamp(23px,3vw,30px)' }}>Can you call the next hot sector? No.</h2>
          <p className="text-ink-mute mb-3" style={{ maxWidth: '64ch' }}>
            Beyond that one month of momentum &mdash; which our system already rides &mdash; leadership is
            effectively unforecastable. Relative-strength acceleration and within-sector breadth, the usual
            &ldquo;early&rdquo; tells, showed <b className="text-ink">no predictive lead</b>. And the one thread that looked
            promising &mdash; a hint of three-month mean-reversion &mdash; <b className="text-ink">did not survive a clean test</b>: measured as a
            real top-vs-bottom forward-return spread, it&rsquo;s a statistically-insignificant &minus;0.8%
            (every horizon <span style={{ fontFamily: MONO }}>|t| &lt; 2</span>). The scary early number was a
            bounded-rank artifact. There is <b className="text-ink">no tradeable sector-timing edge here, in either direction.</b>
          </p>
          <p className="text-ink-mute" style={{ maxWidth: '64ch' }}>
            Where it clearly earns its keep is <b className="text-ink">context</b>: a Rotating Bull runs on Technology, Energy and
            Basic Materials trading the lead; defensive regimes hand it to Staples and Utilities. That map is
            the shareable, authority-building story.
          </p>
        </div>

        <p className="mt-2.5" style={{ fontFamily: MONO, fontSize: '11.5px', color: '#8A8172', maxWidth: '70ch' }}>
          Caveat: covers the live liquid universe (10 yrs of point-in-time depth), not the full 21-year record.
          Findings are descriptive; no sector-timing rule cleared statistical significance when tested as a real forward-return spread.
        </p>
        <div className="mt-11 pt-4 border-t border-rule flex justify-between flex-wrap gap-2" style={{ fontFamily: MONO, fontSize: '11px', color: '#8A8172' }}>
          <span>RigaCap · read-only research, not investment advice</span>
          <span>{p.first}–{p.last}</span>
        </div>
      </article>

      {/* Signup */}
      <section className="pb-16 sm:pb-20">
        <div className="max-w-[720px] mx-auto px-4 sm:px-8">
          <MarketMeasuredSignup source="blog_sector_observatory" />
        </div>
      </section>

      {/* Disclaimer */}
      <section className="max-w-[720px] mx-auto px-4 sm:px-8 pb-8">
        <p className="text-[0.78rem] text-ink-light leading-relaxed">
          All performance figures are from walk-forward simulations using historical market data.
          For information purposes only — not a solicitation to invest, purchase, or sell securities in which RigaCap has an interest.
          RigaCap, LLC is not a registered investment advisor. Past performance does not guarantee future results.
          Execute trades through your own brokerage account. See our{' '}
          <Link to="/terms" className="text-ink-mute underline hover:text-ink transition-colors">Terms of Service</Link>{' '}
          for full disclaimers.
        </p>
      </section>

      {/* Footer */}
      <footer className="border-t border-rule py-8 text-center text-[0.78rem] text-ink-light">
        <p>&copy; {new Date().getFullYear()} RigaCap, LLC. All rights reserved.</p>
      </footer>

      {/* Heatmap tooltip (imperatively driven) */}
      <div ref={tipRef} style={{
        position: 'fixed', zIndex: 20, pointerEvents: 'none', background: '#141210', color: '#F5F1E8',
        fontFamily: MONO, fontSize: '11.5px', lineHeight: 1.5, padding: '9px 11px', borderRadius: '7px',
        boxShadow: '0 8px 24px rgba(20,18,16,.28)', maxWidth: '230px', opacity: 0, transition: 'opacity .1s',
      }} />
      <style>{`
        [data-r]::before { content: attr(data-r) "  "; color: #C9A46A; }
        .rw { display: flex; justify-content: space-between; gap: 14px; }
        .th { font-weight: 700; letter-spacing: .02em; margin-bottom: 3px; }
        .rg { display: inline-block; padding: 1px 6px; border-radius: 99px; font-size: 10px; margin-bottom: 5px; }
      `}</style>
    </div>
  );
}
