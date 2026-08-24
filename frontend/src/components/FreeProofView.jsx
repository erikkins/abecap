import React from 'react';
import { Lock, TrendingUp, ShieldCheck, LogIn, ArrowRight } from 'lucide-react';
import { PERF } from '../perf_numbers';

// FreeProofView — the FREE tier's dashboard experience (project_free_first_spec §2).
// PROOF ONLY, honest by construction: no live names/entries/weights/exits. Composition Erik
// approved (Aug 24): dynamic "where we are right now" phase → long-term walk-forward record
// WITH its real drawdowns → what we hold right now (COUNTS, no names) → recent named winners
// (closed/resolved catches) → ticker-free market read → upgrade CTA. Warts shown on purpose:
// the current live book is young and down, and we say so.

const pctText = (v, dp = 1) =>
  v === null || v === undefined ? '—' : `${v > 0 ? '+' : ''}${Number(v).toFixed(dp)}%`;

function LongTermRecord() {
  const p = PERF.preserver.yr21;
  const m = PERF.maximizer.yr21;
  const spy = PERF.benchmarks.spy_21yr;
  const raw = PERF.benchmarks.raw_mom_21yr;
  const y08 = PERF.supporting.yr2008;
  const Row = ({ label, cagr, maxdd, accent }) => (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <span className={`font-medium ${accent ? 'text-claret' : 'text-ink'}`}>{label}</span>
      <span className="tabular-nums text-ink">{cagr}%<span className="text-ink-mute">/yr</span></span>
      <span className="tabular-nums text-ink-mute">{maxdd}% max DD</span>
    </div>
  );
  return (
    <div className="bg-paper-card rounded-[2px] border border-ink/10 p-4">
      <div className="flex items-center gap-2 mb-2">
        <ShieldCheck className="w-4 h-4 text-claret" />
        <h4 className="font-display text-[1.05rem] font-medium text-ink" style={{ fontVariationSettings: '"opsz" 32' }}>
          The 21-year walk-forward record
        </h4>
      </div>
      <p className="text-xs text-ink-mute mb-2">
        Tested honestly through 2008 and 2022 — drawdowns shown, not hidden.
      </p>
      <div className="divide-y divide-ink/5">
        <Row label="Preserver" cagr={p.cagr} maxdd={p.maxdd} accent />
        <Row label="Maximizer" cagr={m.cagr} maxdd={m.maxdd} accent />
        <Row label="S&P 500" cagr={spy.cagr} maxdd={spy.maxdd} />
        <Row label="Raw momentum (no floor)" cagr={raw.cagr} maxdd={raw.maxdd} />
      </div>
      <p className="text-xs text-ink-mute mt-2 leading-relaxed">
        The edge isn't a higher return — it's the <span className="text-ink font-medium">drawdown</span>:
        roughly the market's return with a fraction of its worst loss. In 2008 the market fell
        {' '}{y08.spy}% while both settings finished about flat.
      </p>
    </div>
  );
}

export default function FreeProofView({ data, user, upgradeLoading, onSubscribe, onSignIn }) {
  const phase = data?.current_phase;
  const ob = data?.open_book || {};
  const pres = ob.preserver;
  const maxi = ob.maximizer;
  const winners = (data?.closed_ledger?.winners) || [];
  const read = data?.market_read_free?.text;

  const isSoft = phase?.phase === 'soft_patch';

  // Live introductory pricing from the founding-status SSOT (so the CTA shows the ACTIVE deal,
  // not a hardcoded price). Falls back to standard if the fetch fails.
  const [pricing, setPricing] = React.useState(null);
  React.useEffect(() => {
    const base = import.meta.env.VITE_API_URL || '';
    fetch(`${base}/api/billing/founding-status`)
      .then((r) => r.json())
      .then((d) => setPricing(d?.pricing || null))
      .catch(() => {});
  }, []);
  const introMo = pricing?.intro_monthly ?? 129;
  const annual = pricing?.annual ?? 1099;
  const introActive = !!pricing?.intro_active;
  const lockMo = pricing?.intro_lock_months ?? 12;

  return (
    <div className="p-4 sm:p-5 space-y-4 text-left">
      {/* 1. Dynamic, honest "where we are right now" */}
      {phase?.text && (
        <div className={`rounded-[2px] border p-4 ${isSoft ? 'border-ink/15 bg-paper-deep' : 'border-positive/30 bg-positive/5'}`}>
          <div className="flex items-center gap-2 mb-1">
            <TrendingUp className={`w-4 h-4 ${isSoft ? 'text-ink-mute' : 'text-positive'}`} />
            <span className="text-xs font-medium uppercase tracking-wide text-ink-mute">Where we are right now</span>
          </div>
          <p className="text-sm text-ink leading-relaxed">{phase.text}</p>
        </div>
      )}

      {/* 2. The long-term record — the real proof, drawdowns included */}
      <LongTermRecord />

      {/* 3. What we hold right now — COUNTS only, no names */}
      <div className="bg-paper-card rounded-[2px] border border-ink/10 p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-display text-[1.05rem] font-medium text-ink" style={{ fontVariationSettings: '"opsz" 32' }}>
            What we hold right now
          </h4>
          <Lock className="w-3.5 h-3.5 text-ink-mute" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          {[['Preserver', pres], ['Maximizer', maxi]].map(([label, b]) => (
            <div key={label} className="rounded-[2px] bg-paper-deep p-3">
              <div className="text-xs font-medium text-claret">{label}</div>
              <div className="text-2xl font-display text-ink tabular-nums leading-tight mt-0.5">
                {b?.holdings_count ?? '—'}<span className="text-sm text-ink-mute font-sans"> names</span>
              </div>
              <div className="text-xs text-ink-mute mt-0.5">
                {b?.new_today ? `${b.new_today} new today` : 'no new today'}
                {b && b.book_return_pct !== null && b.book_return_pct !== undefined && (
                  <> · <span className={b.book_return_pct >= 0 ? 'text-positive' : 'text-negative'}>{pctText(b.book_return_pct)} since launch</span></>
                )}
              </div>
            </div>
          ))}
        </div>
        <p className="text-xs text-ink-mute mt-3 leading-relaxed">
          Our live track began mid-June — right into this soft patch — so the book is down a little
          so far. Same engine as the 21-year record above; we just started the clock at a dip.
          {' '}<span className="text-ink font-medium">Unlock to see every name, weight, and exit.</span>
        </p>
      </div>

      {/* 4. Recent named winners — closed/resolved catches (proof, not actionable) */}
      {winners.length > 0 && (
        <div className="bg-paper-card rounded-[2px] border border-ink/10 p-4">
          <h4 className="font-display text-[1.05rem] font-medium text-ink mb-1" style={{ fontVariationSettings: '"opsz" 32' }}>
            Recent catches
          </h4>
          <p className="text-xs text-ink-mute mb-2">Closed walk-forward trades — names shown after the trade finished.</p>
          <div className="divide-y divide-ink/5">
            {winners.slice(0, 6).map((w, i) => (
              <div key={i} className="flex items-center justify-between py-1.5 text-sm">
                <span className="font-medium text-ink">{w.symbol}</span>
                <span className="text-xs text-ink-mute">{w.days_held ? `${w.days_held}d` : ''}</span>
                <span className="tabular-nums font-medium text-positive">{pctText(w.would_be_return)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 5. Ticker-free market read */}
      {read && (
        <div className="text-xs text-ink-mute italic leading-relaxed px-1">{read}</div>
      )}

      {/* 6. Upgrade CTA */}
      <div className="pt-1 space-y-2">
        {user ? (
          <>
            <button
              onClick={() => onSubscribe('monthly')}
              disabled={upgradeLoading}
              className="w-full px-4 py-2.5 bg-ink text-paper font-medium rounded-[2px] hover:bg-claret transition-all disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {upgradeLoading ? 'Loading…' : <>Unlock live signals — ${introMo}/mo{introActive ? ' introductory' : ''} <ArrowRight className="w-4 h-4" /></>}
            </button>
            <button
              onClick={() => onSubscribe('annual')}
              disabled={upgradeLoading}
              className="w-full px-4 py-2 text-sm text-ink-mute hover:text-ink font-medium transition-colors"
            >
              or ${annual.toLocaleString()}/year{introActive ? ` · rate locked ${lockMo} mo` : ''} · 30-day money-back
            </button>
          </>
        ) : (
          <button
            onClick={onSignIn}
            className="w-full px-4 py-2.5 bg-ink text-paper font-medium rounded-[2px] hover:bg-claret transition-all flex items-center justify-center gap-2"
          >
            <LogIn className="w-4 h-4" /> Create a free account
          </button>
        )}
      </div>
    </div>
  );
}
