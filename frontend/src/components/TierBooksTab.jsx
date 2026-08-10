import React, { useState, useEffect, useCallback } from 'react';
import { RefreshCw } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Admin 3-book compare: Core / Preserver / Maximizer equity side-by-side + each book's
// STR (transaction/fill log). Reads GET /api/admin/tier-books. (Jul 24 2026)
const TIERS = [
  { id: 'core', label: 'Core', sub: 't30v · internal model book' },
  { id: 'preserver', label: 'Preserver', sub: 't30v + capitulation overlay' },
  { id: 'maximizer', label: 'Maximizer', sub: 'breakout book + vol-target' },
];

export default function TierBooksTab({ fetchWithAuth }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [quotes, setQuotes] = useState({});
  const [showCore, setShowCore] = useState(false);  // Core STR collapsed by default

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetchWithAuth(`${API_URL}/api/admin/tier-books?limit=60`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e) {
      setError(e.message || 'Failed to load');
    } finally {
      setLoading(false);
    }
  }, [fetchWithAuth]);

  useEffect(() => { load(); }, [load]);

  // Poll live quotes for the union of held symbols → intraday reprice (equity + open P&L).
  useEffect(() => {
    if (!data?.books) return;
    const syms = new Set();
    Object.values(data.books).forEach(b => (b.holdings || []).forEach(h => h.symbol && syms.add(h.symbol)));
    if (!syms.size) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetchWithAuth(`${API_URL}/api/quotes/live?symbols=${[...syms].join(',')}`);
        if (!res.ok) return;
        const j = await res.json();
        if (!cancelled) setQuotes(j.quotes || {});
      } catch { /* leave EOD values */ }
    };
    poll();
    const iv = setInterval(poll, 30000);
    return () => { cancelled = true; clearInterval(iv); };
  }, [data, fetchWithAuth]);

  if (loading) return <div className="p-8 text-center text-ink-mute"><RefreshCw className="w-5 h-5 animate-spin inline" /> Loading tier books…</div>;
  if (error) return <div className="p-8 text-center text-claret">Error: {error} <button onClick={load} className="underline ml-2">retry</button></div>;
  if (!data) return null;

  const usd = (v) => v == null ? '—' : `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  const pnl = (v) => v == null ? '' : `${v >= 0 ? '+' : ''}$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  const books = data.books || {};
  const fills = data.fills || {};

  // equity_live = equity_eod + Σ shares×(live − eod). Symbols without a live quote add 0.
  const liveEquity = (b) => {
    if (!b || b.equity == null || !Array.isArray(b.holdings)) return { value: b?.equity, live: false };
    let delta = 0, any = false;
    for (const h of b.holdings) {
      const q = quotes[h.symbol];
      if (q && q.price && h.eod_price) { delta += h.shares * (q.price - h.eod_price); any = true; }
    }
    return { value: any ? b.equity + delta : b.equity, live: any };
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-ink">Tier Books — Core / Preserver / Maximizer</h2>
        <button onClick={load} className="text-sm text-ink-mute hover:text-ink flex items-center gap-1">
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      {/* Equity side-by-side */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {TIERS.map((t) => {
          const b = books[t.id] || {};
          return (
            <div key={t.id} className="border border-rule rounded p-4 bg-white">
              <div className="flex items-baseline justify-between">
                <span className="font-semibold text-ink">{t.label}</span>
                {b.regime && <span className="text-[0.6rem] uppercase tracking-wide text-ink-mute">{b.regime}</span>}
              </div>
              <div className="text-xs text-ink-mute mb-2">{t.sub}</div>
              {(() => { const le = liveEquity(b); return (
                <>
                  <div className="text-2xl font-semibold text-ink">
                    {usd(le.value)}
                    {le.live && <span title="Live intraday" className="ml-2 inline-block w-2 h-2 rounded-full bg-positive align-middle" />}
                  </div>
                  <div className="text-[0.7rem] text-ink-mute mt-1">
                    {le.live ? 'live · intraday' : `as of ${b.as_of || '—'}`}{b.held != null ? ` · ${b.held} held` : ''}
                  </div>
                </>
              ); })()}
              {b.note && <div className="text-[0.68rem] text-ink-mute mt-2 leading-snug italic">{b.note}</div>}
            </div>
          );
        })}
      </div>

      {/* STR fill logs per tier. Core is collapsed by default — Preserver + Maximizer are
          the books of interest. */}
      {TIERS.map((t) => {
        const rows = fills[t.id] || [];
        const collapsed = t.id === 'core' && !showCore;
        return (
          <div key={t.id}>
            <div className="flex items-baseline justify-between border-b border-rule pb-2 mb-2">
              <h3 className="font-semibold text-ink">{t.label} — transaction log (STR)</h3>
              <div className="flex items-baseline gap-3">
                <span className="text-xs text-ink-mute">{rows.length} fills</span>
                {t.id === 'core' && (
                  <button onClick={() => setShowCore(v => !v)} className="text-xs text-ink-mute hover:text-ink underline">
                    {showCore ? 'hide' : 'show'}
                  </button>
                )}
              </div>
            </div>
            {collapsed ? (
              <div className="text-xs text-ink-mute py-1.5 italic">Collapsed — click “show” to view the Core log.</div>
            ) : rows.length === 0 ? (
              <div className="text-sm text-ink-mute py-3">
                No fills logged yet{t.id === 'core' ? ' (Core trades live in the model portfolio).' : '.'}
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm" style={{ fontFeatureSettings: '"tnum"' }}>
                  <thead>
                    <tr className="text-[0.6rem] uppercase tracking-wide text-ink-mute border-b border-rule">
                      <th className="text-left py-1.5 px-2">Date</th>
                      <th className="text-left py-1.5 px-2">Symbol</th>
                      <th className="text-left py-1.5 px-2">Side</th>
                      <th className="text-right py-1.5 px-2">Shares</th>
                      <th className="text-right py-1.5 px-2">Entry</th>
                      <th className="text-right py-1.5 px-2">Current</th>
                      <th className="text-right py-1.5 px-2 hidden sm:table-cell">Gross</th>
                      <th className="text-left py-1.5 px-2 hidden md:table-cell">Reason</th>
                      <th className="text-right py-1.5 px-2 hidden md:table-cell">Days</th>
                      <th className="text-right py-1.5 px-2">P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r, i) => {
                      // Live-first current price + P&L for open rows; else server EOD mark; closed = realized.
                      const isOpenBuy = r.unrealized && r.side === 'buy';
                      const live = quotes[r.symbol]?.price || null;
                      const cur = isOpenBuy ? (live || r.current_price) : null;
                      let v = r.realized_pnl;
                      let pct = r.pnl_pct;
                      if (isOpenBuy && cur != null && r.price) {
                        v = r.shares * (cur - r.price);
                        pct = (cur - r.price) / r.price * 100;
                      }
                      return (
                        <tr key={`${r.symbol}-${r.fill_date}-${r.side}-${i}`} className="border-b border-rule/50">
                          <td className="py-1.5 px-2 font-mono text-xs text-ink-mute">{r.fill_date || '—'}</td>
                          <td className="py-1.5 px-2 font-medium text-ink">{r.symbol}</td>
                          <td className={`py-1.5 px-2 uppercase text-xs font-medium ${r.side === 'buy' ? 'text-positive' : 'text-claret'}`}>{r.side}</td>
                          <td className="py-1.5 px-2 text-right font-mono">{r.shares != null ? Number(r.shares).toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—'}</td>
                          <td className="py-1.5 px-2 text-right font-mono">{r.price != null ? `$${Number(r.price).toFixed(2)}` : '—'}</td>
                          <td className="py-1.5 px-2 text-right font-mono">
                            {cur != null ? (
                              <span>${Number(cur).toFixed(2)}{live && <span title="Live intraday" className="ml-1 inline-block w-1.5 h-1.5 rounded-full bg-positive align-middle" />}</span>
                            ) : '—'}
                          </td>
                          <td className="py-1.5 px-2 text-right font-mono hidden sm:table-cell">{usd(r.gross)}</td>
                          <td className="py-1.5 px-2 text-xs text-ink-mute hidden md:table-cell">
                            {r.reason}{r.source ? ` · ${r.source}` : ''}{r.vol_scale != null ? ` · vs ${Number(r.vol_scale).toFixed(2)}` : ''}
                          </td>
                          <td className="py-1.5 px-2 text-right font-mono text-xs text-ink-mute hidden md:table-cell">{r.days_held != null ? `${r.days_held}d` : '—'}</td>
                          <td className={`py-1.5 px-2 text-right font-mono ${(v ?? 0) >= 0 ? 'text-positive' : 'text-claret'}`} title={r.unrealized ? 'Unrealized (live/EOD mark)' : 'Realized'}>
                            {v != null ? (
                              <span>{r.unrealized ? '~' : ''}{pnl(v)}{pct != null && <span className="text-xs ml-1">({pct >= 0 ? '+' : ''}{Number(pct).toFixed(1)}%)</span>}</span>
                            ) : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
