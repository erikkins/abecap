import React, { useState } from 'react';

// Capital-scaled MIRROR of a tier's model book. The user sets their capital once; we scale
// the book's positions to it (implied_shares = book_shares x capital/book_value) so their
// portfolio auto-mirrors the book with zero per-trade entry. Maximizer = breakout book
// (day-X/29 exits); Preserver = t30v book (30% trailing). (Jul 24 2026)
export default function TierBookView({ book, onSetCapital, onRowClick }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(String(book?.capital ?? 100000));
  const [saving, setSaving] = useState(false);

  if (!book) return null;
  const isMax = book.tier === 'maximizer';
  const usd0 = (v) => v == null ? '—' : `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  const sh = (v) => v == null ? '—' : Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 });

  const save = async () => {
    const val = Number(String(draft).replace(/[^0-9.]/g, ''));
    if (!(val >= 100 && val <= 10000000)) { setEditing(false); return; }
    setSaving(true);
    try { await onSetCapital(val); } finally { setSaving(false); setEditing(false); }
  };

  return (
    <div className="border border-rule rounded bg-paper-card mb-6">
      {/* Header */}
      <div className="px-4 sm:px-5 py-4 border-b border-rule flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="font-display text-[1.2rem] font-medium text-ink tracking-tight flex items-center gap-2" style={{ fontVariationSettings: '"opsz" 48' }}>
            {isMax && <span className="text-claret text-[0.7rem]">◆</span>}
            Your {isMax ? 'Maximizer' : 'Preserver'} Book
          </h2>
          <p className="font-display italic text-[0.82rem] text-ink-mute mt-0.5" style={{ fontVariationSettings: '"opsz" 24' }}>
            Auto-mirrored to the model book — no manual entry. {isMax ? 'Breakouts, held ~29 trading days.' : '30% trailing stop, let winners run.'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {book.new_today > 0 && (
            <span className="font-mono text-[0.62rem] uppercase tracking-[0.14em] bg-claret text-paper px-2 py-0.5 rounded">
              {book.new_today} new today
            </span>
          )}
          {book.regime && <span className="font-mono text-[0.6rem] uppercase tracking-[0.16em] text-ink-mute">{book.regime}</span>}
        </div>
      </div>

      {/* Capital + summary */}
      <div className="px-4 sm:px-5 py-4 grid grid-cols-2 sm:grid-cols-4 gap-4 border-b border-rule">
        <div>
          <div className="text-[0.6rem] uppercase tracking-[0.16em] text-ink-mute mb-1">Your capital</div>
          {editing ? (
            <div className="flex items-center gap-1">
              <input
                autoFocus value={draft} onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && save()}
                className="w-24 border border-ink rounded px-2 py-1 font-mono text-[0.95rem]"
              />
              <button onClick={save} disabled={saving} className="text-[0.7rem] uppercase tracking-wide bg-ink text-paper px-2 py-1 rounded">{saving ? '…' : 'Set'}</button>
            </div>
          ) : (
            <button onClick={() => { setDraft(String(book.capital)); setEditing(true); }} className="font-mono text-[1.15rem] text-ink hover:text-claret transition-colors">
              {usd0(book.capital)} <span className="text-[0.6rem] text-ink-mute">edit</span>
            </button>
          )}
        </div>
        <div>
          <div className="text-[0.6rem] uppercase tracking-[0.16em] text-ink-mute mb-1">Invested</div>
          <div className="font-mono text-[1.15rem] text-ink">{usd0(book.invested_value)}</div>
          <div className="font-mono text-[0.6rem] text-ink-mute">{book.invested_pct}% of book</div>
        </div>
        <div>
          <div className="text-[0.6rem] uppercase tracking-[0.16em] text-ink-mute mb-1">Cash</div>
          <div className="font-mono text-[1.15rem] text-ink">{usd0(book.cash_value)}</div>
          <div className="font-mono text-[0.6rem] text-ink-mute">{book.cash_pct}%</div>
        </div>
        <div>
          <div className="text-[0.6rem] uppercase tracking-[0.16em] text-ink-mute mb-1">Holdings</div>
          <div className="font-mono text-[1.15rem] text-ink">{book.holdings?.length || 0}</div>
          <div className="font-mono text-[0.6rem] text-ink-mute">names</div>
        </div>
      </div>

      {/* Holdings table */}
      <div className="overflow-x-auto">
        <table className="w-full" style={{ fontFeatureSettings: '"tnum"' }}>
          <thead>
            <tr className="text-[0.58rem] uppercase tracking-[0.16em] text-ink-mute border-b border-rule">
              <th className="text-left py-2 px-3 sm:px-5">Symbol</th>
              <th className="text-right py-2 px-3 hidden sm:table-cell">Weight</th>
              <th className="text-right py-2 px-3">Price</th>
              <th className="text-right py-2 px-3">Your shares</th>
              <th className="text-right py-2 px-3">Your value</th>
              <th className="text-left py-2 px-3">Exit</th>
              <th className="text-right py-2 px-3 hidden sm:table-cell">P&L</th>
            </tr>
          </thead>
          <tbody>
            {(book.holdings || []).map((h) => (
              <tr
                key={h.symbol}
                onClick={() => onRowClick && onRowClick(h)}
                className={`border-b border-rule/50 ${onRowClick ? 'cursor-pointer hover:bg-paper-deep transition-colors' : ''}`}
              >
                <td className="py-2.5 px-3 sm:px-5">
                  <span className="font-display text-[1rem] font-medium text-ink" style={{ fontVariationSettings: '"opsz" 32' }}>{h.symbol}</span>
                  {h.is_new && (
                    <span className="font-mono text-[0.55rem] uppercase tracking-[0.14em] text-claret border border-claret/40 px-1 py-0.5 ml-2 align-middle">New</span>
                  )}
                </td>
                <td className="py-2.5 px-3 text-right font-mono text-[0.82rem] text-ink-mute hidden sm:table-cell">{h.weight_pct}%</td>
                <td className="py-2.5 px-3 text-right font-mono text-[0.82rem]">${h.price?.toFixed(2)}</td>
                <td className="py-2.5 px-3 text-right font-mono text-[0.9rem] text-ink">{sh(h.implied_shares)}</td>
                <td className="py-2.5 px-3 text-right font-mono text-[0.9rem] text-ink">{usd0(h.implied_value)}</td>
                <td className="py-2.5 px-3 font-mono text-[0.72rem] text-claret whitespace-nowrap">
                  {h.exit_rule === 'hold'
                    ? `day ${h.days_held}/${h.hold_days} · ~${h.days_left}d`
                    : `30% trail · $${h.trailing_stop_level?.toFixed(2)}`}
                </td>
                <td className={`py-2.5 px-3 text-right font-mono text-[0.82rem] hidden sm:table-cell ${h.pnl_pct >= 0 ? 'text-positive' : 'text-negative'}`}>
                  {h.pnl_pct >= 0 ? '+' : ''}{h.pnl_pct}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div className="px-4 sm:px-5 py-3 border-t border-rule flex flex-wrap items-center justify-between gap-2 text-[0.7rem] text-ink-mute">
        <span>As of {book.as_of || '—'} · mirror the book to match its results — partial-following diverges.</span>
        <span className="font-display italic" style={{ fontVariationSettings: '"opsz" 24' }}>Signals only — execute via your broker</span>
      </div>
    </div>
  );
}
