/**
 * Portfolio — the two SERVED books, mobilized: Preserver + Maximizer (Core dropped).
 *
 * Top: a Cascade Guard status card (governs both books — "clear" or "entries
 * paused until X"). Then a section per book: equity + holdings count + regime,
 * and a compact card per open holding:
 *   line 1  SYM .................... +10.8%   (P&L since entry)
 *   line 2  N sh · $value ......... exit signal   (Preserver: 30% trail · Maximizer: day X/29)
 *   line 3  entry $X → $Y                          (+ today's live move if available)
 *
 * Open holdings come from the tier-books `fills` (unrealized buy rows carry
 * current_price + pnl_pct + days_held). Current price + daily change overlay
 * from /api/quotes/live (polled 30s), falling back to the fill's current_price.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, RefreshControl, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  CascadeGuard,
  LiveQuote,
  TierBook,
  TierBooks,
  TierFillRow,
  getCurrentRegime,
  getLiveQuotes,
  getTierBooks,
} from '@/services/admin';
import StatCard from '@/components/StatCard';
import Section from '@/components/Section';
import { Fonts, FontSize, Palette, Radii, Regime, Spacing } from '@/constants/theme';

const money = (n: any) => (typeof n === 'number' ? '$' + Math.round(n).toLocaleString('en-US') : '—');
const price2 = (n: any) => (typeof n === 'number' ? '$' + n.toFixed(2) : '—');
const pct = (n: any) => (typeof n === 'number' ? `${n >= 0 ? '+' : ''}${n.toFixed(1)}%` : '—');
const signedMoney = (n: any) =>
  typeof n === 'number' ? `${n >= 0 ? '+' : '−'}$${Math.abs(Math.round(n)).toLocaleString('en-US')}` : '—';
const toneColor = (n: any) =>
  typeof n === 'number' ? (n >= 0 ? Palette.positive : Palette.negative) : Palette.inkLight;
const shortDate = (iso: any) => {
  if (!iso) return undefined;
  try {
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return undefined;
  }
};

const HOLD_DAYS = 29;

// Open holdings for a book = its unrealized buy fills (they carry current price,
// P&L, and days-held). Deduped by symbol (latest wins).
function openHoldings(fills?: TierFillRow[]): TierFillRow[] {
  const rows = (fills || []).filter((f) => f.side === 'buy' && f.unrealized);
  const bySym = new Map<string, TierFillRow>();
  for (const r of rows) bySym.set(r.symbol, r);
  return Array.from(bySym.values());
}

function CascadeGuardCard({ cg }: { cg?: CascadeGuard | null }) {
  if (!cg) return null;
  if (cg.enabled === false) {
    return (
      <View style={[styles.cgCard, { borderLeftColor: Palette.inkLight }]}>
        <Text style={styles.cgTitle}>Cascade Guard · off</Text>
        <Text style={styles.cgSub}>Circuit breaker disabled.</Text>
      </View>
    );
  }
  if (cg.paused) {
    const syms = (cg.last_stopped_symbols || []).join(', ');
    return (
      <View style={[styles.cgCard, { borderLeftColor: Palette.claret }]}>
        <Text style={[styles.cgTitle, { color: Palette.claret }]}>
          Cascade Guard · entries paused
        </Text>
        <Text style={styles.cgSub}>
          Until {cg.pause_until || '—'}
          {syms ? ` · triggered by ${syms}` : ''}
        </Text>
      </View>
    );
  }
  return (
    <View style={[styles.cgCard, { borderLeftColor: Palette.positive }]}>
      <Text style={[styles.cgTitle, { color: Palette.positive }]}>Cascade Guard · clear</Text>
      <Text style={styles.cgSub}>
        {cg.threshold_stops ?? 3} same-day stops → {cg.pause_days ?? 10}-day pause
        {cg.last_triggered_at ? ` · last fired ${shortDate(cg.last_triggered_at) || cg.last_triggered_at}` : ''}
      </Text>
    </View>
  );
}

// Fallback for pre-deploy backends that don't yet send weight-sorted `positions`:
// derive a position-shaped list from the (date-ordered) unrealized buy fills.
function positionsFor(book?: TierBook, fills?: TierFillRow[]) {
  if (book?.positions && book.positions.length) return book.positions;
  return openHoldings(fills).map((h) => ({
    symbol: h.symbol, price: h.price ?? null, pnl_pct: h.pnl_pct ?? null,
    entry_price: h.price ?? null, days_held: h.days_held ?? null,
    weight_pct: null, value: null, hold_days: null, days_left: null,
    trailing_stop_level: null, high_water_mark: null,
  }));
}

function BookSection({
  tierKey,
  book,
  fills,
  quotes,
}: {
  tierKey: 'preserver' | 'maximizer';
  book?: TierBook;
  fills?: TierFillRow[];
  quotes: Record<string, LiveQuote>;
}) {
  const isMax = tierKey === 'maximizer';
  const label = isMax ? 'Maximizer' : 'Preserver';
  const accent = isMax ? Palette.claret : Palette.positive;
  const positions = positionsFor(book, fills); // weight-sorted (matches the subscriber webapp)
  const txns = (fills || []).slice(0, 6);       // date-sorted transaction log (distinct view)

  return (
    <Section title={`${isMax ? '◆ ' : ''}${label} book`} hint={book?.as_of ? `as of ${book.as_of}` : ''}>
      <View style={styles.bookHead}>
        <View style={[styles.tierPill, { borderColor: accent }]}>
          <Text style={[styles.tierPillText, { color: accent }]}>{label.toUpperCase()}</Text>
        </View>
        <View style={styles.bookEquityBox}>
          <Text style={styles.bookEquity}>{money(book?.equity)}</Text>
          {/* Clarify: this is the MODEL book's mark-to-market from a $100k inception — NOT a
              per-user balance (which we don't track; signals-only). */}
          <Text style={styles.bookEquitySub}>
            model book · from $100k{typeof book?.return_pct === 'number' ? ` · ${pct(book.return_pct)}` : ''}
          </Text>
        </View>
        <Text style={styles.bookHeld}>
          {(book?.held ?? positions.length) || 0} {(book?.held ?? positions.length) === 1 ? 'name' : 'names'}
        </Text>
      </View>

      {positions.length === 0 ? (
        <Text style={styles.empty}>
          {isMax ? 'No open breakouts — hunting resumes in rotating-bull.' : 'Flat — no open positions.'}
        </Text>
      ) : (
        positions.map((p, i) => {
          const q = quotes[p.symbol];
          const entry = typeof p.entry_price === 'number' ? p.entry_price : undefined;
          const cur = q?.price ?? p.price ?? entry;
          const plPct =
            typeof entry === 'number' && entry > 0 && typeof cur === 'number'
              ? (cur / entry - 1) * 100
              : p.pnl_pct ?? undefined;
          const todayPct = q?.change_pct;
          const daysLeft =
            typeof p.days_left === 'number'
              ? p.days_left
              : typeof p.days_held === 'number'
                ? Math.max(0, (p.hold_days || HOLD_DAYS) - p.days_held)
                : undefined;
          // Exit signal: Maximizer = day X/29 clock; Preserver = 30% trailing stop.
          const exit = isMax
            ? typeof p.days_held === 'number'
              ? `day ${p.days_held}/${p.hold_days || HOLD_DAYS} · ~${daysLeft}d`
              : 'breakout · ~29d hold'
            : typeof p.trailing_stop_level === 'number'
              ? `30% trail · stop ${price2(p.trailing_stop_level)}`
              : '30% trailing stop';
          const wt = typeof p.weight_pct === 'number' ? `${Math.round(p.weight_pct)}% of book` : null;
          return (
            <View key={`${p.symbol}-${i}`} style={styles.posRow}>
              <View style={styles.posLine}>
                <Text style={styles.sym}>{p.symbol}</Text>
                <Text style={[styles.plPct, { color: toneColor(plPct) }]}>{pct(plPct)}</Text>
              </View>
              <View style={styles.posLine}>
                <Text style={styles.posMeta} numberOfLines={1}>
                  {[wt, money(p.value)].filter(Boolean).join(' · ')}
                </Text>
                <Text style={[styles.exit, { color: accent }]} numberOfLines={1}>
                  {exit}
                </Text>
              </View>
              <Text style={styles.posPrices} numberOfLines={1}>
                {`entry ${price2(entry)} → ${price2(cur)}`}
                {q && todayPct != null ? `   ·   today ${pct(todayPct)}` : ''}
              </Text>
            </View>
          );
        })
      )}

      {txns.length > 0 && (
        <View style={styles.txnBlock}>
          <Text style={styles.txnHead}>Recent transactions</Text>
          {txns.map((t, i) => (
            <View key={`${t.symbol}-${t.fill_date}-${i}`} style={styles.txnRow}>
              <Text style={styles.txnSym} numberOfLines={1}>
                <Text style={{ color: t.side === 'sell' ? Palette.claret : Palette.positive }}>
                  {t.side === 'sell' ? 'SELL' : 'BUY '}
                </Text>
                {'  '}
                {t.symbol}
              </Text>
              <Text style={styles.txnMeta} numberOfLines={1}>
                {[shortDate(t.fill_date), price2(t.price), t.reason].filter(Boolean).join(' · ')}
              </Text>
            </View>
          ))}
        </View>
      )}
    </Section>
  );
}

export default function Portfolio() {
  const [tb, setTb] = useState<TierBooks | null>(null);
  const [regime, setRegime] = useState<any>(null);
  const [quotes, setQuotes] = useState<Record<string, LiveQuote>>({});
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    const [t, r] = await Promise.allSettled([getTierBooks(), getCurrentRegime()]);
    if (t.status === 'fulfilled') setTb(t.value);
    else setError('Could not load the books.');
    if (r.status === 'fulfilled') setRegime(r.value);
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Poll live quotes for the union of both books' held symbols.
  const symbolsKey = useMemo(() => {
    const syms = new Set<string>();
    for (const tier of ['preserver', 'maximizer'] as const) {
      for (const p of positionsFor(tb?.books?.[tier], tb?.fills?.[tier])) if (p.symbol) syms.add(p.symbol);
    }
    return Array.from(syms).join(',');
  }, [tb]);

  useEffect(() => {
    if (!symbolsKey) return;
    let active = true;
    const fetchQuotes = async () => {
      try {
        const q = await getLiveQuotes(symbolsKey.split(','));
        if (active) setQuotes(q);
      } catch {
        /* best-effort */
      }
    };
    fetchQuotes();
    const id = setInterval(fetchQuotes, 30000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [symbolsKey]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  }, [load]);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={Palette.claret} />
      </View>
    );
  }

  const regimeName = (regime?.regime || regime?.current_regime || regime?.name || tb?.books?.preserver?.regime || '')
    .toString();
  const regimeColor = Regime[regimeName] || Palette.inkLight;

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={Palette.claret} />}
      >
        {regimeName ? (
          <View style={[styles.regime, { borderColor: regimeColor }]}>
            <View style={[styles.dot, { backgroundColor: regimeColor }]} />
            <Text style={styles.regimeText}>{regimeName.replace(/_/g, ' ')}</Text>
          </View>
        ) : null}

        <CascadeGuardCard cg={tb?.cascade_guard} />

        {/* Book equities at a glance — model book MTM from $100k inception (not a per-user balance). */}
        <View style={styles.grid}>
          <StatCard
            label="Preserver"
            value={money(tb?.books?.preserver?.equity)}
            sub={typeof tb?.books?.preserver?.return_pct === 'number'
              ? `from $100k · ${pct(tb.books.preserver.return_pct)}`
              : 'from $100k'}
          />
          <StatCard
            label="Maximizer"
            value={money(tb?.books?.maximizer?.equity)}
            sub={typeof tb?.books?.maximizer?.return_pct === 'number'
              ? `from $100k · ${pct(tb.books.maximizer.return_pct)}`
              : 'from $100k'}
          />
        </View>

        <BookSection tierKey="preserver" book={tb?.books?.preserver} fills={tb?.fills?.preserver} quotes={quotes} />
        <BookSection tierKey="maximizer" book={tb?.books?.maximizer} fills={tb?.fills?.maximizer} quotes={quotes} />

        {error ? <Text style={styles.error}>{error}</Text> : null}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: Palette.paper },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: Palette.paper },
  content: { padding: Spacing.lg, paddingBottom: Spacing.xxl },
  regime: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderRadius: Radii.pill,
    paddingHorizontal: Spacing.md,
    paddingVertical: 6,
    marginBottom: Spacing.md,
  },
  dot: { width: 8, height: 8, borderRadius: 4, marginRight: Spacing.sm },
  regimeText: {
    fontFamily: Fonts.body.semibold,
    fontSize: FontSize.sm,
    color: Palette.ink,
    textTransform: 'capitalize',
    letterSpacing: 0.4,
  },
  cgCard: {
    backgroundColor: Palette.paperCard,
    borderWidth: 1,
    borderColor: Palette.rule,
    borderLeftWidth: 4,
    borderRadius: Radii.lg,
    padding: Spacing.md,
    marginBottom: Spacing.lg,
  },
  cgTitle: { fontFamily: Fonts.display.medium, fontSize: FontSize.md, color: Palette.ink },
  cgSub: { fontFamily: Fonts.mono.regular, fontSize: FontSize.xs, color: Palette.inkMute, marginTop: 3 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.md, marginBottom: Spacing.xl },
  bookHead: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.sm,
    gap: Spacing.sm,
  },
  tierPill: { borderWidth: 1, borderRadius: Radii.pill, paddingHorizontal: Spacing.sm, paddingVertical: 2 },
  tierPillText: { fontFamily: Fonts.body.medium, fontSize: 9, letterSpacing: 0.6 },
  bookEquityBox: { flexShrink: 1 },
  bookEquity: { fontFamily: Fonts.display.semibold, fontSize: FontSize.lg, color: Palette.ink },
  bookEquitySub: { fontFamily: Fonts.mono.regular, fontSize: 10, color: Palette.inkLight, marginTop: 1 },
  bookHeld: { fontFamily: Fonts.mono.regular, fontSize: FontSize.xs, color: Palette.inkLight, marginLeft: 'auto' },
  txnBlock: { marginTop: Spacing.md, paddingTop: Spacing.sm, borderTopWidth: 1, borderTopColor: Palette.rule },
  txnHead: {
    fontFamily: Fonts.body.medium, fontSize: 10, letterSpacing: 1, textTransform: 'uppercase',
    color: Palette.inkMute, marginBottom: Spacing.xs,
  },
  txnRow: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between', paddingVertical: 3 },
  txnSym: { fontFamily: Fonts.mono.medium, fontSize: FontSize.xs, color: Palette.ink },
  txnMeta: { fontFamily: Fonts.mono.regular, fontSize: FontSize.xs, color: Palette.inkLight, marginLeft: Spacing.sm },
  posRow: { paddingVertical: Spacing.sm, borderBottomWidth: 1, borderBottomColor: Palette.rule },
  posLine: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between' },
  sym: { fontFamily: Fonts.display.medium, fontSize: FontSize.lg, color: Palette.ink },
  plPct: { fontFamily: Fonts.mono.medium, fontSize: FontSize.md },
  posMeta: { fontFamily: Fonts.mono.regular, fontSize: FontSize.xs, color: Palette.inkMute, marginTop: 2, flex: 1, marginRight: Spacing.sm },
  exit: { fontFamily: Fonts.mono.regular, fontSize: FontSize.xs },
  posPrices: { fontFamily: Fonts.mono.regular, fontSize: FontSize.xs, color: Palette.inkLight, marginTop: 3 },
  empty: { fontFamily: Fonts.body.regular, fontSize: FontSize.sm, color: Palette.inkMute, paddingVertical: Spacing.sm },
  error: { fontFamily: Fonts.body.medium, fontSize: FontSize.sm, color: Palette.negative, marginTop: Spacing.md },
});
