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
  const holds = openHoldings(fills);
  const regimeName = (book?.regime || '').toString();

  return (
    <Section title={`${isMax ? '◆ ' : ''}${label} book`} hint={book?.as_of ? `as of ${book.as_of}` : ''}>
      <View style={styles.bookHead}>
        <View style={[styles.tierPill, { borderColor: accent }]}>
          <Text style={[styles.tierPillText, { color: accent }]}>{label.toUpperCase()}</Text>
        </View>
        <Text style={styles.bookEquity}>{money(book?.equity)}</Text>
        <Text style={styles.bookHeld}>
          {(book?.held ?? holds.length) || 0} {(book?.held ?? holds.length) === 1 ? 'name' : 'names'}
        </Text>
      </View>
      {holds.length === 0 ? (
        <Text style={styles.empty}>
          {isMax ? 'No open breakouts — hunting resumes in rotating-bull.' : 'Flat — no open positions.'}
        </Text>
      ) : (
        holds.map((h, i) => {
          const q = quotes[h.symbol];
          const entry = typeof h.price === 'number' ? h.price : undefined;
          const cur = q?.price ?? h.current_price ?? entry;
          const shares = h.shares;
          const plPct =
            typeof entry === 'number' && entry > 0 && typeof cur === 'number'
              ? (cur / entry - 1) * 100
              : h.pnl_pct ?? undefined;
          const value = typeof cur === 'number' && typeof shares === 'number' ? cur * shares : undefined;
          const todayPct = q?.change_pct;
          const todayDol = q && typeof shares === 'number' ? q.change * shares : undefined;
          // Exit signal: Maximizer = day X/29 clock; Preserver = 30% trailing stop.
          const daysHeldN = typeof h.days_held === 'number' ? h.days_held : undefined;
          const exit = isMax
            ? daysHeldN != null
              ? `day ${daysHeldN}/${HOLD_DAYS} · ~${Math.max(0, HOLD_DAYS - daysHeldN)}d`
              : 'breakout · ~29d hold'
            : '30% trailing stop';
          return (
            <View key={`${h.symbol}-${i}`} style={styles.posRow}>
              <View style={styles.posLine}>
                <Text style={styles.sym}>{h.symbol}</Text>
                <Text style={[styles.plPct, { color: toneColor(plPct) }]}>{pct(plPct)}</Text>
              </View>
              <View style={styles.posLine}>
                <Text style={styles.posMeta} numberOfLines={1}>
                  {[typeof shares === 'number' ? `${shares.toFixed(2)} sh` : null, money(value)]
                    .filter(Boolean)
                    .join(' · ')}
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
      for (const h of openHoldings(tb?.fills?.[tier])) if (h.symbol) syms.add(h.symbol);
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

        {/* Book equities at a glance */}
        <View style={styles.grid}>
          <StatCard label="Preserver" value={money(tb?.books?.preserver?.equity)} sub="protect" />
          <StatCard label="Maximizer" value={money(tb?.books?.maximizer?.equity)} sub="grow" />
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
  bookEquity: { fontFamily: Fonts.display.semibold, fontSize: FontSize.lg, color: Palette.ink },
  bookHeld: { fontFamily: Fonts.mono.regular, fontSize: FontSize.xs, color: Palette.inkLight, marginLeft: 'auto' },
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
