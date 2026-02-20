/**
 * Dashboard — main screen showing market regime, signal summary, and model portfolio.
 * Tabbed layout: Signals | Positions | Missed
 */

import React, { useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useDashboard, Position, MissedOpportunity } from '@/hooks/useSignals';
import SignalCard from '@/components/SignalCard';
import RegimeBadge from '@/components/RegimeBadge';
import { Colors, FontSize, Spacing } from '@/constants/theme';

type Tab = 'signals' | 'positions' | 'missed';

export default function DashboardScreen() {
  const { data, isLoading, error, refresh } = useDashboard();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<Tab>('signals');

  if (isLoading && !data) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={Colors.gold} />
      </View>
    );
  }

  if (error && !data) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>{error}</Text>
      </View>
    );
  }

  const signals = data?.buy_signals || [];
  const freshSignals = signals.filter((s) => s.is_fresh);
  const monitoringSignals = signals.filter((s) => !s.is_fresh);
  const freshCount = freshSignals.length;
  const regime = data?.regime_forecast;
  const stats = data?.market_stats;
  const portfolio = data?.model_portfolio;
  const positions = data?.positions_with_guidance || [];
  const missed = data?.missed_opportunities || [];

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          refreshing={isLoading}
          onRefresh={refresh}
          tintColor={Colors.gold}
        />
      }
    >
      {/* Market Regime */}
      {regime && (
        <RegimeBadge regime={regime.current_regime} />
      )}

      {/* Market Stats */}
      {stats && (
        <View style={styles.statsRow}>
          <StatBox
            label="S&P 500"
            value={`$${stats.spy_price?.toFixed(0) ?? '—'}`}
            change={stats.spy_change_pct}
          />
          <StatBox
            label="VIX"
            value={stats.vix_level?.toFixed(1) ?? '—'}
          />
          <StatBox
            label="Signals"
            value={`${signals.length}`}
            sub={freshCount > 0 ? `${freshCount} fresh` : undefined}
          />
        </View>
      )}

      {/* Model Portfolio Summary */}
      {portfolio && (
        <View style={styles.portfolioCard}>
          <Text style={styles.sectionTitle}>Model Portfolio</Text>
          <View style={styles.portfolioStats}>
            <View>
              <Text style={styles.portfolioValue}>
                ${portfolio.total_value?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? '—'}
              </Text>
              <Text style={styles.portfolioLabel}>Total Value</Text>
            </View>
            <View>
              <Text
                style={[
                  styles.portfolioReturn,
                  {
                    color:
                      (portfolio.total_return_pct ?? 0) >= 0
                        ? Colors.green
                        : Colors.red,
                  },
                ]}
              >
                {(portfolio.total_return_pct ?? 0) >= 0 ? '+' : ''}
                {portfolio.total_return_pct?.toFixed(1) ?? '0'}%
              </Text>
              <Text style={styles.portfolioLabel}>Total Return</Text>
            </View>
          </View>
        </View>
      )}

      {/* Segmented Tab Bar */}
      <View style={styles.tabBar}>
        <TabPill label="Signals" tab="signals" active={activeTab} onPress={setActiveTab} />
        <TabPill label="Positions" tab="positions" active={activeTab} onPress={setActiveTab} />
        <TabPill label="Missed" tab="missed" active={activeTab} onPress={setActiveTab} />
      </View>

      {/* Tab Content */}
      {activeTab === 'signals' && (
        <SignalsTab
          freshSignals={freshSignals}
          monitoringSignals={monitoringSignals}
          onSignalPress={(symbol) => router.push(`/signal/${symbol}`)}
        />
      )}
      {activeTab === 'positions' && (
        <PositionsTab positions={positions} />
      )}
      {activeTab === 'missed' && (
        <MissedTab missed={missed} />
      )}
    </ScrollView>
  );
}

/* ── Tab Pill ─────────────────────────────────────────── */

function TabPill({
  label,
  tab,
  active,
  onPress,
}: {
  label: string;
  tab: Tab;
  active: Tab;
  onPress: (t: Tab) => void;
}) {
  const isActive = tab === active;
  return (
    <Pressable
      style={[styles.tabPill, isActive && styles.tabPillActive]}
      onPress={() => onPress(tab)}
    >
      <Text style={[styles.tabPillText, isActive && styles.tabPillTextActive]}>
        {label}
      </Text>
    </Pressable>
  );
}

/* ── Signals Tab ──────────────────────────────────────── */

function SignalsTab({
  freshSignals,
  monitoringSignals,
  onSignalPress,
}: {
  freshSignals: any[];
  monitoringSignals: any[];
  onSignalPress: (symbol: string) => void;
}) {
  const hasAny = freshSignals.length > 0 || monitoringSignals.length > 0;

  if (!hasAny) {
    return (
      <View style={styles.emptyCard}>
        <Text style={styles.emptyText}>No active signals.</Text>
        <Text style={styles.emptySubtext}>Check back after the 4 PM ET scan.</Text>
      </View>
    );
  }

  return (
    <>
      {/* Fresh Signals */}
      <View style={styles.sectionHeader}>
        <View style={styles.sectionDot} />
        <Text style={[styles.sectionTitle, { color: Colors.green }]}>
          Buy Signals ({freshSignals.length})
        </Text>
      </View>
      {freshSignals.length > 0 ? (
        freshSignals.map((signal) => (
          <SignalCard
            key={signal.symbol}
            signal={signal}
            onPress={() => onSignalPress(signal.symbol)}
          />
        ))
      ) : (
        <View style={styles.emptyCard}>
          <Text style={styles.emptyText}>No fresh signals today</Text>
          <Text style={styles.emptySubtext}>
            Monitoring {monitoringSignals.length} strong momentum stock{monitoringSignals.length !== 1 ? 's' : ''} for entry
          </Text>
        </View>
      )}

      {/* Monitoring Signals */}
      {monitoringSignals.length > 0 && (
        <>
          <View style={styles.sectionHeader}>
            <View style={[styles.sectionDot, { backgroundColor: Colors.textMuted }]} />
            <View>
              <Text style={[styles.sectionTitle, { color: Colors.textMuted }]}>
                Monitoring ({monitoringSignals.length})
              </Text>
              <Text style={styles.sectionSubtitle}>
                Strong momentum — watching for fresh entry
              </Text>
            </View>
          </View>
          {monitoringSignals.map((signal) => (
            <SignalCard
              key={signal.symbol}
              signal={signal}
              onPress={() => onSignalPress(signal.symbol)}
            />
          ))}
        </>
      )}
    </>
  );
}

/* ── Positions Tab ────────────────────────────────────── */

function PositionsTab({ positions }: { positions: Position[] }) {
  if (positions.length === 0) {
    return (
      <View style={styles.emptyCard}>
        <Text style={styles.emptyText}>No open positions</Text>
      </View>
    );
  }

  return (
    <>
      {positions.map((pos) => (
        <View key={pos.symbol} style={styles.positionCard}>
          <View style={styles.positionHeader}>
            <Text style={styles.positionSymbol}>{pos.symbol}</Text>
            <Text
              style={[
                styles.positionPnl,
                { color: pos.pnl_pct >= 0 ? Colors.green : Colors.red },
              ]}
            >
              {pos.pnl_pct >= 0 ? '+' : ''}{pos.pnl_pct.toFixed(1)}%
            </Text>
          </View>
          <View style={styles.positionDetails}>
            <View style={styles.positionDetail}>
              <Text style={styles.positionDetailLabel}>Entry</Text>
              <Text style={styles.positionDetailValue}>${pos.entry_price.toFixed(2)}</Text>
            </View>
            <View style={styles.positionDetail}>
              <Text style={styles.positionDetailLabel}>Current</Text>
              <Text style={styles.positionDetailValue}>${pos.current_price.toFixed(2)}</Text>
            </View>
            <View style={styles.positionDetail}>
              <Text style={styles.positionDetailLabel}>High</Text>
              <Text style={styles.positionDetailValue}>${pos.highest_price.toFixed(2)}</Text>
            </View>
            <View style={styles.positionDetail}>
              <Text style={styles.positionDetailLabel}>Shares</Text>
              <Text style={styles.positionDetailValue}>{pos.shares}</Text>
            </View>
          </View>
          {pos.sell_guidance && (
            <View style={styles.guidanceRow}>
              <Text style={styles.guidanceText}>{pos.sell_guidance}</Text>
            </View>
          )}
        </View>
      ))}
    </>
  );
}

/* ── Missed Tab ───────────────────────────────────────── */

function MissedTab({ missed }: { missed: MissedOpportunity[] }) {
  if (missed.length === 0) {
    return (
      <View style={styles.emptyCard}>
        <Text style={styles.emptyText}>No recent missed opportunities</Text>
      </View>
    );
  }

  return (
    <>
      {missed.map((m, i) => (
        <View key={`${m.symbol}-${m.entry_date}-${i}`} style={styles.missedCard}>
          <View style={styles.missedHeader}>
            <Text style={styles.missedSymbol}>{m.symbol}</Text>
            <Text style={[styles.missedReturn, { color: Colors.green }]}>
              +{m.would_be_return.toFixed(1)}%
            </Text>
          </View>
          <View style={styles.missedDetails}>
            <View style={styles.missedDetail}>
              <Text style={styles.missedDetailLabel}>Entry</Text>
              <Text style={styles.missedDetailValue}>
                ${m.entry_price.toFixed(2)} · {m.entry_date}
              </Text>
            </View>
            <View style={styles.missedDetail}>
              <Text style={styles.missedDetailLabel}>Exit</Text>
              <Text style={styles.missedDetailValue}>
                ${m.sell_price.toFixed(2)} · {m.sell_date}
              </Text>
            </View>
            <View style={styles.missedDetail}>
              <Text style={styles.missedDetailLabel}>Held</Text>
              <Text style={styles.missedDetailValue}>{m.days_held}d</Text>
            </View>
          </View>
          <View style={styles.exitReasonRow}>
            <Text style={styles.exitReasonText}>{formatExitReason(m.exit_reason)}</Text>
          </View>
        </View>
      ))}
    </>
  );
}

function formatExitReason(reason: string): string {
  return reason
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/* ── StatBox ──────────────────────────────────────────── */

function StatBox({
  label,
  value,
  change,
  sub,
}: {
  label: string;
  value: string;
  change?: number;
  sub?: string;
}) {
  return (
    <View style={styles.statBox}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
      {change != null && (
        <Text
          style={[
            styles.statChange,
            { color: change >= 0 ? Colors.green : Colors.red },
          ]}
        >
          {change >= 0 ? '+' : ''}
          {change.toFixed(2)}%
        </Text>
      )}
      {sub && <Text style={styles.statSub}>{sub}</Text>}
    </View>
  );
}

/* ── Styles ───────────────────────────────────────────── */

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    padding: Spacing.md,
    gap: Spacing.md,
    paddingBottom: Spacing.xl,
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: Colors.background,
  },
  errorText: {
    color: Colors.red,
    fontSize: FontSize.md,
  },

  // Stats row
  statsRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  statBox: {
    flex: 1,
    backgroundColor: Colors.card,
    borderRadius: 8,
    padding: Spacing.md,
    alignItems: 'center',
  },
  statLabel: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    marginBottom: 4,
  },
  statValue: {
    color: Colors.textPrimary,
    fontSize: FontSize.lg,
    fontWeight: '700',
  },
  statChange: {
    fontSize: FontSize.xs,
    fontWeight: '600',
    marginTop: 2,
  },
  statSub: {
    color: Colors.green,
    fontSize: FontSize.xs,
    marginTop: 2,
  },

  // Portfolio card
  portfolioCard: {
    backgroundColor: Colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.cardBorder,
    padding: Spacing.md,
  },
  portfolioStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: Spacing.sm,
  },
  portfolioValue: {
    color: Colors.textPrimary,
    fontSize: FontSize.xl,
    fontWeight: '700',
    textAlign: 'center',
  },
  portfolioReturn: {
    fontSize: FontSize.xl,
    fontWeight: '700',
    textAlign: 'center',
  },
  portfolioLabel: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    textAlign: 'center',
    marginTop: 4,
  },

  // Tab bar
  tabBar: {
    flexDirection: 'row',
    backgroundColor: Colors.card,
    borderRadius: 10,
    padding: 3,
    gap: 3,
  },
  tabPill: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    alignItems: 'center',
  },
  tabPillActive: {
    backgroundColor: Colors.gold,
  },
  tabPillText: {
    color: Colors.textMuted,
    fontSize: FontSize.sm,
    fontWeight: '600',
  },
  tabPillTextActive: {
    color: Colors.navy,
    fontWeight: '700',
  },

  // Section headers
  sectionTitle: {
    color: Colors.textPrimary,
    fontSize: FontSize.lg,
    fontWeight: '700',
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  sectionDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.green,
  },
  sectionSubtitle: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    marginTop: 2,
  },

  // Empty state
  emptyCard: {
    backgroundColor: Colors.card,
    borderRadius: 12,
    padding: Spacing.lg,
    alignItems: 'center',
  },
  emptyText: {
    color: Colors.textSecondary,
    fontSize: FontSize.md,
  },
  emptySubtext: {
    color: Colors.textMuted,
    fontSize: FontSize.sm,
    marginTop: Spacing.xs,
  },

  // Position cards
  positionCard: {
    backgroundColor: Colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.cardBorder,
    padding: Spacing.md,
  },
  positionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  positionSymbol: {
    color: Colors.textPrimary,
    fontSize: FontSize.lg,
    fontWeight: '700',
  },
  positionPnl: {
    fontSize: FontSize.lg,
    fontWeight: '700',
  },
  positionDetails: {
    flexDirection: 'row',
    gap: Spacing.lg,
  },
  positionDetail: {},
  positionDetailLabel: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    marginBottom: 2,
  },
  positionDetailValue: {
    color: Colors.textSecondary,
    fontSize: FontSize.sm,
    fontWeight: '600',
  },
  guidanceRow: {
    marginTop: Spacing.sm,
    paddingTop: Spacing.sm,
    borderTopWidth: 1,
    borderTopColor: Colors.cardBorder,
  },
  guidanceText: {
    color: Colors.yellow,
    fontSize: FontSize.xs,
  },

  // Missed cards
  missedCard: {
    backgroundColor: Colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.cardBorder,
    padding: Spacing.md,
  },
  missedHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  missedSymbol: {
    color: Colors.textPrimary,
    fontSize: FontSize.lg,
    fontWeight: '700',
  },
  missedReturn: {
    fontSize: FontSize.lg,
    fontWeight: '700',
  },
  missedDetails: {
    flexDirection: 'row',
    gap: Spacing.lg,
    flexWrap: 'wrap',
  },
  missedDetail: {
    marginBottom: Spacing.xs,
  },
  missedDetailLabel: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    marginBottom: 2,
  },
  missedDetailValue: {
    color: Colors.textSecondary,
    fontSize: FontSize.sm,
    fontWeight: '600',
  },
  exitReasonRow: {
    marginTop: Spacing.sm,
    paddingTop: Spacing.sm,
    borderTopWidth: 1,
    borderTopColor: Colors.cardBorder,
  },
  exitReasonText: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
  },
});
