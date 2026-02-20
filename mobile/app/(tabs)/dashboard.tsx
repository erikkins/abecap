/**
 * Dashboard — main screen showing market regime, signal summary, and model portfolio.
 * Tabbed layout: Signals | Positions | History | Missed
 */

import React, { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useRouter } from 'expo-router';
import {
  useDashboard,
  useTradeHistory,
  Position,
  MissedOpportunity,
  Signal,
  Trade,
  trackSignal,
  sellPosition,
} from '@/hooks/useSignals';
import SignalCard from '@/components/SignalCard';
import RegimeBadge from '@/components/RegimeBadge';
import ConfirmModal from '@/components/ConfirmModal';
import { Colors, FontSize, Spacing } from '@/constants/theme';

type Tab = 'signals' | 'positions' | 'history' | 'missed';

export default function DashboardScreen() {
  const { data, isLoading, error, refresh } = useDashboard();
  const { trades, isLoading: tradesLoading, refresh: refreshTrades } = useTradeHistory();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<Tab>('signals');

  // Track modal state
  const [trackModal, setTrackModal] = useState<{ signal: Signal } | null>(null);
  const [trackLoading, setTrackLoading] = useState(false);

  // Sell modal state
  const [sellModal, setSellModal] = useState<{ position: Position } | null>(null);
  const [sellLoading, setSellLoading] = useState(false);

  const handleTrack = async () => {
    if (!trackModal) return;
    setTrackLoading(true);
    try {
      await trackSignal(trackModal.signal.symbol, trackModal.signal.price);
      setTrackModal(null);
      refresh();
    } catch (err: any) {
      Alert.alert('Error', err.response?.data?.detail || 'Failed to track signal');
    } finally {
      setTrackLoading(false);
    }
  };

  const handleSell = async () => {
    if (!sellModal) return;
    setSellLoading(true);
    try {
      await sellPosition(sellModal.position.id, sellModal.position.current_price);
      setSellModal(null);
      refresh();
      refreshTrades();
    } catch (err: any) {
      Alert.alert('Error', err.response?.data?.detail || 'Failed to sell position');
    } finally {
      setSellLoading(false);
    }
  };

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
    <>
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl
            refreshing={isLoading}
            onRefresh={() => { refresh(); if (activeTab === 'history') refreshTrades(); }}
            tintColor={Colors.gold}
          />
        }
      >
        {/* Market Regime */}
        {regime && (
          <RegimeBadge forecast={regime} marketStats={stats} />
        )}

        {/* Key Stats */}
        <View style={styles.statsRow}>
          <StatBox
            label="Portfolio"
            value={portfolio ? `$${portfolio.total_value?.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '—'}
          />
          <StatBox
            label="P&L"
            value={
              portfolio
                ? `${(portfolio.total_return_pct ?? 0) >= 0 ? '+' : ''}${portfolio.total_return_pct?.toFixed(1) ?? '0'}%`
                : '—'
            }
            change={portfolio?.total_return_pct}
          />
          <StatBox
            label="Signals"
            value={`${signals.length}`}
            sub={freshCount > 0 ? `${freshCount} fresh` : undefined}
          />
        </View>

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
          <TabPill label="History" tab="history" active={activeTab} onPress={setActiveTab} />
          <TabPill label="Missed" tab="missed" active={activeTab} onPress={setActiveTab} />
        </View>

        {/* Tab Content */}
        {activeTab === 'signals' && (
          <SignalsTab
            freshSignals={freshSignals}
            monitoringSignals={monitoringSignals}
            onSignalPress={(symbol) => router.push(`/signal/${symbol}`)}
            onTrack={(signal) => setTrackModal({ signal })}
          />
        )}
        {activeTab === 'positions' && (
          <PositionsTab
            positions={positions}
            onPositionPress={(symbol) => router.push(`/signal/${symbol}`)}
            onSell={(position) => setSellModal({ position })}
          />
        )}
        {activeTab === 'history' && (
          <HistoryTab trades={trades} isLoading={tradesLoading} />
        )}
        {activeTab === 'missed' && (
          <MissedTab
            missed={missed}
            onMissedPress={(symbol) => router.push(`/signal/${symbol}`)}
          />
        )}
      </ScrollView>

      {/* Track Confirmation Modal */}
      {trackModal && (
        <ConfirmModal
          visible
          title={`Track ${trackModal.signal.symbol}`}
          confirmLabel="Track"
          confirmColor={Colors.gold}
          onConfirm={handleTrack}
          onCancel={() => setTrackModal(null)}
          loading={trackLoading}
        >
          <View style={styles.modalRow}>
            <Text style={styles.modalLabel}>Price</Text>
            <Text style={styles.modalValue}>${trackModal.signal.price.toFixed(2)}</Text>
          </View>
          <View style={styles.modalRow}>
            <Text style={styles.modalLabel}>~Shares ($10k)</Text>
            <Text style={styles.modalValue}>
              {Math.floor(10000 / trackModal.signal.price)}
            </Text>
          </View>
          <View style={styles.modalRow}>
            <Text style={styles.modalLabel}>Ensemble Score</Text>
            <Text style={styles.modalValue}>{trackModal.signal.ensemble_score.toFixed(0)}</Text>
          </View>
        </ConfirmModal>
      )}

      {/* Sell Confirmation Modal */}
      {sellModal && (
        <ConfirmModal
          visible
          title={`Sell ${sellModal.position.symbol}`}
          confirmLabel="Sell"
          confirmColor={Colors.red}
          onConfirm={handleSell}
          onCancel={() => setSellModal(null)}
          loading={sellLoading}
        >
          <View style={styles.modalRow}>
            <Text style={styles.modalLabel}>Entry Price</Text>
            <Text style={styles.modalValue}>${sellModal.position.entry_price.toFixed(2)}</Text>
          </View>
          <View style={styles.modalRow}>
            <Text style={styles.modalLabel}>Current Price</Text>
            <Text style={styles.modalValue}>${sellModal.position.current_price.toFixed(2)}</Text>
          </View>
          <View style={styles.modalRow}>
            <Text style={styles.modalLabel}>P&L</Text>
            <Text
              style={[
                styles.modalValue,
                { color: sellModal.position.pnl_pct >= 0 ? Colors.green : Colors.red },
              ]}
            >
              {sellModal.position.pnl_pct >= 0 ? '+' : ''}{sellModal.position.pnl_pct.toFixed(1)}%
            </Text>
          </View>
          <View style={styles.modalRow}>
            <Text style={styles.modalLabel}>Shares</Text>
            <Text style={styles.modalValue}>{sellModal.position.shares}</Text>
          </View>
        </ConfirmModal>
      )}
    </>
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
  onTrack,
}: {
  freshSignals: Signal[];
  monitoringSignals: Signal[];
  onSignalPress: (symbol: string) => void;
  onTrack: (signal: Signal) => void;
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
          <View key={signal.symbol}>
            <SignalCard
              signal={signal}
              onPress={() => onSignalPress(signal.symbol)}
            />
            <Pressable
              style={styles.trackButton}
              onPress={() => onTrack(signal)}
            >
              <Text style={styles.trackButtonText}>Track</Text>
            </Pressable>
          </View>
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

function PositionsTab({
  positions,
  onPositionPress,
  onSell,
}: {
  positions: Position[];
  onPositionPress: (symbol: string) => void;
  onSell: (position: Position) => void;
}) {
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
        <Pressable
          key={pos.symbol}
          style={({ pressed }) => [styles.positionCard, pressed && styles.pressed]}
          onPress={() => onPositionPress(pos.symbol)}
        >
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
          <Pressable
            style={styles.sellButton}
            onPress={() => onSell(pos)}
          >
            <Text style={styles.sellButtonText}>Sell</Text>
          </Pressable>
        </Pressable>
      ))}
    </>
  );
}

/* ── History Tab ──────────────────────────────────────── */

function HistoryTab({
  trades,
  isLoading,
}: {
  trades: Trade[];
  isLoading: boolean;
}) {
  if (isLoading && trades.length === 0) {
    return (
      <View style={styles.emptyCard}>
        <ActivityIndicator color={Colors.gold} />
      </View>
    );
  }

  if (trades.length === 0) {
    return (
      <View style={styles.emptyCard}>
        <Text style={styles.emptyText}>No completed trades yet</Text>
      </View>
    );
  }

  const wins = trades.filter((t) => t.pnl_pct > 0).length;
  const winRate = ((wins / trades.length) * 100).toFixed(0);
  const avgReturn = (trades.reduce((sum, t) => sum + t.pnl_pct, 0) / trades.length).toFixed(1);

  return (
    <>
      {/* Summary Stats */}
      <View style={styles.historyStatsRow}>
        <View style={styles.historyStat}>
          <Text style={styles.historyStatValue}>{trades.length}</Text>
          <Text style={styles.historyStatLabel}>Trades</Text>
        </View>
        <View style={styles.historyStat}>
          <Text style={[styles.historyStatValue, { color: Colors.green }]}>{winRate}%</Text>
          <Text style={styles.historyStatLabel}>Win Rate</Text>
        </View>
        <View style={styles.historyStat}>
          <Text
            style={[
              styles.historyStatValue,
              { color: Number(avgReturn) >= 0 ? Colors.green : Colors.red },
            ]}
          >
            {Number(avgReturn) >= 0 ? '+' : ''}{avgReturn}%
          </Text>
          <Text style={styles.historyStatLabel}>Avg Return</Text>
        </View>
      </View>

      {/* Trade Cards */}
      {trades.map((trade) => (
        <View key={trade.id} style={styles.tradeCard}>
          <View style={styles.tradeHeader}>
            <Text style={styles.tradeSymbol}>{trade.symbol}</Text>
            <Text
              style={[
                styles.tradePnl,
                { color: trade.pnl_pct >= 0 ? Colors.green : Colors.red },
              ]}
            >
              {trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct.toFixed(1)}%
            </Text>
          </View>
          <View style={styles.tradeDetails}>
            <View style={styles.tradeDetail}>
              <Text style={styles.tradeDetailLabel}>Entry</Text>
              <Text style={styles.tradeDetailValue}>
                ${trade.entry_price.toFixed(2)} · {trade.entry_date}
              </Text>
            </View>
            <View style={styles.tradeDetail}>
              <Text style={styles.tradeDetailLabel}>Exit</Text>
              <Text style={styles.tradeDetailValue}>
                ${trade.exit_price.toFixed(2)} · {trade.exit_date}
              </Text>
            </View>
          </View>
          <View style={styles.tradeFooter}>
            <View style={styles.exitBadge}>
              <Text style={styles.exitBadgeText}>{formatExitReason(trade.exit_reason)}</Text>
            </View>
            {trade.pnl !== 0 && (
              <Text
                style={[
                  styles.tradePnlDollar,
                  { color: trade.pnl >= 0 ? Colors.green : Colors.red },
                ]}
              >
                {trade.pnl >= 0 ? '+' : ''}${Math.abs(trade.pnl).toFixed(0)}
              </Text>
            )}
          </View>
        </View>
      ))}
    </>
  );
}

/* ── Missed Tab ───────────────────────────────────────── */

function MissedTab({
  missed,
  onMissedPress,
}: {
  missed: MissedOpportunity[];
  onMissedPress: (symbol: string) => void;
}) {
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
        <Pressable
          key={`${m.symbol}-${m.entry_date}-${i}`}
          style={({ pressed }) => [styles.missedCard, pressed && styles.pressed]}
          onPress={() => onMissedPress(m.symbol)}
        >
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
        </Pressable>
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
  const valueColor =
    change != null
      ? change >= 0
        ? Colors.green
        : Colors.red
      : Colors.textPrimary;
  return (
    <View style={styles.statBox}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={[styles.statValue, { color: valueColor }]}>{value}</Text>
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
  pressed: {
    opacity: 0.8,
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

  // Track button
  trackButton: {
    backgroundColor: Colors.gold,
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: 'center',
    marginTop: -4,
    marginBottom: Spacing.sm,
  },
  trackButtonText: {
    color: Colors.navy,
    fontSize: FontSize.sm,
    fontWeight: '700',
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

  // Sell button
  sellButton: {
    backgroundColor: Colors.red + '22',
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: 'center',
    marginTop: Spacing.sm,
  },
  sellButtonText: {
    color: Colors.red,
    fontSize: FontSize.sm,
    fontWeight: '700',
  },

  // History tab
  historyStatsRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  historyStat: {
    flex: 1,
    backgroundColor: Colors.card,
    borderRadius: 8,
    padding: Spacing.md,
    alignItems: 'center',
  },
  historyStatValue: {
    color: Colors.textPrimary,
    fontSize: FontSize.lg,
    fontWeight: '700',
  },
  historyStatLabel: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    marginTop: 4,
  },

  // Trade cards
  tradeCard: {
    backgroundColor: Colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.cardBorder,
    padding: Spacing.md,
  },
  tradeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  tradeSymbol: {
    color: Colors.textPrimary,
    fontSize: FontSize.lg,
    fontWeight: '700',
  },
  tradePnl: {
    fontSize: FontSize.lg,
    fontWeight: '700',
  },
  tradeDetails: {
    gap: Spacing.xs,
  },
  tradeDetail: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  tradeDetailLabel: {
    color: Colors.textMuted,
    fontSize: FontSize.sm,
  },
  tradeDetailValue: {
    color: Colors.textSecondary,
    fontSize: FontSize.sm,
    fontWeight: '600',
  },
  tradeFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: Spacing.sm,
    paddingTop: Spacing.sm,
    borderTopWidth: 1,
    borderTopColor: Colors.cardBorder,
  },
  exitBadge: {
    backgroundColor: Colors.cardBorder,
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  exitBadgeText: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    fontWeight: '600',
  },
  tradePnlDollar: {
    fontSize: FontSize.sm,
    fontWeight: '600',
  },

  // Modal rows
  modalRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: Spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: Colors.cardBorder,
  },
  modalLabel: {
    color: Colors.textSecondary,
    fontSize: FontSize.md,
  },
  modalValue: {
    color: Colors.textPrimary,
    fontSize: FontSize.md,
    fontWeight: '600',
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
