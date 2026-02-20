/**
 * Dashboard — main screen showing market regime, signal summary, and model portfolio.
 */

import React from 'react';
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useDashboard } from '@/hooks/useSignals';
import SignalCard from '@/components/SignalCard';
import RegimeBadge from '@/components/RegimeBadge';
import { Colors, FontSize, Spacing } from '@/constants/theme';

export default function DashboardScreen() {
  const { data, isLoading, error, refresh } = useDashboard();
  const router = useRouter();

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
  const freshCount = signals.filter((s) => s.is_fresh).length;
  const regime = data?.regime_forecast;
  const stats = data?.market_stats;
  const portfolio = data?.model_portfolio;

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

      {/* Today's Signals */}
      <Text style={styles.sectionTitle}>
        Today's Buy Signals ({signals.length})
      </Text>

      {signals.length === 0 ? (
        <View style={styles.emptyCard}>
          <Text style={styles.emptyText}>
            No ensemble buy signals today.
          </Text>
          <Text style={styles.emptySubtext}>
            Check back after the 4 PM ET scan.
          </Text>
        </View>
      ) : (
        signals.map((signal) => (
          <SignalCard
            key={signal.symbol}
            signal={signal}
            onPress={() => router.push(`/signal/${signal.symbol}`)}
          />
        ))
      )}
    </ScrollView>
  );
}

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
  sectionTitle: {
    color: Colors.textPrimary,
    fontSize: FontSize.lg,
    fontWeight: '700',
  },
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
});
