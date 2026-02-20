/**
 * Individual signal detail screen.
 */

import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { Stack, useLocalSearchParams } from 'expo-router';
import api from '@/services/api';
import RegimeBadge from '@/components/RegimeBadge';
import { Colors, FontSize, Spacing } from '@/constants/theme';

interface SignalDetail {
  symbol: string;
  price: number;
  pct_above_dwap: number;
  is_strong: boolean;
  is_fresh: boolean;
  momentum_rank: number;
  ensemble_score: number;
  dwap_crossover_date: string | null;
  ensemble_entry_date: string | null;
  days_since_crossover: number | null;
  days_since_entry: number | null;
  sector?: string;
}

export default function SignalDetailScreen() {
  const { symbol } = useLocalSearchParams<{ symbol: string }>();
  const [signal, setSignal] = useState<SignalDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        // Fetch from dashboard data and find the matching signal
        const { data } = await api.get('/api/signals/dashboard');
        const found = data.buy_signals?.find(
          (s: SignalDetail) => s.symbol === symbol
        );
        setSignal(found || null);
      } catch {
        // Ignore — show empty state
      } finally {
        setLoading(false);
      }
    })();
  }, [symbol]);

  if (loading) {
    return (
      <View style={styles.center}>
        <Stack.Screen options={{ title: symbol || 'Signal' }} />
        <ActivityIndicator size="large" color={Colors.gold} />
      </View>
    );
  }

  if (!signal) {
    return (
      <View style={styles.center}>
        <Stack.Screen options={{ title: symbol || 'Signal' }} />
        <Text style={styles.emptyText}>Signal not found for {symbol}</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Stack.Screen
        options={{
          title: signal.symbol,
          headerStyle: { backgroundColor: Colors.navy },
          headerTintColor: Colors.textPrimary,
          headerShown: true,
        }}
      />

      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.symbol}>{signal.symbol}</Text>
          <View style={styles.badges}>
            {signal.is_fresh && (
              <View style={styles.freshBadge}>
                <Text style={styles.freshText}>FRESH</Text>
              </View>
            )}
            {signal.is_strong && (
              <View style={styles.strongBadge}>
                <Text style={styles.strongText}>STRONG</Text>
              </View>
            )}
          </View>
        </View>
        <Text style={styles.price}>${signal.price.toFixed(2)}</Text>
      </View>

      {/* Ensemble Score */}
      <View style={styles.scoreCard}>
        <Text style={styles.scoreLabel}>Ensemble Score</Text>
        <Text style={styles.scoreValue}>{signal.ensemble_score.toFixed(1)}</Text>
      </View>

      {/* Stats Grid */}
      <View style={styles.grid}>
        <DetailRow label="Breakout %" value={`+${signal.pct_above_dwap.toFixed(1)}%`} />
        <DetailRow label="Momentum Rank" value={`#${signal.momentum_rank}`} />
        <DetailRow
          label="Breakout Date"
          value={signal.dwap_crossover_date || '—'}
        />
        <DetailRow
          label="Days Since Breakout"
          value={signal.days_since_crossover != null ? `${signal.days_since_crossover}` : '—'}
        />
        <DetailRow
          label="Ensemble Entry"
          value={signal.ensemble_entry_date || '—'}
        />
        <DetailRow
          label="Days Since Entry"
          value={signal.days_since_entry != null ? `${signal.days_since_entry}` : '—'}
        />
        {signal.sector && (
          <DetailRow label="Sector" value={signal.sector} />
        )}
      </View>

      {/* Disclaimer */}
      <Text style={styles.disclaimer}>
        This is a buy signal, not a recommendation. Execute via your broker.
        Always do your own research.
      </Text>
    </ScrollView>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={styles.detailValue}>{value}</Text>
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
  emptyText: {
    color: Colors.textMuted,
    fontSize: FontSize.md,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  headerLeft: {
    gap: Spacing.sm,
  },
  symbol: {
    color: Colors.textPrimary,
    fontSize: FontSize.xxl,
    fontWeight: '800',
  },
  badges: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  freshBadge: {
    backgroundColor: Colors.green + '22',
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  freshText: {
    color: Colors.green,
    fontSize: FontSize.xs,
    fontWeight: '700',
  },
  strongBadge: {
    backgroundColor: Colors.gold + '22',
    borderRadius: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  strongText: {
    color: Colors.gold,
    fontSize: FontSize.xs,
    fontWeight: '700',
  },
  price: {
    color: Colors.textPrimary,
    fontSize: FontSize.xxl,
    fontWeight: '700',
  },
  scoreCard: {
    backgroundColor: Colors.gold + '15',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.gold + '33',
    padding: Spacing.lg,
    alignItems: 'center',
  },
  scoreLabel: {
    color: Colors.gold,
    fontSize: FontSize.sm,
    fontWeight: '600',
    marginBottom: Spacing.xs,
  },
  scoreValue: {
    color: Colors.gold,
    fontSize: 48,
    fontWeight: '800',
  },
  grid: {
    backgroundColor: Colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.cardBorder,
    overflow: 'hidden',
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.cardBorder,
  },
  detailLabel: {
    color: Colors.textSecondary,
    fontSize: FontSize.md,
  },
  detailValue: {
    color: Colors.textPrimary,
    fontSize: FontSize.md,
    fontWeight: '600',
  },
  disclaimer: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    textAlign: 'center',
    lineHeight: 18,
    marginTop: Spacing.md,
  },
});
