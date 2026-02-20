/**
 * Buy signal card — used in dashboard and signal list.
 */

import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Colors, FontSize, Spacing } from '@/constants/theme';
import { Signal } from '@/hooks/useSignals';

interface SignalCardProps {
  signal: Signal;
  onPress?: () => void;
}

export default function SignalCard({ signal, onPress }: SignalCardProps) {
  return (
    <Pressable
      style={({ pressed }) => [styles.card, pressed && styles.pressed]}
      onPress={onPress}
    >
      <View style={styles.header}>
        <View style={styles.symbolRow}>
          <Text style={styles.symbol}>{signal.symbol}</Text>
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
        <Text style={styles.price}>${signal.price.toFixed(2)}</Text>
      </View>

      <View style={styles.stats}>
        <StatItem label="Breakout" value={`+${signal.pct_above_dwap.toFixed(1)}%`} />
        <StatItem label="Rank" value={`#${signal.momentum_rank}`} />
        <StatItem label="Score" value={signal.ensemble_score.toFixed(0)} />
        {signal.days_since_crossover != null && (
          <StatItem label="Age" value={`${signal.days_since_crossover}d`} />
        )}
      </View>
    </Pressable>
  );
}

function StatItem({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.statItem}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: Colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.cardBorder,
    padding: Spacing.md,
    marginBottom: Spacing.sm,
  },
  pressed: {
    opacity: 0.8,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  symbolRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  symbol: {
    color: Colors.textPrimary,
    fontSize: FontSize.lg,
    fontWeight: '700',
  },
  price: {
    color: Colors.textPrimary,
    fontSize: FontSize.lg,
    fontWeight: '600',
  },
  freshBadge: {
    backgroundColor: Colors.green + '22',
    borderRadius: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  freshText: {
    color: Colors.green,
    fontSize: FontSize.xs,
    fontWeight: '700',
  },
  strongBadge: {
    backgroundColor: Colors.gold + '22',
    borderRadius: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  strongText: {
    color: Colors.gold,
    fontSize: FontSize.xs,
    fontWeight: '700',
  },
  stats: {
    flexDirection: 'row',
    gap: Spacing.lg,
  },
  statItem: {},
  statLabel: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    marginBottom: 2,
  },
  statValue: {
    color: Colors.textSecondary,
    fontSize: FontSize.sm,
    fontWeight: '600',
  },
});
