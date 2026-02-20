/**
 * Market regime indicator badge.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Colors, FontSize, Spacing } from '@/constants/theme';

const REGIME_LABELS: Record<string, string> = {
  strong_bull: 'Strong Bull',
  weak_bull: 'Weak Bull',
  rotating_bull: 'Rotating Bull',
  range_bound: 'Range Bound',
  weak_bear: 'Weak Bear',
  panic_crash: 'Panic / Crash',
  recovery: 'Recovery',
};

interface RegimeBadgeProps {
  regime: string;
  compact?: boolean;
}

export default function RegimeBadge({ regime, compact }: RegimeBadgeProps) {
  const color = Colors.regime[regime] || Colors.textMuted;
  const label = REGIME_LABELS[regime] || regime;

  if (compact) {
    return (
      <View style={[styles.compactBadge, { backgroundColor: color + '22' }]}>
        <View style={[styles.dot, { backgroundColor: color }]} />
        <Text style={[styles.compactText, { color }]}>{label}</Text>
      </View>
    );
  }

  return (
    <View style={[styles.banner, { borderLeftColor: color }]}>
      <View style={[styles.dot, { backgroundColor: color }]} />
      <Text style={styles.bannerLabel}>Market Regime</Text>
      <Text style={[styles.bannerValue, { color }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.card,
    borderRadius: 8,
    borderLeftWidth: 3,
    padding: Spacing.md,
    gap: Spacing.sm,
  },
  bannerLabel: {
    color: Colors.textSecondary,
    fontSize: FontSize.sm,
  },
  bannerValue: {
    fontSize: FontSize.md,
    fontWeight: '700',
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  compactBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 4,
    gap: 6,
  },
  compactText: {
    fontSize: FontSize.xs,
    fontWeight: '600',
  },
});
