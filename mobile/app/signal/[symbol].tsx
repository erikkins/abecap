/**
 * Individual signal detail screen.
 *
 * Portrait: data card + compact chart.
 * Landscape: full-screen interactive chart with pinch-to-zoom.
 */

import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Dimensions,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { Stack, useLocalSearchParams } from 'expo-router';
import * as ScreenOrientation from 'expo-screen-orientation';
import api from '@/services/api';
import PriceChart from '@/components/PriceChart';
import { useChartData } from '@/hooks/useChartData';
import { useStockInfo } from '@/hooks/useStockInfo';
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

const PERIOD_OPTIONS = [
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '6M', days: 180 },
  { label: '1Y', days: 252 },
];

export default function SignalDetailScreen() {
  const { symbol } = useLocalSearchParams<{ symbol: string }>();
  const [signal, setSignal] = useState<SignalDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [isLandscape, setIsLandscape] = useState(false);
  const [chartDays, setChartDays] = useState(180);

  const { data: chartData, isLoading: chartLoading } = useChartData(
    symbol || '',
    chartDays
  );
  const { info: stockInfo } = useStockInfo(symbol || '');

  // Unlock landscape for this screen, lock back on unmount
  useEffect(() => {
    ScreenOrientation.unlockAsync();
    const sub = ScreenOrientation.addOrientationChangeListener((event) => {
      const o = event.orientationInfo.orientation;
      setIsLandscape(
        o === ScreenOrientation.Orientation.LANDSCAPE_LEFT ||
        o === ScreenOrientation.Orientation.LANDSCAPE_RIGHT
      );
    });

    // Check initial orientation
    ScreenOrientation.getOrientationAsync().then((o) => {
      setIsLandscape(
        o === ScreenOrientation.Orientation.LANDSCAPE_LEFT ||
        o === ScreenOrientation.Orientation.LANDSCAPE_RIGHT
      );
    });

    return () => {
      sub.remove();
      ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.PORTRAIT_UP);
    };
  }, []);

  // Fetch signal data
  useEffect(() => {
    (async () => {
      try {
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
        <Stack.Screen options={{ title: symbol || 'Signal', headerShown: !isLandscape, headerBackTitle: 'Back' }} />
        <ActivityIndicator size="large" color={Colors.gold} />
      </View>
    );
  }

  if (!signal) {
    return (
      <View style={styles.center}>
        <Stack.Screen options={{ title: symbol || 'Signal', headerShown: true, headerBackTitle: 'Back' }} />
        <Text style={styles.emptyText}>Signal not found for {symbol}</Text>
      </View>
    );
  }

  // ── Landscape: full-screen chart ──
  if (isLandscape) {
    return (
      <View style={styles.landscapeWrap}>
        <Stack.Screen options={{ headerShown: false }} />
        {/* Symbol + price overlay */}
        <View style={styles.landscapeHeader}>
          <Text style={styles.landscapeSymbol}>{signal.symbol}</Text>
          <Text style={styles.landscapePrice}>${signal.price.toFixed(2)}</Text>
        </View>
        {/* Period selector */}
        <View style={styles.landscapePeriods}>
          {PERIOD_OPTIONS.map((p) => (
            <Pressable
              key={p.label}
              onPress={() => setChartDays(p.days)}
              style={[
                styles.periodButton,
                chartDays === p.days && styles.periodActive,
              ]}
            >
              <Text
                style={[
                  styles.periodText,
                  chartDays === p.days && styles.periodTextActive,
                ]}
              >
                {p.label}
              </Text>
            </Pressable>
          ))}
        </View>
        {chartLoading ? (
          <View style={styles.center}>
            <ActivityIndicator color={Colors.gold} />
          </View>
        ) : (
          <PriceChart
            data={chartData}
            entryDate={signal.ensemble_entry_date}
            breakoutDate={signal.dwap_crossover_date}
            isLandscape
          />
        )}
        <Text style={styles.landscapeHint}>Pinch to zoom  |  Drag for crosshair</Text>
      </View>
    );
  }

  // ── Portrait: data card + compact chart ──
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Stack.Screen
        options={{
          title: signal.symbol,
          headerStyle: { backgroundColor: Colors.navy },
          headerTintColor: Colors.textPrimary,
          headerBackTitle: 'Back',
          headerShown: true,
        }}
      />

      {/* Chart */}
      <View>
        <View style={styles.periodRow}>
          {PERIOD_OPTIONS.map((p) => (
            <Pressable
              key={p.label}
              onPress={() => setChartDays(p.days)}
              style={[
                styles.periodButton,
                chartDays === p.days && styles.periodActive,
              ]}
            >
              <Text
                style={[
                  styles.periodText,
                  chartDays === p.days && styles.periodTextActive,
                ]}
              >
                {p.label}
              </Text>
            </Pressable>
          ))}
        </View>
        {chartLoading ? (
          <View style={[styles.chartPlaceholder]}>
            <ActivityIndicator color={Colors.gold} />
          </View>
        ) : (
          <PriceChart
            data={chartData}
            entryDate={signal.ensemble_entry_date}
            breakoutDate={signal.dwap_crossover_date}
          />
        )}
        <Text style={styles.rotateHint}>Rotate for full-screen chart</Text>
      </View>

      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.symbol}>{signal.symbol}</Text>
          {stockInfo?.name && (
            <Text style={styles.companyName}>{stockInfo.name}</Text>
          )}
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

      {/* Company Info */}
      {stockInfo && (
        <View style={styles.companySection}>
          {stockInfo.sector && stockInfo.industry && (
            <View style={styles.industryBadge}>
              <Text style={styles.industryText}>
                {stockInfo.sector} — {stockInfo.industry}
              </Text>
            </View>
          )}
          {stockInfo.description && (
            <Text style={styles.companyDesc} numberOfLines={4}>
              {stockInfo.description}
            </Text>
          )}
        </View>
      )}

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
  // Portrait chart
  periodRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: Spacing.sm,
    marginBottom: Spacing.sm,
  },
  periodButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    backgroundColor: Colors.card,
  },
  periodActive: {
    backgroundColor: Colors.gold + '33',
  },
  periodText: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    fontWeight: '600',
  },
  periodTextActive: {
    color: Colors.gold,
  },
  chartPlaceholder: {
    height: 220,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: Colors.card,
    borderRadius: 12,
  },
  rotateHint: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    textAlign: 'center',
    marginTop: Spacing.xs,
  },
  // Landscape
  landscapeWrap: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  landscapeHeader: {
    position: 'absolute',
    top: Spacing.sm,
    left: Spacing.lg,
    zIndex: 10,
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: Spacing.sm,
  },
  landscapeSymbol: {
    color: Colors.textPrimary,
    fontSize: FontSize.lg,
    fontWeight: '800',
  },
  landscapePrice: {
    color: Colors.gold,
    fontSize: FontSize.md,
    fontWeight: '600',
  },
  landscapePeriods: {
    position: 'absolute',
    top: Spacing.sm,
    right: Spacing.lg,
    zIndex: 10,
    flexDirection: 'row',
    gap: Spacing.xs,
  },
  landscapeHint: {
    position: 'absolute',
    bottom: 4,
    left: 0,
    right: 0,
    textAlign: 'center',
    color: Colors.textMuted,
    fontSize: 9,
    opacity: 0.6,
  },
  // Data card
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
  companyName: {
    color: Colors.textSecondary,
    fontSize: FontSize.sm,
  },
  companySection: {
    gap: Spacing.sm,
  },
  industryBadge: {
    alignSelf: 'flex-start',
    backgroundColor: '#4338CA22',
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  industryText: {
    color: '#818CF8',
    fontSize: FontSize.xs,
    fontWeight: '600',
  },
  companyDesc: {
    color: Colors.textMuted,
    fontSize: FontSize.sm,
    lineHeight: 18,
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
