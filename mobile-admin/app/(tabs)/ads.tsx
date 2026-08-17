/**
 * Ads — Google Ads spend/clicks/conversions + first-party Traffic & the
 * /should-i-sell conversion funnel.
 *
 * Google data: a Google Ads Script POSTs campaign stats hourly to
 * /api/admin/ads/ingest; /ads/summary serves the latest snapshot (null on 404).
 * Traffic data: the cookieless first-party beacon → /api/admin/pageviews/summary
 * (works regardless of the Ads Script), including the /should-i-sell funnel.
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { AdsSummary, TrafficSummary, getAdsSummary, getTrafficSummary } from '@/services/admin';
import StatCard from '@/components/StatCard';
import Section from '@/components/Section';
import { Fonts, FontSize, Palette, Radii, Spacing } from '@/constants/theme';

const money = (n: any) => (typeof n === 'number' ? '$' + n.toFixed(2) : '—');
const int = (n: any) => (typeof n === 'number' ? n.toLocaleString('en-US') : '—');
const prettyRange = (s: any) => (s ? String(s).toLowerCase().replace(/_/g, ' ') : 'last 30 days');
const updatedAgo = (iso: any) => {
  if (!iso) return null;
  try {
    const m = Math.floor((Date.now() - new Date(iso).getTime()) / 60000);
    if (m < 1) return 'updated just now';
    if (m < 60) return `updated ${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `updated ${h}h ago`;
    return `updated ${Math.floor(h / 24)}d ago`;
  } catch {
    return null;
  }
};

const DAY_OPTS = [1, 7, 30];

export default function Ads() {
  const [ads, setAds] = useState<AdsSummary | null>(null);
  const [adsNotConfigured, setAdsNotConfigured] = useState(false);
  const [traffic, setTraffic] = useState<TrafficSummary | null>(null);
  const [days, setDays] = useState(7);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    const [a, t] = await Promise.allSettled([getAdsSummary(), getTrafficSummary(days)]);
    if (a.status === 'fulfilled') {
      if (a.value === null) setAdsNotConfigured(true);
      else {
        setAds(a.value);
        setAdsNotConfigured(false);
      }
    }
    if (t.status === 'fulfilled') setTraffic(t.value);
    if (a.status === 'rejected' && t.status === 'rejected') setError('Could not load ad or traffic stats.');
    setLoading(false);
  }, [days]);

  useEffect(() => {
    load();
  }, [load]);

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

  const pctOf = (n: any) =>
    traffic && traffic.total && typeof n === 'number' ? `${Math.round((n / traffic.total) * 100)}% of views` : undefined;

  // Both ad doors, same funnel steps → compare which is working.
  const FUNNEL_DOORS: Array<{ key: 'sis_funnel' | 'mom_funnel'; label: string; note: string }> = [
    { key: 'sis_funnel', label: '/should-i-sell', note: 'Preserver' },
    { key: 'mom_funnel', label: '/momentum', note: 'Maximizer' },
  ];

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={Palette.claret} />}
      >
        {/* Day-range toggle (drives Traffic) */}
        <View style={styles.toggleRow}>
          {DAY_OPTS.map((d) => (
            <TouchableOpacity
              key={d}
              onPress={() => setDays(d)}
              style={[styles.toggle, days === d && styles.toggleOn]}
              activeOpacity={0.7}
            >
              <Text style={[styles.toggleText, days === d && styles.toggleTextOn]}>{d === 1 ? '24h' : `${d}d`}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* ── Google Ads ─────────────────────────────────────────── */}
        {adsNotConfigured ? (
          <View style={styles.placeholder}>
            <Text style={styles.phTitle}>Google Ads not reporting yet</Text>
            <Text style={styles.phBody}>
              The hourly Google Ads Script hasn&rsquo;t posted a snapshot to{' '}
              <Text style={styles.mono}>/api/admin/ads/ingest</Text> yet. Traffic below is live regardless.
            </Text>
          </View>
        ) : (
          <>
            {updatedAgo(ads?.updated_at) ? <Text style={styles.freshness}>Google Ads · {updatedAgo(ads?.updated_at)}</Text> : null}
            <Section title="Google Ads" hint={prettyRange(ads?.date_range)}>
              <View style={styles.grid}>
                <StatCard label="Spend" value={money(ads?.spend)} />
                <StatCard label="Clicks" value={int(ads?.clicks)} />
                <StatCard label="Impressions" value={int(ads?.impressions)} />
                <StatCard label="Avg CPC" value={money(ads?.cpc)} />
                <StatCard
                  label="Conversions"
                  value={int(ads?.conversions)}
                  tone={typeof ads?.conversions === 'number' && ads.conversions > 0 ? 'positive' : 'default'}
                />
              </View>
            </Section>

            {ads?.campaigns?.length ? (
              <Section title="Campaigns">
                {ads.campaigns.map((c, i) => (
                  <View key={i} style={styles.row}>
                    <Text style={styles.rowName} numberOfLines={1}>
                      {c.name || c.campaign || `Campaign ${i + 1}`}
                    </Text>
                    <Text style={styles.rowVal}>{money(c.spend)}</Text>
                  </View>
                ))}
              </Section>
            ) : null}
          </>
        )}

        {/* ── Traffic (first-party, cookieless) ──────────────────── */}
        {traffic ? (
          <>
            <Section title="Traffic" hint={`last ${days === 1 ? '24h' : `${days}d`}`}>
              <View style={styles.grid}>
                <StatCard label="Pageviews" value={int(traffic.total)} />
                <StatCard label="Paid clicks" value={int(traffic.paid_clicks)} sub={pctOf(traffic.paid_clicks)} />
                <StatCard label="Mobile" value={int(traffic.mobile)} sub={pctOf(traffic.mobile)} />
              </View>
            </Section>

            {/* Ad-door conversion funnels — /should-i-sell (Preserver) vs /momentum (Maximizer) */}
            {FUNNEL_DOORS.map(({ key, label, note }) => {
              const f = traffic[key] || [];
              const landed = f.find((s) => s.step === 'pageview')?.count ?? f[0]?.count ?? 0;
              if (!f.length) return null;
              const fpct = (n: number) => (landed ? `${Math.round((n / landed) * 100)}%` : '—');
              return (
                <Section key={key} title={`${label} funnel`} hint={landed ? `${note} · ${landed} landed` : `${note} · no traffic yet`}>
                  {f.map((s) => {
                    const w = landed && s.count > 0 ? Math.max(2, Math.round((s.count / landed) * 100)) : 0;
                    return (
                      <View key={s.step} style={styles.funnelRow}>
                        <Text style={styles.funnelLabel} numberOfLines={1}>
                          {s.label}
                        </Text>
                        <View style={styles.funnelBarTrack}>
                          {w > 0 ? <View style={[styles.funnelBarFill, { width: `${w}%` }]} /> : null}
                        </View>
                        <Text style={styles.funnelCount}>
                          {s.count} <Text style={styles.funnelPct}>{fpct(s.count)}</Text>
                        </Text>
                      </View>
                    );
                  })}
                </Section>
              );
            })}

            {traffic.by_path?.length ? (
              <Section title="Top pages">
                {traffic.by_path.slice(0, 10).map((p) => (
                  <View key={p.path} style={styles.row}>
                    <Text style={styles.rowName} numberOfLines={1}>
                      {p.path}
                    </Text>
                    <Text style={styles.rowVal}>{p.count}</Text>
                  </View>
                ))}
              </Section>
            ) : null}

            {traffic.by_source?.length ? (
              <Section title="By source">
                {traffic.by_source.slice(0, 8).map((s) => (
                  <View key={s.source} style={styles.row}>
                    <Text style={styles.rowName} numberOfLines={1}>
                      {s.source}
                    </Text>
                    <Text style={styles.rowVal}>{s.count}</Text>
                  </View>
                ))}
              </Section>
            ) : null}
          </>
        ) : null}

        {error ? <Text style={styles.error}>{error}</Text> : null}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: Palette.paper },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: Palette.paper },
  content: { padding: Spacing.lg, paddingBottom: Spacing.xxl },
  toggleRow: { flexDirection: 'row', gap: Spacing.sm, marginBottom: Spacing.lg },
  toggle: {
    borderWidth: 1,
    borderColor: Palette.rule,
    borderRadius: Radii.pill,
    paddingHorizontal: Spacing.md,
    paddingVertical: 5,
  },
  toggleOn: { backgroundColor: Palette.ink, borderColor: Palette.ink },
  toggleText: { fontFamily: Fonts.body.medium, fontSize: FontSize.xs, color: Palette.inkMute },
  toggleTextOn: { color: Palette.paper },
  freshness: { fontFamily: Fonts.mono.regular, fontSize: FontSize.xs, color: Palette.inkLight, marginBottom: Spacing.md },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.md },
  placeholder: {
    backgroundColor: Palette.paperCard,
    borderWidth: 1,
    borderColor: Palette.rule,
    borderRadius: Radii.lg,
    padding: Spacing.lg,
    marginBottom: Spacing.xl,
  },
  phTitle: { fontFamily: Fonts.display.semibold, fontSize: FontSize.md, color: Palette.ink, marginBottom: Spacing.sm },
  phBody: { fontFamily: Fonts.body.regular, fontSize: FontSize.sm, color: Palette.inkMute, lineHeight: 20 },
  mono: { fontFamily: Fonts.mono.regular, fontSize: FontSize.xs, color: Palette.claret },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: Spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: Palette.rule,
  },
  rowName: { fontFamily: Fonts.body.medium, fontSize: FontSize.sm, color: Palette.ink, flex: 1, marginRight: Spacing.md },
  rowVal: { fontFamily: Fonts.mono.medium, fontSize: FontSize.sm, color: Palette.ink },
  funnelRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 5 },
  funnelLabel: { fontFamily: Fonts.body.regular, fontSize: FontSize.xs, color: Palette.inkMute, width: 108 },
  funnelBarTrack: {
    flex: 1,
    height: 14,
    backgroundColor: Palette.paperDeep,
    borderRadius: Radii.sm,
    marginHorizontal: Spacing.sm,
    overflow: 'hidden',
  },
  funnelBarFill: { height: '100%', backgroundColor: Palette.claret, borderRadius: Radii.sm },
  funnelCount: { fontFamily: Fonts.mono.medium, fontSize: FontSize.xs, color: Palette.ink, width: 74, textAlign: 'right' },
  funnelPct: { fontFamily: Fonts.mono.regular, color: Palette.inkLight },
  error: { fontFamily: Fonts.body.medium, fontSize: FontSize.sm, color: Palette.negative, marginTop: Spacing.md },
});
