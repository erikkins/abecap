/**
 * Social — read-only follow-along queue. "Pending approval" (draft posts, esp.
 * contextual replies) and "Scheduled" (upcoming autoposts, soonest first).
 * Approve/kill still happen via the one-click email links; this is just the
 * at-a-glance view of what's in the pipe.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, RefreshControl, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { SocialPost, getSocialPosts } from '@/services/admin';
import Section from '@/components/Section';
import { Fonts, FontSize, Palette, Radii, Spacing } from '@/constants/theme';

const PLATFORM: Record<string, string> = {
  twitter: 'Twitter/X',
  instagram: 'Instagram',
  threads: 'Threads',
  tiktok: 'TikTok',
};
const TYPE: Record<string, string> = {
  contextual_reply: 'Reply',
  research_insight: 'Insight',
  trade_result: 'Trade result',
  missed_opportunity: 'Missed opp',
  we_called_it: 'We called it',
};

const whenLabel = (iso: any, prefix: string) => {
  if (!iso) return null;
  try {
    const d = new Date(iso);
    const now = Date.now();
    const diffM = Math.round((d.getTime() - now) / 60000);
    const abs = Math.abs(diffM);
    const rel =
      abs < 60 ? `${abs}m` : abs < 1440 ? `${Math.round(abs / 60)}h` : `${Math.round(abs / 1440)}d`;
    const dateStr = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    const timeStr = d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    return diffM >= 0 ? `${prefix} ${dateStr} ${timeStr} · in ${rel}` : `${prefix} ${dateStr} ${timeStr} · ${rel} ago`;
  } catch {
    return null;
  }
};

function PostCard({ post, mode }: { post: SocialPost; mode: 'draft' | 'scheduled' }) {
  const platform = PLATFORM[post.platform] || post.platform || '—';
  const type = TYPE[post.post_type] || post.post_type || 'Post';
  const tier = post.tier || (post as any)?.tier_hint;
  const when =
    mode === 'scheduled'
      ? whenLabel(post.scheduled_for, 'Posts')
      : whenLabel(post.created_at, 'Drafted');
  const reply = post.reply_to_username ? `↳ @${post.reply_to_username}` : null;
  return (
    <View style={styles.card}>
      <View style={styles.cardHead}>
        <Text style={styles.type}>{type}</Text>
        <View style={styles.chips}>
          {tier ? (
            <View style={[styles.chip, { borderColor: Palette.claret }]}>
              <Text style={[styles.chipText, { color: Palette.claret }]}>{String(tier).toUpperCase()}</Text>
            </View>
          ) : null}
          {post.ai_generated ? (
            <View style={styles.chip}>
              <Text style={styles.chipText}>AI</Text>
            </View>
          ) : null}
          <View style={styles.chip}>
            <Text style={styles.chipText}>{platform}</Text>
          </View>
        </View>
      </View>
      {reply ? (
        <Text style={styles.reply} numberOfLines={1}>
          {reply}
        </Text>
      ) : null}
      <Text style={styles.body} numberOfLines={4}>
        {post.text_content || '(no text)'}
      </Text>
      {when ? <Text style={styles.when}>{when}</Text> : null}
    </View>
  );
}

export default function Social() {
  const [drafts, setDrafts] = useState<SocialPost[]>([]);
  const [scheduled, setScheduled] = useState<SocialPost[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    const [d, s] = await Promise.allSettled([getSocialPosts('draft'), getSocialPosts('scheduled')]);
    if (d.status === 'fulfilled') setDrafts(d.value);
    if (s.status === 'fulfilled') setScheduled(s.value);
    if (d.status === 'rejected' && s.status === 'rejected') setError('Could not load the social queue.');
    setLoading(false);
  }, []);

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

  return (
    <SafeAreaView style={styles.safe} edges={['bottom']}>
      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={Palette.claret} />}
      >
        <Text style={styles.note}>Read-only · approve or kill from the email links.</Text>

        <Section title="Pending approval" hint={drafts.length ? `${drafts.length}` : ''}>
          {drafts.length === 0 ? (
            <Text style={styles.empty}>Nothing waiting on you.</Text>
          ) : (
            drafts.map((p) => <PostCard key={p.id} post={p} mode="draft" />)
          )}
        </Section>

        <Section title="Scheduled" hint={scheduled.length ? `${scheduled.length}` : ''}>
          {scheduled.length === 0 ? (
            <Text style={styles.empty}>Nothing scheduled.</Text>
          ) : (
            scheduled.map((p) => <PostCard key={p.id} post={p} mode="scheduled" />)
          )}
        </Section>

        {error ? <Text style={styles.error}>{error}</Text> : null}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: Palette.paper },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: Palette.paper },
  content: { padding: Spacing.lg, paddingBottom: Spacing.xxl },
  note: { fontFamily: Fonts.body.italic, fontSize: FontSize.xs, color: Palette.inkLight, marginBottom: Spacing.lg },
  card: {
    backgroundColor: Palette.paperCard,
    borderWidth: 1,
    borderColor: Palette.rule,
    borderRadius: Radii.lg,
    padding: Spacing.md,
    marginBottom: Spacing.md,
  },
  cardHead: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: Spacing.xs },
  type: { fontFamily: Fonts.display.medium, fontSize: FontSize.md, color: Palette.ink },
  chips: { flexDirection: 'row', gap: Spacing.xs },
  chip: { borderWidth: 1, borderColor: Palette.rule, borderRadius: Radii.sm, paddingHorizontal: 6, paddingVertical: 2 },
  chipText: { fontFamily: Fonts.mono.regular, fontSize: 9, letterSpacing: 0.4, color: Palette.inkMute },
  reply: { fontFamily: Fonts.mono.regular, fontSize: FontSize.xs, color: Palette.inkLight, marginBottom: 4 },
  body: { fontFamily: Fonts.body.regular, fontSize: FontSize.sm, color: Palette.ink, lineHeight: 20 },
  when: { fontFamily: Fonts.mono.regular, fontSize: FontSize.xs, color: Palette.inkLight, marginTop: Spacing.sm },
  empty: { fontFamily: Fonts.body.regular, fontSize: FontSize.sm, color: Palette.inkMute },
  error: { fontFamily: Fonts.body.medium, fontSize: FontSize.sm, color: Palette.negative, marginTop: Spacing.md },
});
