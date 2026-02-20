/**
 * Settings screen — account info, subscription, notification prefs, logout.
 */

import React from 'react';
import {
  Alert,
  Linking,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useAuth } from '@/hooks/useAuth';
import { Colors, FontSize, Spacing } from '@/constants/theme';

export default function SettingsScreen() {
  const { user, logout } = useAuth();

  const handleLogout = () => {
    Alert.alert('Sign Out', 'Are you sure you want to sign out?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Sign Out', style: 'destructive', onPress: logout },
    ]);
  };

  const sub = user?.subscription;

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
    >
      {/* Account */}
      <Text style={styles.sectionTitle}>Account</Text>
      <View style={styles.card}>
        <Row label="Email" value={user?.email || '—'} />
        <Row label="Name" value={user?.name || '—'} />
      </View>

      {/* Subscription */}
      <Text style={styles.sectionTitle}>Subscription</Text>
      <View style={styles.card}>
        <Row
          label="Status"
          value={sub?.status?.toUpperCase() || 'NONE'}
          valueColor={sub?.is_valid ? Colors.green : Colors.red}
        />
        {sub?.days_remaining != null && (
          <Row label="Days Remaining" value={`${sub.days_remaining}`} />
        )}
        <Pressable
          style={styles.linkRow}
          onPress={() => Linking.openURL('https://rigacap.com/#pricing')}
        >
          <Text style={styles.linkText}>Manage Subscription</Text>
          <Text style={styles.arrow}>→</Text>
        </Pressable>
      </View>

      {/* Links */}
      <Text style={styles.sectionTitle}>About</Text>
      <View style={styles.card}>
        <Pressable
          style={styles.linkRow}
          onPress={() => Linking.openURL('https://rigacap.com/privacy')}
        >
          <Text style={styles.linkText}>Privacy Policy</Text>
          <Text style={styles.arrow}>→</Text>
        </Pressable>
        <Pressable
          style={styles.linkRow}
          onPress={() => Linking.openURL('https://rigacap.com/terms')}
        >
          <Text style={styles.linkText}>Terms of Service</Text>
          <Text style={styles.arrow}>→</Text>
        </Pressable>
        <Pressable
          style={styles.linkRow}
          onPress={() => Linking.openURL('https://rigacap.com/contact')}
        >
          <Text style={styles.linkText}>Contact Us</Text>
          <Text style={styles.arrow}>→</Text>
        </Pressable>
      </View>

      {/* Disclaimer */}
      <View style={styles.disclaimerCard}>
        <Text style={styles.disclaimerTitle}>Disclaimer</Text>
        <Text style={styles.disclaimerText}>
          RigaCap provides trading signals only. We are not a broker and do not
          execute trades on your behalf. All signals should be considered
          informational — not financial advice. Always do your own research
          before making investment decisions.
        </Text>
      </View>

      {/* Logout */}
      <Pressable style={styles.logoutButton} onPress={handleLogout}>
        <Text style={styles.logoutText}>Sign Out</Text>
      </Pressable>

      <Text style={styles.version}>RigaCap v1.0.0</Text>
    </ScrollView>
  );
}

function Row({
  label,
  value,
  valueColor,
}: {
  label: string;
  value: string;
  valueColor?: string;
}) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={[styles.rowValue, valueColor ? { color: valueColor } : {}]}>
        {value}
      </Text>
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
    paddingBottom: Spacing.xl * 2,
    gap: Spacing.sm,
  },
  sectionTitle: {
    color: Colors.textSecondary,
    fontSize: FontSize.sm,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginTop: Spacing.md,
    marginBottom: Spacing.xs,
  },
  card: {
    backgroundColor: Colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.cardBorder,
    overflow: 'hidden',
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.cardBorder,
  },
  rowLabel: {
    color: Colors.textSecondary,
    fontSize: FontSize.md,
  },
  rowValue: {
    color: Colors.textPrimary,
    fontSize: FontSize.md,
    fontWeight: '500',
  },
  linkRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.cardBorder,
  },
  linkText: {
    color: Colors.gold,
    fontSize: FontSize.md,
  },
  arrow: {
    color: Colors.textMuted,
    fontSize: FontSize.md,
  },
  disclaimerCard: {
    backgroundColor: Colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.cardBorder,
    padding: Spacing.md,
    marginTop: Spacing.md,
  },
  disclaimerTitle: {
    color: Colors.textSecondary,
    fontSize: FontSize.sm,
    fontWeight: '600',
    marginBottom: Spacing.xs,
  },
  disclaimerText: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    lineHeight: 18,
  },
  logoutButton: {
    backgroundColor: Colors.red + '22',
    borderRadius: 8,
    padding: Spacing.md,
    alignItems: 'center',
    marginTop: Spacing.lg,
  },
  logoutText: {
    color: Colors.red,
    fontSize: FontSize.md,
    fontWeight: '600',
  },
  version: {
    color: Colors.textMuted,
    fontSize: FontSize.xs,
    textAlign: 'center',
    marginTop: Spacing.md,
  },
});
