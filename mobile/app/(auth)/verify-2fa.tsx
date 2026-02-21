/**
 * 2FA verification screen — enter TOTP code or backup code after login.
 */

import React, { useState, useRef } from 'react';
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '@/hooks/useAuth';
import { Colors, FontSize, Spacing } from '@/constants/theme';

export default function Verify2FAScreen() {
  const { verify2FA, cancel2FA } = useAuth();
  const [code, setCode] = useState('');
  const [trustDevice, setTrustDevice] = useState(false);
  const [useBackupCode, setUseBackupCode] = useState(false);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<TextInput>(null);

  const handleVerify = async () => {
    if (!code.trim()) return;
    setLoading(true);
    try {
      await verify2FA(code.trim(), trustDevice, useBackupCode);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Verification failed';
      Alert.alert('Invalid Code', msg);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    cancel2FA();
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        style={styles.inner}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <View style={styles.header}>
          <Text style={styles.title}>Two-Factor{'\n'}Authentication</Text>
          <Text style={styles.subtitle}>
            {useBackupCode
              ? 'Enter one of your 8-character backup codes.'
              : 'Enter the 6-digit code from your authenticator app.'}
          </Text>
        </View>

        <View style={styles.form}>
          <TextInput
            ref={inputRef}
            style={styles.codeInput}
            value={code}
            onChangeText={setCode}
            placeholder={useBackupCode ? 'ABCD1234' : '000000'}
            placeholderTextColor={Colors.textMuted}
            maxLength={useBackupCode ? 8 : 6}
            keyboardType={useBackupCode ? 'default' : 'number-pad'}
            autoFocus
            textAlign="center"
            autoCapitalize={useBackupCode ? 'characters' : 'none'}
            autoComplete="one-time-code"
            textContentType="oneTimeCode"
          />

          <View style={styles.trustRow}>
            <Text style={styles.trustLabel}>Trust this device for 30 days</Text>
            <Switch
              value={trustDevice}
              onValueChange={setTrustDevice}
              trackColor={{ false: Colors.navyLight, true: Colors.gold }}
              thumbColor="#fff"
            />
          </View>

          <Pressable
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={handleVerify}
            disabled={loading || (!useBackupCode && code.length !== 6) || (useBackupCode && code.length < 8)}
          >
            <Text style={styles.buttonText}>
              {loading ? 'Verifying...' : 'Verify'}
            </Text>
          </Pressable>

          <View style={styles.links}>
            <Pressable
              onPress={() => {
                setUseBackupCode(!useBackupCode);
                setCode('');
              }}
            >
              <Text style={styles.link}>
                {useBackupCode ? 'Use authenticator app' : 'Use a backup code'}
              </Text>
            </Pressable>

            <Pressable onPress={handleCancel}>
              <Text style={styles.cancelLink}>Back to login</Text>
            </Pressable>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  inner: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: Spacing.lg,
  },
  header: {
    marginBottom: Spacing.xl,
  },
  title: {
    fontSize: FontSize.xxl,
    fontWeight: '700',
    color: Colors.textPrimary,
    marginBottom: Spacing.sm,
  },
  subtitle: {
    fontSize: FontSize.md,
    color: Colors.textSecondary,
    lineHeight: 22,
  },
  form: {
    gap: Spacing.md,
  },
  codeInput: {
    backgroundColor: Colors.card,
    borderWidth: 1,
    borderColor: Colors.cardBorder,
    borderRadius: 12,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    fontSize: 28,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    color: Colors.textPrimary,
    letterSpacing: 8,
    textAlign: 'center',
  },
  trustRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: Spacing.sm,
  },
  trustLabel: {
    fontSize: FontSize.sm,
    color: Colors.textSecondary,
  },
  button: {
    backgroundColor: Colors.gold,
    paddingVertical: Spacing.md,
    borderRadius: 12,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    fontSize: FontSize.lg,
    fontWeight: '600',
    color: Colors.navy,
  },
  links: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: Spacing.sm,
  },
  link: {
    fontSize: FontSize.sm,
    color: Colors.gold,
  },
  cancelLink: {
    fontSize: FontSize.sm,
    color: Colors.textMuted,
  },
});
