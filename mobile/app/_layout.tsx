/**
 * Root layout — wraps entire app with auth provider and handles
 * routing between auth and main app based on login state.
 */

import React, { useEffect } from 'react';
import { Stack, useRouter, useSegments } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { ActivityIndicator, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import * as Notifications from 'expo-notifications';
import * as Updates from 'expo-updates';
import * as ScreenOrientation from 'expo-screen-orientation';
import { AuthProvider, useAuth } from '@/hooks/useAuth';
import { Colors } from '@/constants/theme';

function RootNavigator() {
  const { user, isLoading, twoFactorRequired } = useAuth();
  const segments = useSegments();
  const router = useRouter();

  useEffect(() => {
    if (isLoading) return;

    const inAuthGroup = segments[0] === '(auth)';

    // 2FA required — redirect to verify screen
    if (twoFactorRequired) {
      const currentScreen = (segments as string[])[1];
      if (currentScreen !== 'verify-2fa') {
        router.replace('/(auth)/verify-2fa' as any);
      }
      return;
    }

    if (!user && !inAuthGroup) {
      router.replace('/(auth)/login');
    } else if (user && inAuthGroup) {
      router.replace('/(tabs)/dashboard');
    }
  }, [user, isLoading, segments, twoFactorRequired]);

  // Handle notification taps — navigate to relevant screen
  useEffect(() => {
    const sub = Notifications.addNotificationResponseReceivedListener(
      (response) => {
        const data = response.notification.request.content.data;
        if (data?.screen === 'dashboard') {
          router.push('/(tabs)/dashboard');
        } else if (data?.screen === 'signal_detail' && data?.symbol) {
          router.push(`/signal/${data.symbol}`);
        }
      }
    );
    return () => sub.remove();
  }, []);

  if (isLoading) {
    return (
      <View
        style={{
          flex: 1,
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: Colors.background,
        }}
      >
        <ActivityIndicator size="large" color={Colors.gold} />
      </View>
    );
  }

  return (
    <Stack
      screenOptions={{
        headerShown: false,
        contentStyle: { backgroundColor: Colors.background },
      }}
    />
  );
}

export default function RootLayout() {
  // Lock to portrait by default — signal detail unlocks landscape for chart
  useEffect(() => {
    ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.PORTRAIT_UP);
  }, []);

  // Check for OTA updates on launch
  useEffect(() => {
    if (__DEV__) return;
    (async () => {
      try {
        console.log('[OTA] Checking for updates...');
        const update = await Updates.checkForUpdateAsync();
        console.log('[OTA] Update available:', update.isAvailable);
        if (update.isAvailable) {
          console.log('[OTA] Fetching update...');
          await Updates.fetchUpdateAsync();
          console.log('[OTA] Reloading...');
          await Updates.reloadAsync();
        }
      } catch (e) {
        console.log('[OTA] Error:', e);
      }
    })();
  }, []);

  return (
    <SafeAreaProvider>
      <AuthProvider>
        <StatusBar style="light" />
        <RootNavigator />
      </AuthProvider>
    </SafeAreaProvider>
  );
}
