/**
 * Auth context — provides user state and login/logout to the entire app.
 */

import React, { createContext, useContext, useEffect, useState } from 'react';
import {
  AuthUser,
  getProfile,
  hasStoredTokens,
  login as authLogin,
  loginWithApple as authApple,
  loginWithGoogle as authGoogle,
  logout as authLogout,
  register as authRegister,
  verify2FA as authVerify2FA,
} from '@/services/auth';
import { registerForPushNotifications } from '@/services/notifications';

interface AuthContextType {
  user: AuthUser | null;
  isLoading: boolean;
  twoFactorRequired: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  loginWithGoogle: (idToken: string) => Promise<void>;
  loginWithApple: (
    identityToken: string,
    fullName?: { givenName?: string | null; familyName?: string | null }
  ) => Promise<void>;
  verify2FA: (code: string, trustDevice: boolean, isBackupCode: boolean) => Promise<void>;
  cancel2FA: () => void;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [twoFactorRequired, setTwoFactorRequired] = useState(false);
  const [challengeToken, setChallengeToken] = useState<string | null>(null);

  // On mount, check for stored tokens and restore session
  useEffect(() => {
    (async () => {
      try {
        const hasTokens = await hasStoredTokens();
        if (hasTokens) {
          const profile = await getProfile();
          setUser(profile);
          registerForPushNotifications().catch(() => {});
        }
      } catch {
        // Token expired or invalid — stay logged out
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  const login = async (email: string, password: string) => {
    const result = await authLogin(email, password);
    if (result.requires_2fa && result.challenge_token) {
      setChallengeToken(result.challenge_token);
      setTwoFactorRequired(true);
      return;
    }
    if (result.user) {
      setUser(result.user);
      registerForPushNotifications().catch(() => {});
    }
  };

  const register = async (email: string, password: string, name: string) => {
    const result = await authRegister(email, password, name);
    if (result.user) {
      setUser(result.user);
      registerForPushNotifications().catch(() => {});
    }
  };

  const loginWithGoogle = async (idToken: string) => {
    const result = await authGoogle(idToken);
    if (result.requires_2fa && result.challenge_token) {
      setChallengeToken(result.challenge_token);
      setTwoFactorRequired(true);
      return;
    }
    if (result.user) {
      setUser(result.user);
      registerForPushNotifications().catch(() => {});
    }
  };

  const loginWithApple = async (
    identityToken: string,
    fullName?: { givenName?: string | null; familyName?: string | null }
  ) => {
    const result = await authApple(identityToken, fullName);
    if (result.requires_2fa && result.challenge_token) {
      setChallengeToken(result.challenge_token);
      setTwoFactorRequired(true);
      return;
    }
    if (result.user) {
      setUser(result.user);
      registerForPushNotifications().catch(() => {});
    }
  };

  const verify2FA = async (code: string, trustDevice: boolean, isBackupCode: boolean) => {
    if (!challengeToken) throw new Error('No 2FA challenge active');
    const u = await authVerify2FA(challengeToken, code, trustDevice, isBackupCode);
    setUser(u);
    setTwoFactorRequired(false);
    setChallengeToken(null);
    registerForPushNotifications().catch(() => {});
  };

  const cancel2FA = () => {
    setTwoFactorRequired(false);
    setChallengeToken(null);
  };

  const logout = async () => {
    await authLogout();
    setUser(null);
    setTwoFactorRequired(false);
    setChallengeToken(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        twoFactorRequired,
        login,
        register,
        loginWithGoogle,
        loginWithApple,
        verify2FA,
        cancel2FA,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
