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
} from '@/services/auth';
import { registerForPushNotifications } from '@/services/notifications';

interface AuthContextType {
  user: AuthUser | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  loginWithGoogle: (idToken: string) => Promise<void>;
  loginWithApple: (
    identityToken: string,
    fullName?: { givenName?: string | null; familyName?: string | null }
  ) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // On mount, check for stored tokens and restore session
  useEffect(() => {
    (async () => {
      try {
        const hasTokens = await hasStoredTokens();
        if (hasTokens) {
          const profile = await getProfile();
          setUser(profile);
          // Re-register push token on app launch
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
    const u = await authLogin(email, password);
    setUser(u);
    registerForPushNotifications().catch(() => {});
  };

  const register = async (email: string, password: string, name: string) => {
    const u = await authRegister(email, password, name);
    setUser(u);
    registerForPushNotifications().catch(() => {});
  };

  const loginWithGoogle = async (idToken: string) => {
    const u = await authGoogle(idToken);
    setUser(u);
    registerForPushNotifications().catch(() => {});
  };

  const loginWithApple = async (
    identityToken: string,
    fullName?: { givenName?: string | null; familyName?: string | null }
  ) => {
    const u = await authApple(identityToken, fullName);
    setUser(u);
    registerForPushNotifications().catch(() => {});
  };

  const logout = async () => {
    await authLogout();
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        login,
        register,
        loginWithGoogle,
        loginWithApple,
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
