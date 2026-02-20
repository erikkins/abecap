/**
 * Authentication service — login, register, token management.
 */

import * as SecureStore from 'expo-secure-store';
import api from './api';

export interface AuthUser {
  id: string;
  email: string;
  name: string | null;
  role: string;
  subscription?: {
    status: string;
    is_valid: boolean;
    days_remaining: number;
  };
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
}

async function storeTokens(tokens: AuthTokens) {
  await SecureStore.setItemAsync('access_token', tokens.access_token);
  await SecureStore.setItemAsync('refresh_token', tokens.refresh_token);
}

export async function login(
  email: string,
  password: string
): Promise<AuthUser> {
  const { data } = await api.post('/api/auth/login', { email, password });
  await storeTokens(data);
  return data.user;
}

export async function register(
  email: string,
  password: string,
  name: string
): Promise<AuthUser> {
  const { data } = await api.post('/api/auth/register', {
    email,
    password,
    name,
    turnstile_token: 'mobile-app', // Backend allows mobile bypass
  });
  await storeTokens(data);
  return data.user;
}

export async function loginWithGoogle(idToken: string): Promise<AuthUser> {
  const { data } = await api.post('/api/auth/google', { id_token: idToken });
  await storeTokens(data);
  return data.user;
}

export async function loginWithApple(
  identityToken: string,
  fullName?: { givenName?: string | null; familyName?: string | null }
): Promise<AuthUser> {
  const userData =
    fullName?.givenName || fullName?.familyName
      ? {
          name: {
            firstName: fullName.givenName || '',
            lastName: fullName.familyName || '',
          },
        }
      : undefined;
  const { data } = await api.post('/api/auth/apple', {
    id_token: identityToken,
    user_data: userData,
  });
  await storeTokens(data);
  return data.user;
}

export async function getProfile(): Promise<AuthUser> {
  const { data } = await api.get('/api/auth/me');
  return data;
}

export async function logout() {
  await SecureStore.deleteItemAsync('access_token');
  await SecureStore.deleteItemAsync('refresh_token');
}

export async function hasStoredTokens(): Promise<boolean> {
  const token = await SecureStore.getItemAsync('access_token');
  return !!token;
}
