import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const AuthContext = createContext(null);

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Get stored tokens
  const getTokens = useCallback(() => {
    const accessToken = localStorage.getItem('accessToken');
    const refreshToken = localStorage.getItem('refreshToken');
    return { accessToken, refreshToken };
  }, []);

  // Store tokens
  const setTokens = useCallback((accessToken, refreshToken) => {
    if (accessToken) {
      localStorage.setItem('accessToken', accessToken);
    }
    if (refreshToken) {
      localStorage.setItem('refreshToken', refreshToken);
    }
  }, []);

  // Clear tokens
  const clearTokens = useCallback(() => {
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
  }, []);

  // Refresh access token
  const refreshAccessToken = useCallback(async () => {
    const { refreshToken } = getTokens();
    if (!refreshToken) {
      return null;
    }

    try {
      const response = await fetch(`${API_URL}/api/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (!response.ok) {
        throw new Error('Refresh failed');
      }

      const data = await response.json();
      setTokens(data.access_token, data.refresh_token);
      setUser(data.user);
      return data.access_token;
    } catch (err) {
      clearTokens();
      setUser(null);
      return null;
    }
  }, [getTokens, setTokens, clearTokens]);

  // Fetch with auth (auto-refresh on 401)
  const fetchWithAuth = useCallback(async (url, options = {}) => {
    const { accessToken } = getTokens();

    const headers = {
      ...options.headers,
      'Authorization': `Bearer ${accessToken}`,
    };

    let response = await fetch(url, { ...options, headers });

    // If 401, try to refresh and retry
    if (response.status === 401) {
      const newToken = await refreshAccessToken();
      if (newToken) {
        headers['Authorization'] = `Bearer ${newToken}`;
        response = await fetch(url, { ...options, headers });
      }
    }

    return response;
  }, [getTokens, refreshAccessToken]);

  // Load user on mount
  useEffect(() => {
    const loadUser = async () => {
      const { accessToken } = getTokens();
      if (!accessToken) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(`${API_URL}/api/auth/me`, {
          headers: { 'Authorization': `Bearer ${accessToken}` },
        });

        if (response.ok) {
          const userData = await response.json();
          setUser(userData);
        } else if (response.status === 401) {
          // Try refresh
          await refreshAccessToken();
        } else {
          clearTokens();
        }
      } catch (err) {
        console.error('Failed to load user:', err);
        clearTokens();
      } finally {
        setLoading(false);
      }
    };

    loadUser();
  }, [getTokens, clearTokens, refreshAccessToken]);

  // Register
  const register = async (email, password, name, turnstileToken) => {
    setError(null);
    try {
      const response = await fetch(`${API_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email,
          password,
          name,
          turnstile_token: turnstileToken,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Registration failed');
      }

      setTokens(data.access_token, data.refresh_token);
      setUser(data.user);
      return { success: true };
    } catch (err) {
      setError(err.message);
      return { success: false, error: err.message };
    }
  };

  // Login
  const login = async (email, password) => {
    console.log('AuthContext: login called for', email);
    console.log('AuthContext: API_URL is', API_URL);
    setError(null);

    const loginUrl = `${API_URL}/api/auth/login`;
    console.log('AuthContext: Full login URL:', loginUrl);

    try {
      console.log('AuthContext: Creating request body');
      const body = JSON.stringify({ email, password });
      console.log('AuthContext: Request body:', body);

      console.log('AuthContext: Making fetch call...');
      const response = await fetch(loginUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: body,
      });

      console.log('AuthContext: Fetch completed, status:', response.status);

      let data;
      try {
        data = await response.json();
        console.log('AuthContext: Response JSON:', data);
      } catch (parseErr) {
        console.error('AuthContext: Failed to parse JSON response:', parseErr);
        throw new Error('Invalid response from server');
      }

      if (!response.ok) {
        console.log('AuthContext: Response not OK, throwing error');
        throw new Error(data.detail || 'Login failed');
      }

      console.log('AuthContext: Login successful, setting tokens and user');
      setTokens(data.access_token, data.refresh_token);
      setUser(data.user);
      console.log('AuthContext: User set, returning success');
      return { success: true };
    } catch (err) {
      console.error('AuthContext: Login error name:', err.name);
      console.error('AuthContext: Login error message:', err.message);
      console.error('AuthContext: Login error stack:', err.stack);
      setError(err.message);
      return { success: false, error: err.message };
    }
  };

  // Google OAuth
  const loginWithGoogle = async (idToken, turnstileToken = null) => {
    setError(null);
    try {
      const response = await fetch(`${API_URL}/api/auth/google`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id_token: idToken,
          turnstile_token: turnstileToken,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Google login failed');
      }

      setTokens(data.access_token, data.refresh_token);
      setUser(data.user);
      return { success: true };
    } catch (err) {
      setError(err.message);
      return { success: false, error: err.message };
    }
  };

  // Apple OAuth
  const loginWithApple = async (idToken, userData = null, turnstileToken = null) => {
    setError(null);
    try {
      const response = await fetch(`${API_URL}/api/auth/apple`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id_token: idToken,
          user_data: userData,
          turnstile_token: turnstileToken,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Apple login failed');
      }

      setTokens(data.access_token, data.refresh_token);
      setUser(data.user);
      return { success: true };
    } catch (err) {
      setError(err.message);
      return { success: false, error: err.message };
    }
  };

  // Logout
  const logout = useCallback(async () => {
    try {
      const { accessToken } = getTokens();
      if (accessToken) {
        await fetch(`${API_URL}/api/auth/logout`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${accessToken}` },
        });
      }
    } catch (err) {
      console.error('Logout error:', err);
    } finally {
      clearTokens();
      setUser(null);
    }
  }, [getTokens, clearTokens]);

  // Check if user is admin
  const isAdmin = user?.role === 'admin' && user?.email === 'erik@rigacap.com';

  // Check if subscription is valid
  const hasValidSubscription = user?.subscription?.is_valid ?? false;

  // Get trial days remaining
  const trialDaysRemaining = user?.subscription?.days_remaining ?? 0;

  const value = {
    user,
    loading,
    error,
    isAuthenticated: !!user,
    isAdmin,
    hasValidSubscription,
    trialDaysRemaining,
    register,
    login,
    loginWithGoogle,
    loginWithApple,
    logout,
    fetchWithAuth,
    clearError: () => setError(null),
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export default AuthContext;
