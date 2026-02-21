import React, { useState, useEffect, useRef } from 'react';
import { X, Mail, Lock, User, Eye, EyeOff, Chrome, Apple, Shield } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const TURNSTILE_SITE_KEY = import.meta.env.VITE_TURNSTILE_SITE_KEY;
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

export default function LoginModal({ isOpen = true, onClose, onSuccess, initialMode = 'login', selectedPlan = 'monthly' }) {
  const { login, register, loginWithGoogle, loginWithApple, verify2FA, cancel2FA, twoFactorRequired, error, clearError } = useAuth();
  const [mode, setMode] = useState(initialMode);
  const [twoFactorCode, setTwoFactorCode] = useState('');
  const [trustDevice, setTrustDevice] = useState(false);
  const [useBackupCode, setUseBackupCode] = useState(false);
  const [twoFactorLoading, setTwoFactorLoading] = useState(false);
  const [twoFactorError, setTwoFactorError] = useState('');
  const twoFactorInputRef = useRef(null);

  // Store selected plan in localStorage for use during checkout
  useEffect(() => {
    if (selectedPlan) {
      localStorage.setItem('rigacap_selected_plan', selectedPlan);
    }
  }, [selectedPlan]);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState('');
  const [turnstileToken, setTurnstileToken] = useState('');
  const turnstileRef = useRef(null);

  // Reset form when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setEmail('');
      setPassword('');
      setName('');
      setLocalError('');
      clearError();
      setTurnstileToken('');
      setMode(initialMode); // Reset mode based on visitor type
    }
  }, [isOpen, clearError, initialMode]);

  // Load Turnstile widget
  useEffect(() => {
    if (!isOpen || !TURNSTILE_SITE_KEY || mode !== 'register') return;

    const loadTurnstile = () => {
      if (window.turnstile && turnstileRef.current) {
        window.turnstile.render(turnstileRef.current, {
          sitekey: TURNSTILE_SITE_KEY,
          callback: (token) => setTurnstileToken(token),
          'error-callback': () => setTurnstileToken(''),
        });
      }
    };

    // Wait for turnstile to load
    if (window.turnstile) {
      loadTurnstile();
    } else {
      const checkInterval = setInterval(() => {
        if (window.turnstile) {
          loadTurnstile();
          clearInterval(checkInterval);
        }
      }, 100);
      return () => clearInterval(checkInterval);
    }
  }, [isOpen, mode]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setLocalError('');

    try {
      if (mode === 'register') {
        if (!turnstileToken && TURNSTILE_SITE_KEY) {
          setLocalError('Please complete the verification');
          setLoading(false);
          return;
        }
        const result = await register(email, password, name, turnstileToken || 'dev-bypass');
        if (result.success) {
          if (!result.redirecting) {
            onSuccess ? onSuccess() : onClose();
          }
        } else {
          setLocalError(result.error);
        }
      } else {
        const result = await login(email, password);
        if (result.success) {
          if (result.requires_2fa) {
            // 2FA step will be shown automatically via twoFactorRequired state
            return;
          }
          if (onSuccess) {
            onSuccess();
          } else if (onClose) {
            onClose();
          }
        } else {
          setLocalError(result.error || 'Login failed');
        }
      }
    } catch (err) {
      setLocalError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    if (!GOOGLE_CLIENT_ID) {
      setLocalError('Google Sign-In is not configured.');
      return;
    }

    try {
      // Use Google Identity Services (GIS)
      const google = window.google;
      if (!google?.accounts?.id) {
        setLocalError('Google Sign-In SDK not loaded. Please refresh and try again.');
        return;
      }

      // Initialize and prompt
      google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: async (response) => {
          if (response.credential) {
            setLoading(true);
            const result = await loginWithGoogle(response.credential);
            setLoading(false);
            if (result.success) {
              if (result.requires_2fa) return;
              if (!result.redirecting) {
                onSuccess ? onSuccess() : onClose();
              }
            } else {
              setLocalError(result.error || 'Google login failed');
            }
          }
        },
      });

      // Show the One Tap or popup
      google.accounts.id.prompt((notification) => {
        if (notification.isNotDisplayed() || notification.isSkippedMoment()) {
          // Fallback: use popup flow
          google.accounts.id.renderButton(
            document.getElementById('google-signin-button'),
            { theme: 'outline', size: 'large', width: '100%' }
          );
        }
      });
    } catch (err) {
      console.error('Google login error:', err);
      setLocalError('Google Sign-In failed. Please try again.');
    }
  };

  const handleAppleLogin = async () => {
    const APPLE_CLIENT_ID = import.meta.env.VITE_APPLE_CLIENT_ID;
    if (!APPLE_CLIENT_ID) {
      setLocalError('Apple Sign-In is not configured.');
      return;
    }

    try {
      if (!window.AppleID) {
        setLocalError('Apple Sign-In SDK not loaded. Please refresh and try again.');
        return;
      }

      window.AppleID.auth.init({
        clientId: APPLE_CLIENT_ID,
        scope: 'name email',
        redirectURI: `${window.location.origin}/auth/apple/callback`,
        usePopup: true,
      });

      const response = await window.AppleID.auth.signIn();
      const idToken = response.authorization.id_token;
      const userData = response.user || null;

      setLoading(true);
      const result = await loginWithApple(idToken, userData);
      setLoading(false);

      if (result.success) {
        if (result.requires_2fa) return;
        if (!result.redirecting) {
          onSuccess ? onSuccess() : onClose();
        }
      } else {
        setLocalError(result.error || 'Apple login failed');
      }
    } catch (err) {
      setLoading(false);
      if (err.error === 'popup_closed_by_user') return;
      console.error('Apple login error:', err);
      setLocalError('Apple Sign-In failed. Please try again.');
    }
  };

  // Handle 2FA code submission
  const handle2FASubmit = async (e) => {
    e.preventDefault();
    setTwoFactorLoading(true);
    setTwoFactorError('');
    try {
      const result = await verify2FA(twoFactorCode, trustDevice, useBackupCode);
      if (result.success) {
        setTwoFactorCode('');
        setTrustDevice(false);
        setUseBackupCode(false);
        onSuccess ? onSuccess() : onClose();
      } else {
        setTwoFactorError(result.error || 'Invalid code');
      }
    } catch (err) {
      setTwoFactorError(err.message);
    } finally {
      setTwoFactorLoading(false);
    }
  };

  // Auto-focus 2FA input
  useEffect(() => {
    if (twoFactorRequired && twoFactorInputRef.current) {
      twoFactorInputRef.current.focus();
    }
  }, [twoFactorRequired, useBackupCode]);

  if (!isOpen) return null;

  // Show 2FA verification step
  if (twoFactorRequired) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-2xl shadow-xl max-w-md w-full overflow-hidden">
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-4 flex justify-between items-center">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Shield size={20} /> Two-Factor Authentication
            </h2>
            <button
              onClick={() => { cancel2FA(); setTwoFactorCode(''); setTwoFactorError(''); setUseBackupCode(false); }}
              className="text-white/80 hover:text-white transition-colors"
            >
              <X size={24} />
            </button>
          </div>
          <div className="p-6">
            <p className="text-sm text-gray-600 mb-4">
              {useBackupCode
                ? 'Enter one of your 8-character backup codes.'
                : 'Enter the 6-digit code from your authenticator app.'}
            </p>

            {(twoFactorError || error) && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">
                {twoFactorError || error}
              </div>
            )}

            <form onSubmit={handle2FASubmit} className="space-y-4">
              <div>
                <input
                  ref={twoFactorInputRef}
                  type="text"
                  value={twoFactorCode}
                  onChange={(e) => setTwoFactorCode(e.target.value)}
                  placeholder={useBackupCode ? 'ABCD1234' : '000000'}
                  maxLength={useBackupCode ? 8 : 6}
                  autoComplete="one-time-code"
                  inputMode={useBackupCode ? 'text' : 'numeric'}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg text-center text-2xl font-mono tracking-widest focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={trustDevice}
                  onChange={(e) => setTrustDevice(e.target.checked)}
                  className="w-4 h-4 text-blue-600 rounded"
                />
                <span className="text-sm text-gray-600">Trust this device for 30 days</span>
              </label>

              <button
                type="submit"
                disabled={twoFactorLoading || (!useBackupCode && twoFactorCode.length !== 6) || (useBackupCode && twoFactorCode.length < 8)}
                className="w-full py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {twoFactorLoading ? 'Verifying...' : 'Verify'}
              </button>
            </form>

            <div className="mt-4 flex justify-between text-sm">
              <button
                onClick={() => { setUseBackupCode(!useBackupCode); setTwoFactorCode(''); setTwoFactorError(''); }}
                className="text-blue-600 hover:text-blue-700"
              >
                {useBackupCode ? 'Use authenticator app' : 'Use a backup code'}
              </button>
              <button
                onClick={() => { cancel2FA(); setTwoFactorCode(''); setTwoFactorError(''); setUseBackupCode(false); }}
                className="text-gray-500 hover:text-gray-700"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl max-w-md w-full overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-4 flex justify-between items-center">
          <h2 className="text-xl font-bold text-white">
            {mode === 'login' ? 'Welcome Back' : 'Start Your Free Trial'}
          </h2>
          <button
            onClick={onClose}
            className="text-white/80 hover:text-white transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* OAuth buttons */}
          <div className="space-y-3 mb-6">
            <div id="google-signin-button" className="w-full">
              <button
                onClick={handleGoogleLogin}
                className="w-full flex items-center justify-center gap-3 px-4 py-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <Chrome size={20} className="text-gray-600" />
                <span className="font-medium text-gray-700">Continue with Google</span>
              </button>
            </div>
            <button
              onClick={handleAppleLogin}
              className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-black text-white rounded-lg hover:bg-gray-800 transition-colors"
            >
              <Apple size={20} />
              <span className="font-medium">Continue with Apple</span>
            </button>
          </div>

          {/* Divider */}
          <div className="relative mb-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-200"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white text-gray-500">or continue with email</span>
            </div>
          </div>

          {/* Error message */}
          {(localError || error) && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">
              {localError || error}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === 'register' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Full Name
                </label>
                <div className="relative">
                  <User size={20} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="John Doe"
                    className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Email
              </label>
              <div className="relative">
                <Mail size={20} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  required
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Password
              </label>
              <div className="relative">
                <Lock size={20} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  minLength={8}
                  className="w-full pl-10 pr-12 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </button>
              </div>
              {mode === 'register' && (
                <p className="text-xs text-gray-500 mt-1">Must be at least 8 characters</p>
              )}
              {mode === 'login' && (
                <div className="text-right mt-1">
                  <button
                    type="button"
                    onClick={() => {
                      onClose();
                      window.location.href = '/forgot-password';
                    }}
                    className="text-xs text-blue-600 hover:text-blue-700"
                  >
                    Forgot password?
                  </button>
                </div>
              )}
            </div>

            {/* Turnstile widget for registration */}
            {mode === 'register' && TURNSTILE_SITE_KEY && (
              <div className="flex justify-center">
                <div ref={turnstileRef}></div>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  {mode === 'login' ? 'Signing in...' : 'Setting up trial...'}
                </span>
              ) : (
                mode === 'login' ? 'Sign In' : 'Start 7-Day Free Trial'
              )}
            </button>
          </form>

          {/* Toggle mode */}
          <div className="mt-6 text-center text-sm text-gray-600">
            {mode === 'login' ? (
              <>
                Don't have an account?{' '}
                <button
                  onClick={() => setMode('register')}
                  className="text-blue-600 hover:text-blue-700 font-medium"
                >
                  Start free trial
                </button>
              </>
            ) : (
              <>
                Already have an account?{' '}
                <button
                  onClick={() => setMode('login')}
                  className="text-blue-600 hover:text-blue-700 font-medium"
                >
                  Sign in
                </button>
              </>
            )}
          </div>

          {/* Trial info */}
          {mode === 'register' && (
            <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-green-800 text-center">
                <span className="font-semibold">7-day free trial</span> · Credit card required
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
