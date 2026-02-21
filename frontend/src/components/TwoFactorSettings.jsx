import React, { useState, useEffect, useRef } from 'react';
import { X, Shield, Copy, Check, AlertTriangle, Loader2 } from 'lucide-react';
import { QRCodeSVG } from 'qrcode.react';
import { useAuth } from '../contexts/AuthContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function TwoFactorSettings({ isOpen, onClose }) {
  const { fetchWithAuth, refreshUser } = useAuth();
  const [status, setStatus] = useState(null); // { totp_enabled, backup_codes_remaining }
  const [loading, setLoading] = useState(true);
  const [step, setStep] = useState('status'); // status | setup | confirm | disable
  const [setupData, setSetupData] = useState(null); // { secret, provisioning_uri, backup_codes }
  const [code, setCode] = useState('');
  const [error, setError] = useState('');
  const [actionLoading, setActionLoading] = useState(false);
  const [codesCopied, setCodesCopied] = useState(false);
  const [newBackupCodes, setNewBackupCodes] = useState(null);
  const codeInputRef = useRef(null);

  // Load 2FA status
  useEffect(() => {
    if (!isOpen) return;
    setStep('status');
    setError('');
    setCode('');
    setSetupData(null);
    setNewBackupCodes(null);
    (async () => {
      setLoading(true);
      try {
        const res = await fetchWithAuth(`${API_URL}/api/auth/2fa/status`);
        if (res.ok) {
          setStatus(await res.json());
        }
      } catch (err) {
        console.error('Failed to load 2FA status:', err);
      } finally {
        setLoading(false);
      }
    })();
  }, [isOpen, fetchWithAuth]);

  // Auto-focus code input
  useEffect(() => {
    if ((step === 'confirm' || step === 'disable' || step === 'regen') && codeInputRef.current) {
      codeInputRef.current.focus();
    }
  }, [step]);

  const handleSetup = async () => {
    setActionLoading(true);
    setError('');
    try {
      const res = await fetchWithAuth(`${API_URL}/api/auth/2fa/setup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Setup failed');
      setSetupData(data);
      setStep('setup');
    } catch (err) {
      setError(err.message);
    } finally {
      setActionLoading(false);
    }
  };

  const handleConfirm = async (e) => {
    e.preventDefault();
    setActionLoading(true);
    setError('');
    try {
      const res = await fetchWithAuth(`${API_URL}/api/auth/2fa/confirm-setup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Verification failed');
      setStatus({ totp_enabled: true, backup_codes_remaining: 10 });
      setStep('status');
      setCode('');
      setSetupData(null);
      refreshUser();
    } catch (err) {
      setError(err.message);
    } finally {
      setActionLoading(false);
    }
  };

  const handleDisable = async (e) => {
    e.preventDefault();
    setActionLoading(true);
    setError('');
    try {
      const res = await fetchWithAuth(`${API_URL}/api/auth/2fa/disable`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to disable');
      setStatus({ totp_enabled: false, backup_codes_remaining: 0 });
      setStep('status');
      setCode('');
      localStorage.removeItem('2fa_trust_token');
      refreshUser();
    } catch (err) {
      setError(err.message);
    } finally {
      setActionLoading(false);
    }
  };

  const handleRegenerate = async (e) => {
    e.preventDefault();
    setActionLoading(true);
    setError('');
    try {
      const res = await fetchWithAuth(`${API_URL}/api/auth/2fa/regenerate-backup-codes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code.trim() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to regenerate');
      setNewBackupCodes(data.backup_codes);
      setStatus(prev => ({ ...prev, backup_codes_remaining: 10 }));
      setCode('');
      setStep('status');
    } catch (err) {
      setError(err.message);
    } finally {
      setActionLoading(false);
    }
  };

  const copyBackupCodes = (codes) => {
    navigator.clipboard.writeText(codes.join('\n'));
    setCodesCopied(true);
    setTimeout(() => setCodesCopied(false), 2000);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-xl max-w-lg w-full max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between p-5 border-b">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <Shield size={18} /> Two-Factor Authentication
          </h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600"><X size={20} /></button>
        </div>

        <div className="p-5">
          {loading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
            </div>
          ) : step === 'status' ? (
            // Status view
            <div className="space-y-4">
              {status?.totp_enabled ? (
                <>
                  <div className="flex items-center gap-2 p-3 bg-green-50 border border-green-200 rounded-lg">
                    <Shield size={16} className="text-green-600" />
                    <span className="text-sm font-medium text-green-800">2FA is enabled</span>
                  </div>

                  {/* Show newly generated backup codes */}
                  {newBackupCodes && (
                    <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle size={16} className="text-amber-600" />
                        <span className="text-sm font-medium text-amber-800">New Backup Codes</span>
                      </div>
                      <p className="text-xs text-amber-700 mb-3">Save these codes somewhere safe. They won't be shown again.</p>
                      <div className="grid grid-cols-2 gap-1 mb-3">
                        {newBackupCodes.map((c, i) => (
                          <code key={i} className="text-sm font-mono bg-white px-2 py-1 rounded border text-gray-800">{c}</code>
                        ))}
                      </div>
                      <button
                        onClick={() => copyBackupCodes(newBackupCodes)}
                        className="flex items-center gap-1.5 text-sm text-amber-700 hover:text-amber-900"
                      >
                        {codesCopied ? <><Check size={14} /> Copied!</> : <><Copy size={14} /> Copy All</>}
                      </button>
                    </div>
                  )}

                  <div className="text-sm text-gray-600">
                    <p>Backup codes remaining: <span className="font-medium text-gray-900">{status.backup_codes_remaining}</span></p>
                  </div>

                  <div className="flex gap-3">
                    <button
                      onClick={() => { setStep('regen'); setCode(''); setError(''); }}
                      className="px-4 py-2 text-sm font-medium text-blue-600 border border-blue-200 rounded-lg hover:bg-blue-50"
                    >
                      Regenerate Backup Codes
                    </button>
                    <button
                      onClick={() => { setStep('disable'); setCode(''); setError(''); }}
                      className="px-4 py-2 text-sm font-medium text-red-600 border border-red-200 rounded-lg hover:bg-red-50"
                    >
                      Disable 2FA
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    Add an extra layer of security to your admin account with a TOTP authenticator app
                    like Google Authenticator or Authy.
                  </p>
                  <button
                    onClick={handleSetup}
                    disabled={actionLoading}
                    className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    {actionLoading ? 'Setting up...' : 'Enable 2FA'}
                  </button>
                </>
              )}

              {error && (
                <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">{error}</div>
              )}
            </div>
          ) : step === 'setup' ? (
            // Setup: show QR + secret + backup codes
            <div className="space-y-5">
              <div>
                <p className="text-sm font-medium text-gray-900 mb-2">1. Scan this QR code with your authenticator app:</p>
                <div className="flex justify-center p-4 bg-white border rounded-lg">
                  <QRCodeSVG value={setupData.provisioning_uri} size={200} />
                </div>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-900 mb-1">Or enter this key manually:</p>
                <div className="flex items-center gap-2">
                  <code className="flex-1 text-sm font-mono bg-gray-50 px-3 py-2 rounded border break-all">{setupData.secret}</code>
                  <button
                    onClick={() => { navigator.clipboard.writeText(setupData.secret); }}
                    className="p-2 text-gray-500 hover:text-gray-700"
                  >
                    <Copy size={16} />
                  </button>
                </div>
              </div>

              <div>
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle size={16} className="text-amber-600" />
                  <p className="text-sm font-medium text-gray-900">2. Save your backup codes:</p>
                </div>
                <p className="text-xs text-gray-500 mb-2">Use these if you lose access to your authenticator. Each code can only be used once.</p>
                <div className="grid grid-cols-2 gap-1 mb-2">
                  {setupData.backup_codes.map((c, i) => (
                    <code key={i} className="text-sm font-mono bg-gray-50 px-2 py-1 rounded border text-gray-800">{c}</code>
                  ))}
                </div>
                <button
                  onClick={() => copyBackupCodes(setupData.backup_codes)}
                  className="flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-700"
                >
                  {codesCopied ? <><Check size={14} /> Copied!</> : <><Copy size={14} /> Copy All</>}
                </button>
              </div>

              <div>
                <p className="text-sm font-medium text-gray-900 mb-2">3. Enter a code from your authenticator to confirm:</p>
                {error && (
                  <div className="mb-2 p-2 bg-red-50 border border-red-200 text-red-700 rounded text-sm">{error}</div>
                )}
                <form onSubmit={handleConfirm} className="flex gap-2">
                  <input
                    ref={codeInputRef}
                    type="text"
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    placeholder="000000"
                    maxLength={6}
                    inputMode="numeric"
                    autoComplete="one-time-code"
                    className="flex-1 px-4 py-2 border rounded-lg text-center font-mono text-lg tracking-widest focus:ring-2 focus:ring-blue-500"
                  />
                  <button
                    type="submit"
                    disabled={actionLoading || code.length !== 6}
                    className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    {actionLoading ? 'Verifying...' : 'Confirm'}
                  </button>
                </form>
              </div>

              <button
                onClick={() => { setStep('status'); setError(''); }}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Cancel setup
              </button>
            </div>
          ) : step === 'disable' ? (
            // Disable: confirm with TOTP code
            <div className="space-y-4">
              <p className="text-sm text-gray-600">Enter your current authenticator code to disable 2FA.</p>
              {error && (
                <div className="p-2 bg-red-50 border border-red-200 text-red-700 rounded text-sm">{error}</div>
              )}
              <form onSubmit={handleDisable} className="space-y-3">
                <input
                  ref={codeInputRef}
                  type="text"
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  placeholder="000000"
                  maxLength={6}
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  className="w-full px-4 py-2 border rounded-lg text-center font-mono text-lg tracking-widest focus:ring-2 focus:ring-blue-500"
                />
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={() => { setStep('status'); setError(''); }}
                    className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={actionLoading || code.length !== 6}
                    className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 disabled:opacity-50"
                  >
                    {actionLoading ? 'Disabling...' : 'Disable 2FA'}
                  </button>
                </div>
              </form>
            </div>
          ) : step === 'regen' ? (
            // Regenerate backup codes: confirm with TOTP code
            <div className="space-y-4">
              <p className="text-sm text-gray-600">Enter your current authenticator code to regenerate backup codes. This will invalidate all existing backup codes.</p>
              {error && (
                <div className="p-2 bg-red-50 border border-red-200 text-red-700 rounded text-sm">{error}</div>
              )}
              <form onSubmit={handleRegenerate} className="space-y-3">
                <input
                  ref={codeInputRef}
                  type="text"
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  placeholder="000000"
                  maxLength={6}
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  className="w-full px-4 py-2 border rounded-lg text-center font-mono text-lg tracking-widest focus:ring-2 focus:ring-blue-500"
                />
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={() => { setStep('status'); setError(''); }}
                    className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={actionLoading || code.length !== 6}
                    className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                  >
                    {actionLoading ? 'Regenerating...' : 'Regenerate Codes'}
                  </button>
                </div>
              </form>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
