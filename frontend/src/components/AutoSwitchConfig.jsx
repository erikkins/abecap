import React, { useState, useEffect } from 'react';
import { Settings, ToggleLeft, ToggleRight, RefreshCw, Save, History, ArrowRight, AlertCircle, CheckCircle } from 'lucide-react';

import { formatDate } from '../utils/formatDate';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function AutoSwitchConfig({ fetchWithAuth }) {
  const [config, setConfig] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  // Local form state
  const [formState, setFormState] = useState({
    is_enabled: false,
    analysis_frequency: 'biweekly',
    min_score_diff_to_switch: 10,
    min_days_since_last_switch: 14,
    notify_on_analysis: true,
    notify_on_switch: true,
    admin_email: ''
  });

  useEffect(() => {
    fetchConfig();
    fetchHistory();
  }, []);

  const fetchConfig = async () => {
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/strategies/auto-switch/config`);
      if (response.ok) {
        const data = await response.json();
        setConfig(data);
        setFormState(data);
      }
    } catch (err) {
      setError('Failed to load configuration');
    } finally {
      setLoading(false);
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/strategies/switch-history?limit=10`);
      if (response.ok) {
        const data = await response.json();
        setHistory(data.history || []);
      }
    } catch (err) {
      console.error('Failed to fetch history:', err);
    }
  };

  const saveConfig = async () => {
    setSaving(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const params = new URLSearchParams();
      if (formState.is_enabled !== config?.is_enabled) params.set('is_enabled', formState.is_enabled);
      if (formState.analysis_frequency !== config?.analysis_frequency) params.set('analysis_frequency', formState.analysis_frequency);
      if (formState.min_score_diff_to_switch !== config?.min_score_diff_to_switch) params.set('min_score_diff', formState.min_score_diff_to_switch);
      if (formState.min_days_since_last_switch !== config?.min_days_since_last_switch) params.set('min_days_cooldown', formState.min_days_since_last_switch);
      if (formState.notify_on_analysis !== config?.notify_on_analysis) params.set('notify_on_analysis', formState.notify_on_analysis);
      if (formState.notify_on_switch !== config?.notify_on_switch) params.set('notify_on_switch', formState.notify_on_switch);
      if (formState.admin_email !== config?.admin_email) params.set('admin_email', formState.admin_email);

      const response = await fetchWithAuth(
        `${API_URL}/api/admin/strategies/auto-switch/config?${params}`,
        { method: 'PATCH' }
      );

      if (response.ok) {
        const data = await response.json();
        setConfig(data.config);
        setSuccessMessage('Configuration saved successfully');
        setTimeout(() => setSuccessMessage(null), 3000);
      } else {
        const err = await response.json();
        setError(err.detail || 'Failed to save configuration');
      }
    } catch (err) {
      setError(err.message || 'Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const triggerAnalysis = async () => {
    try {
      const response = await fetchWithAuth(
        `${API_URL}/api/admin/strategies/auto-switch/trigger`,
        { method: 'POST' }
      );
      if (response.ok) {
        const data = await response.json();
        alert(`Analysis complete!\n\nRecommendation: ${data.recommendation_notes}\n\nSwitch recommended: ${data.switch_recommended ? 'Yes' : 'No'}\nSafeguards pass: ${data.safeguards_pass ? 'Yes' : 'No'}`);
        fetchHistory();
      }
    } catch (err) {
      alert('Failed to trigger analysis');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-8 h-8 text-blue-600 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Configuration Card */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-amber-50 to-orange-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-amber-100 rounded-lg">
                <Settings className="w-5 h-5 text-amber-600" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Auto-Switch Configuration</h3>
                <p className="text-sm text-gray-600">Configure automated strategy switching</p>
              </div>
            </div>
            <button
              onClick={() => setFormState(prev => ({ ...prev, is_enabled: !prev.is_enabled }))}
              className={`p-2 rounded-lg transition-colors ${
                formState.is_enabled
                  ? 'bg-emerald-100 text-emerald-600'
                  : 'bg-gray-100 text-gray-400'
              }`}
            >
              {formState.is_enabled ? <ToggleRight size={28} /> : <ToggleLeft size={28} />}
            </button>
          </div>
        </div>

        {/* Status Banner */}
        <div className={`px-6 py-3 ${formState.is_enabled ? 'bg-emerald-50' : 'bg-gray-50'}`}>
          <div className="flex items-center gap-2">
            {formState.is_enabled ? (
              <CheckCircle size={18} className="text-emerald-600" />
            ) : (
              <AlertCircle size={18} className="text-gray-400" />
            )}
            <span className={`text-sm font-medium ${formState.is_enabled ? 'text-emerald-700' : 'text-gray-600'}`}>
              Auto-switch is {formState.is_enabled ? 'enabled' : 'disabled'}
            </span>
          </div>
        </div>

        {/* Configuration Form */}
        <div className="p-6 space-y-6">
          {/* Analysis Frequency */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Analysis Frequency
            </label>
            <select
              value={formState.analysis_frequency}
              onChange={(e) => setFormState(prev => ({ ...prev, analysis_frequency: e.target.value }))}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
            >
              <option value="weekly">Weekly (every Friday)</option>
              <option value="biweekly">Biweekly (every other Friday)</option>
              <option value="monthly">Monthly (first Friday)</option>
            </select>
          </div>

          {/* Safeguard Settings */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Minimum Score Difference
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="0"
                  max="30"
                  value={formState.min_score_diff_to_switch}
                  onChange={(e) => setFormState(prev => ({ ...prev, min_score_diff_to_switch: parseFloat(e.target.value) }))}
                  className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-600"
                />
                <span className="text-sm font-medium text-gray-900 w-12">
                  {formState.min_score_diff_to_switch}
                </span>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Only switch if new strategy scores at least this much higher
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Cooldown Period (days)
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="7"
                  max="60"
                  value={formState.min_days_since_last_switch}
                  onChange={(e) => setFormState(prev => ({ ...prev, min_days_since_last_switch: parseInt(e.target.value) }))}
                  className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-600"
                />
                <span className="text-sm font-medium text-gray-900 w-12">
                  {formState.min_days_since_last_switch}d
                </span>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Minimum days between automatic switches
              </p>
            </div>
          </div>

          {/* Notification Settings */}
          <div className="border-t border-gray-200 pt-6">
            <h4 className="font-medium text-gray-900 mb-4">Email Notifications</h4>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Admin Email
                </label>
                <input
                  type="email"
                  value={formState.admin_email || ''}
                  onChange={(e) => setFormState(prev => ({ ...prev, admin_email: e.target.value }))}
                  placeholder="admin@example.com"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                />
              </div>

              <div className="flex items-center gap-6">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formState.notify_on_analysis}
                    onChange={(e) => setFormState(prev => ({ ...prev, notify_on_analysis: e.target.checked }))}
                    className="w-4 h-4 text-amber-600 rounded focus:ring-amber-500"
                  />
                  <span className="text-sm text-gray-700">Notify on analysis</span>
                </label>

                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formState.notify_on_switch}
                    onChange={(e) => setFormState(prev => ({ ...prev, notify_on_switch: e.target.checked }))}
                    className="w-4 h-4 text-amber-600 rounded focus:ring-amber-500"
                  />
                  <span className="text-sm text-gray-700">Notify on switch</span>
                </label>
              </div>
            </div>
          </div>

          {/* Error/Success Messages */}
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <span className="text-red-700">{error}</span>
            </div>
          )}

          {successMessage && (
            <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-lg flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-emerald-500" />
              <span className="text-emerald-700">{successMessage}</span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              onClick={saveConfig}
              disabled={saving}
              className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-amber-600 text-white rounded-xl font-medium hover:bg-amber-700 disabled:opacity-50 transition-colors"
            >
              {saving ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
              {saving ? 'Saving...' : 'Save Configuration'}
            </button>

            <button
              onClick={triggerAnalysis}
              className="px-6 py-3 border border-gray-300 text-gray-700 rounded-xl font-medium hover:bg-gray-50 transition-colors"
            >
              Trigger Analysis Now
            </button>
          </div>
        </div>
      </div>

      {/* Switch History */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 flex items-center gap-3">
          <History className="w-5 h-5 text-gray-500" />
          <h3 className="font-semibold text-gray-900">Switch History</h3>
        </div>

        {history.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            No strategy switches recorded yet
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {history.map((item) => (
              <div key={item.id} className="px-6 py-4 flex items-center gap-4">
                <div className="w-32">
                  <p className="text-sm text-gray-900">
                    {formatDate(item.switch_date)}
                  </p>
                  <p className="text-xs text-gray-500">
                    {new Date(item.switch_date).toLocaleTimeString()}
                  </p>
                </div>

                <div className="flex items-center gap-2 flex-1">
                  {item.from_strategy_name && (
                    <>
                      <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm">
                        {item.from_strategy_name}
                      </span>
                      <ArrowRight size={16} className="text-gray-400" />
                    </>
                  )}
                  <span className="px-2 py-1 bg-emerald-100 text-emerald-700 rounded text-sm font-medium">
                    {item.to_strategy_name}
                  </span>
                </div>

                <div className="text-right">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    item.trigger === 'auto_scheduled'
                      ? 'bg-blue-100 text-blue-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {item.trigger === 'auto_scheduled' ? 'Auto' : 'Manual'}
                  </span>
                </div>

                {item.score_before != null && item.score_after != null && (
                  <div className="w-24 text-right">
                    <span className="text-sm text-emerald-600 font-medium">
                      +{(item.score_after - item.score_before).toFixed(0)} pts
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
