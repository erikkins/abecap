import React, { useState, useEffect } from 'react';
import { Plus, Save, Trash2, Copy, Play, RefreshCw, AlertCircle, X } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Default parameter templates
const PARAMETER_TEMPLATES = {
  momentum: {
    max_positions: 5,
    position_size_pct: 18,
    short_momentum_days: 10,
    long_momentum_days: 60,
    trailing_stop_pct: 15,
    market_filter_enabled: true,
    rebalance_frequency: 'weekly',
    short_mom_weight: 0.5,
    long_mom_weight: 0.3,
    volatility_penalty: 0.2,
    near_50d_high_pct: 5,
    min_volume: 500000,
    min_price: 20
  },
  dwap: {
    max_positions: 15,
    position_size_pct: 6.6,
    dwap_threshold_pct: 5,
    stop_loss_pct: 8,
    profit_target_pct: 20,
    volume_spike_mult: 1.5,
    min_volume: 500000,
    min_price: 20
  }
};

export default function StrategyEditor({ fetchWithAuth, strategies, onStrategyChange }) {
  const [mode, setMode] = useState('list'); // 'list', 'create', 'edit'
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [testResult, setTestResult] = useState(null);

  // Form state
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    strategy_type: 'momentum',
    parameters: {}
  });

  useEffect(() => {
    if (mode === 'create') {
      setFormData({
        name: '',
        description: '',
        strategy_type: 'momentum',
        parameters: { ...PARAMETER_TEMPLATES.momentum }
      });
    }
  }, [mode]);

  const handleTypeChange = (type) => {
    setFormData(prev => ({
      ...prev,
      strategy_type: type,
      parameters: { ...PARAMETER_TEMPLATES[type] }
    }));
  };

  const handleParamChange = (key, value) => {
    setFormData(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [key]: value
      }
    }));
  };

  const handleEdit = (strategy) => {
    setSelectedStrategy(strategy);
    setFormData({
      name: strategy.name,
      description: strategy.description || '',
      strategy_type: strategy.strategy_type,
      parameters: { ...strategy.parameters }
    });
    setMode('edit');
  };

  const handleClone = (strategy) => {
    setSelectedStrategy(null);
    setFormData({
      name: `${strategy.name} (Copy)`,
      description: strategy.description || '',
      strategy_type: strategy.strategy_type,
      parameters: { ...strategy.parameters }
    });
    setMode('create');
  };

  const handleSave = async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        name: formData.name,
        strategy_type: formData.strategy_type,
        parameters: JSON.stringify(formData.parameters),
        description: formData.description
      });

      let response;
      if (mode === 'edit' && selectedStrategy) {
        response = await fetchWithAuth(
          `${API_URL}/api/admin/strategies/${selectedStrategy.id}?${params}`,
          { method: 'PUT' }
        );
      } else {
        response = await fetchWithAuth(
          `${API_URL}/api/admin/strategies?${params}`,
          { method: 'POST' }
        );
      }

      if (response.ok) {
        setMode('list');
        onStrategyChange?.();
      } else {
        const err = await response.json();
        setError(err.detail || 'Failed to save strategy');
      }
    } catch (err) {
      setError(err.message || 'Failed to save strategy');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (strategy) => {
    if (!confirm(`Are you sure you want to delete "${strategy.name}"?`)) return;

    try {
      const response = await fetchWithAuth(
        `${API_URL}/api/admin/strategies/${strategy.id}`,
        { method: 'DELETE' }
      );

      if (response.ok) {
        onStrategyChange?.();
      } else {
        const err = await response.json();
        alert(err.detail || 'Failed to delete strategy');
      }
    } catch (err) {
      alert(err.message || 'Failed to delete strategy');
    }
  };

  const runQuickTest = async () => {
    setLoading(true);
    setTestResult(null);

    try {
      const params = new URLSearchParams({
        strategy_type: formData.strategy_type,
        custom_params: JSON.stringify(formData.parameters),
        lookback_days: 90
      });

      const response = await fetchWithAuth(
        `${API_URL}/api/admin/strategies/backtest?${params}`,
        { method: 'POST' }
      );

      if (response.ok) {
        const data = await response.json();
        setTestResult(data.metrics);
      } else {
        const err = await response.json();
        setError(err.detail || 'Test failed');
      }
    } catch (err) {
      setError(err.message || 'Test failed');
    } finally {
      setLoading(false);
    }
  };

  // Custom strategies (editable)
  const customStrategies = strategies?.filter(s => s.is_custom || s.source === 'ai_generated') || [];

  // Render parameter input based on key and value type
  const renderParamInput = (key, value) => {
    const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

    if (typeof value === 'boolean') {
      return (
        <label key={key} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
          <input
            type="checkbox"
            checked={formData.parameters[key] ?? value}
            onChange={(e) => handleParamChange(key, e.target.checked)}
            className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
          />
          <span className="text-sm text-gray-700">{label}</span>
        </label>
      );
    }

    if (key === 'rebalance_frequency') {
      return (
        <div key={key} className="p-3 bg-gray-50 rounded-lg">
          <label className="block text-xs text-gray-500 mb-1">{label}</label>
          <select
            value={formData.parameters[key] ?? value}
            onChange={(e) => handleParamChange(key, e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm"
          >
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
        </div>
      );
    }

    return (
      <div key={key} className="p-3 bg-gray-50 rounded-lg">
        <label className="block text-xs text-gray-500 mb-1">{label}</label>
        <input
          type="number"
          step={key.includes('pct') || key.includes('weight') ? '0.1' : '1'}
          value={formData.parameters[key] ?? value}
          onChange={(e) => handleParamChange(key, parseFloat(e.target.value))}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm"
        />
      </div>
    );
  };

  // List View
  if (mode === 'list') {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <h3 className="font-semibold text-gray-900">Custom Strategies</h3>
          <button
            onClick={() => setMode('create')}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
          >
            <Plus size={16} />
            New Strategy
          </button>
        </div>

        {customStrategies.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            <p className="mb-2">No custom strategies yet</p>
            <p className="text-sm">Create a new strategy or use the AI generator to optimize parameters</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {customStrategies.map((strategy) => (
              <div key={strategy.id} className="px-6 py-4 flex items-center gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-900">{strategy.name}</span>
                    {strategy.source === 'ai_generated' && (
                      <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs rounded">AI</span>
                    )}
                    {strategy.is_active && (
                      <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 text-xs rounded">Active</span>
                    )}
                  </div>
                  <p className="text-sm text-gray-500 mt-1">{strategy.description}</p>
                </div>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  strategy.strategy_type === 'momentum' ? 'bg-purple-100 text-purple-800' : 'bg-blue-100 text-blue-800'
                }`}>
                  {strategy.strategy_type}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleClone(strategy)}
                    className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
                    title="Clone"
                  >
                    <Copy size={16} />
                  </button>
                  <button
                    onClick={() => handleEdit(strategy)}
                    className="p-2 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded-lg"
                    title="Edit"
                  >
                    <Save size={16} />
                  </button>
                  {!strategy.is_active && (
                    <button
                      onClick={() => handleDelete(strategy)}
                      className="p-2 text-red-600 hover:text-red-800 hover:bg-red-50 rounded-lg"
                      title="Delete"
                    >
                      <Trash2 size={16} />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Create/Edit View
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
        <h3 className="font-semibold text-gray-900">
          {mode === 'create' ? 'Create Strategy' : 'Edit Strategy'}
        </h3>
        <button
          onClick={() => setMode('list')}
          className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg"
        >
          <X size={20} />
        </button>
      </div>

      <div className="p-6 space-y-6">
        {/* Basic Info */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Strategy Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
              placeholder="My Custom Strategy"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Strategy Type</label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => handleTypeChange('momentum')}
                disabled={mode === 'edit'}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  formData.strategy_type === 'momentum'
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                } ${mode === 'edit' ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                Momentum
              </button>
              <button
                type="button"
                onClick={() => handleTypeChange('dwap')}
                disabled={mode === 'edit'}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  formData.strategy_type === 'dwap'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                } ${mode === 'edit' ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                DWAP
              </button>
            </div>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Description</label>
          <textarea
            value={formData.description}
            onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
            placeholder="Describe your strategy..."
            rows={2}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        {/* Parameters */}
        <div>
          <h4 className="font-medium text-gray-900 mb-4">Parameters</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {Object.entries(PARAMETER_TEMPLATES[formData.strategy_type]).map(([key, defaultValue]) =>
              renderParamInput(key, defaultValue)
            )}
          </div>
        </div>

        {/* Quick Test */}
        <div className="border-t border-gray-200 pt-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-medium text-gray-900">Quick Test (90-day backtest)</h4>
            <button
              onClick={runQuickTest}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200 disabled:opacity-50 transition-colors"
            >
              {loading ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} />}
              Run Test
            </button>
          </div>

          {testResult && (
            <div className="grid grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
              <div className="text-center">
                <p className="text-xs text-gray-500">Sharpe</p>
                <p className="text-lg font-bold text-gray-900">{testResult.sharpe_ratio?.toFixed(2)}</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500">Return</p>
                <p className={`text-lg font-bold ${testResult.total_return_pct >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                  {testResult.total_return_pct >= 0 ? '+' : ''}{testResult.total_return_pct?.toFixed(1)}%
                </p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500">Max DD</p>
                <p className="text-lg font-bold text-red-600">-{testResult.max_drawdown_pct?.toFixed(1)}%</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500">Win Rate</p>
                <p className="text-lg font-bold text-gray-900">{testResult.win_rate?.toFixed(0)}%</p>
              </div>
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <span className="text-red-700">{error}</span>
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-4">
          <button
            onClick={handleSave}
            disabled={loading || !formData.name}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Save className="w-5 h-5" />}
            {mode === 'create' ? 'Create Strategy' : 'Save Changes'}
          </button>
          <button
            onClick={() => setMode('list')}
            className="px-6 py-3 border border-gray-300 text-gray-700 rounded-xl font-medium hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
