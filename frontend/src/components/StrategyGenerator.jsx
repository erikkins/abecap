import React, { useState } from 'react';
import { Zap, RefreshCw, Check, TrendingUp, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function StrategyGenerator({ fetchWithAuth, onStrategyCreated }) {
  const [lookbackWeeks, setLookbackWeeks] = useState(12);
  const [strategyType, setStrategyType] = useState('momentum');
  const [optimizationMetric, setOptimizationMetric] = useState('sharpe');
  const [autoCreate, setAutoCreate] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const runGeneration = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const params = new URLSearchParams({
        lookback_weeks: lookbackWeeks,
        strategy_type: strategyType,
        optimization_metric: optimizationMetric,
        auto_create: autoCreate
      });

      const response = await fetchWithAuth(
        `${API_URL}/api/admin/strategies/generate?${params}`,
        { method: 'POST' }
      );

      if (response.ok) {
        const data = await response.json();
        setResult(data);
        if (autoCreate && onStrategyCreated) {
          onStrategyCreated();
        }
      } else {
        try {
          const err = await response.json();
          setError(err.detail || JSON.stringify(err));
        } catch {
          const text = await response.text();
          setError(`Server error (${response.status}): ${text.slice(0, 200)}`);
        }
      }
    } catch (err) {
      setError(`Request failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-purple-50 to-indigo-50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-100 rounded-lg">
            <Zap className="w-5 h-5 text-purple-600" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">AI Strategy Generator</h3>
            <p className="text-sm text-gray-600">Optimize parameters based on market conditions</p>
          </div>
        </div>
      </div>

      {/* Configuration */}
      <div className="p-6 space-y-6">
        {/* Main Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Lookback Period */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Lookback Period
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="4"
                max="52"
                value={lookbackWeeks}
                onChange={(e) => setLookbackWeeks(parseInt(e.target.value))}
                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
              />
              <span className="text-sm font-medium text-gray-900 w-20">
                {lookbackWeeks} weeks
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              ~{lookbackWeeks * 5} trading days
            </p>
          </div>

          {/* Strategy Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Strategy Type
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setStrategyType('momentum')}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  strategyType === 'momentum'
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Momentum
              </button>
              <button
                onClick={() => setStrategyType('dwap')}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  strategyType === 'dwap'
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                DWAP
              </button>
            </div>
          </div>

          {/* Optimization Metric */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Optimize For
            </label>
            <select
              value={optimizationMetric}
              onChange={(e) => setOptimizationMetric(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            >
              <option value="sharpe">Sharpe Ratio (Recommended)</option>
              <option value="return">Total Return</option>
              <option value="calmar">Calmar Ratio</option>
            </select>
          </div>
        </div>

        {/* Advanced Options */}
        <div>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
          >
            {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            Advanced Options
          </button>

          {showAdvanced && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={autoCreate}
                  onChange={(e) => setAutoCreate(e.target.checked)}
                  className="w-4 h-4 text-purple-600 rounded focus:ring-purple-500"
                />
                <span className="text-sm text-gray-700">
                  Automatically create strategy from results
                </span>
              </label>
              <p className="text-xs text-gray-500 mt-2 ml-7">
                When enabled, a new strategy will be added to your library with the optimal parameters found.
              </p>
            </div>
          )}
        </div>

        {/* Generate Button */}
        <button
          onClick={runGeneration}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-xl font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <>
              <RefreshCw className="w-5 h-5 animate-spin" />
              Optimizing Parameters...
            </>
          ) : (
            <>
              <Zap className="w-5 h-5" />
              Generate Optimal Strategy
            </>
          )}
        </button>

        {/* Error Display */}
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
            <div>
              <p className="font-medium text-red-800">Generation Failed</p>
              <p className="text-sm text-red-600">{error}</p>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="border border-gray-200 rounded-xl overflow-hidden">
            {/* Results Header */}
            <div className="px-4 py-3 bg-gradient-to-r from-emerald-50 to-green-50 border-b border-gray-200">
              <div className="flex items-center gap-2">
                <Check className="w-5 h-5 text-emerald-600" />
                <span className="font-semibold text-gray-900">Optimization Complete</span>
                <span className="ml-auto text-sm text-gray-500">
                  {result.combinations_tested} combinations tested
                </span>
              </div>
            </div>

            {/* Market Regime */}
            <div className="px-4 py-3 border-b border-gray-200">
              <span className="text-sm text-gray-500">Market Regime Detected:</span>
              <span className={`ml-2 px-2 py-1 rounded-full text-xs font-medium ${
                result.market_regime === 'bull' ? 'bg-emerald-100 text-emerald-800' :
                result.market_regime === 'bear' ? 'bg-red-100 text-red-800' :
                'bg-gray-100 text-gray-800'
              }`}>
                {result.market_regime?.toUpperCase()}
              </span>
            </div>

            {/* Expected Metrics */}
            <div className="p-4 grid grid-cols-3 gap-4 border-b border-gray-200">
              <div className="text-center">
                <p className="text-sm text-gray-500">Expected Sharpe</p>
                <p className="text-2xl font-bold text-emerald-600">
                  {result.expected_sharpe?.toFixed(2)}
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-500">Expected Return</p>
                <p className="text-2xl font-bold text-emerald-600">
                  +{result.expected_return_pct?.toFixed(1)}%
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-500">Max Drawdown</p>
                <p className="text-2xl font-bold text-red-500">
                  -{result.expected_drawdown_pct?.toFixed(1)}%
                </p>
              </div>
            </div>

            {/* Best Parameters */}
            <div className="p-4">
              <h4 className="font-medium text-gray-900 mb-3">Optimal Parameters</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {Object.entries(result.best_params || {}).map(([key, value]) => (
                  <div key={key} className="px-3 py-2 bg-gray-50 rounded-lg">
                    <p className="text-xs text-gray-500">
                      {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </p>
                    <p className="font-semibold text-gray-900">{value}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Top 5 Results */}
            {result.top_5_results && result.top_5_results.length > 1 && (
              <div className="p-4 border-t border-gray-200">
                <h4 className="font-medium text-gray-900 mb-3">Top 5 Combinations</h4>
                <div className="space-y-2">
                  {result.top_5_results.map((r, i) => (
                    <div key={i} className="flex items-center justify-between px-3 py-2 bg-gray-50 rounded-lg">
                      <span className="text-sm">
                        {i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i + 1}`}
                      </span>
                      <span className="text-sm font-medium">
                        Sharpe: {r.sharpe_ratio?.toFixed(2)}
                      </span>
                      <span className={`text-sm ${r.total_return_pct >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                        {r.total_return_pct >= 0 ? '+' : ''}{r.total_return_pct?.toFixed(1)}%
                      </span>
                      <span className="text-sm text-gray-500">
                        {r.total_trades} trades
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Create Strategy Button */}
            {!autoCreate && (
              <div className="p-4 border-t border-gray-200 bg-gray-50">
                <button
                  onClick={() => {
                    // TODO: Implement create strategy from results
                    alert('Strategy creation from results coming soon. Use auto-create option for now.');
                  }}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 transition-colors"
                >
                  <TrendingUp size={18} />
                  Create Strategy from Results
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
