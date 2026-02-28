import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, BarChart, Bar, Cell } from 'recharts';
import { PlayCircle, RefreshCw, TrendingUp, Calendar, List, Settings, AlertCircle, GitCompare, Trophy } from 'lucide-react';

import { formatDate, formatChartDate } from '../utils/formatDate';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const PRESET_UNIVERSES = {
  'nasdaq100': 'NASDAQ-100',
  'sp500': 'S&P 500',
  'custom': 'Custom Tickers'
};

const EXIT_STRATEGIES = {
  trailing_stop: { label: 'Trailing Stop', description: 'Exit when price drops X% from high water mark' },
  fixed_target: { label: 'Fixed Target', description: 'Exit at X% profit target' },
  hybrid: { label: 'Hybrid', description: 'Hit initial target, then switch to trailing stop' },
  time_based: { label: 'Time-Based', description: 'Exit after max holding period' },
  stop_loss_target: { label: 'Stop + Target', description: 'Fixed stop loss and profit target (legacy)' }
};

export default function FlexibleBacktest({ fetchWithAuth, strategies = [] }) {
  const [startDate, setStartDate] = useState(() => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 1);
    return d.toISOString().split('T')[0];
  });
  const [endDate, setEndDate] = useState(() => new Date().toISOString().split('T')[0]);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [useCustomParams, setUseCustomParams] = useState(false);
  const [strategyType, setStrategyType] = useState('momentum');
  const [customParams, setCustomParams] = useState({
    short_momentum_days: 10,
    long_momentum_days: 60,
    trailing_stop_pct: 15,
    max_positions: 5,
    position_size_pct: 18
  });
  const [tickerUniverse, setTickerUniverse] = useState('nasdaq100');
  const [customTickers, setCustomTickers] = useState('AAPL, MSFT, GOOGL, AMZN, NVDA');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Exit strategy state
  const [exitStrategy, setExitStrategy] = useState('trailing_stop');
  const [exitParams, setExitParams] = useState({
    trailing_stop_pct: 15,
    profit_target_pct: 20,
    stop_loss_pct: 8,
    hybrid_initial_target_pct: 15,
    hybrid_trailing_pct: 8,
    max_hold_days: 60
  });

  // Comparison mode
  const [compareMode, setCompareMode] = useState(false);
  const [comparisonResult, setComparisonResult] = useState(null);

  const PARAM_CONFIG = {
    momentum: {
      short_momentum_days: { label: 'Short Momentum Days', min: 5, max: 30, step: 5 },
      long_momentum_days: { label: 'Long Momentum Days', min: 20, max: 120, step: 10 },
      trailing_stop_pct: { label: 'Trailing Stop %', min: 5, max: 25, step: 1 },
      max_positions: { label: 'Max Positions', min: 3, max: 10, step: 1 },
      position_size_pct: { label: 'Position Size %', min: 10, max: 25, step: 1 }
    },
    dwap: {
      dwap_threshold_pct: { label: 'DWAP Threshold %', min: 1, max: 10, step: 1 },
      stop_loss_pct: { label: 'Stop Loss %', min: 5, max: 15, step: 1 },
      profit_target_pct: { label: 'Profit Target %', min: 10, max: 40, step: 5 },
      max_positions: { label: 'Max Positions', min: 5, max: 25, step: 5 }
    }
  };

  useEffect(() => {
    // Reset custom params when strategy type changes
    if (strategyType === 'momentum') {
      setCustomParams({
        short_momentum_days: 10,
        long_momentum_days: 60,
        trailing_stop_pct: 15,
        max_positions: 5,
        position_size_pct: 18
      });
    } else {
      setCustomParams({
        dwap_threshold_pct: 5,
        stop_loss_pct: 8,
        profit_target_pct: 20,
        max_positions: 15
      });
    }
  }, [strategyType]);

  const runBacktest = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setComparisonResult(null);

    try {
      const body = {
        start_date: startDate,
        end_date: endDate,
        strategy_type: strategyType
      };

      if (useCustomParams) {
        body.custom_params = customParams;
      } else if (selectedStrategy) {
        body.strategy_id = selectedStrategy;
      }

      if (tickerUniverse === 'custom') {
        body.ticker_list = customTickers
          .split(/[,\s]+/)
          .map(t => t.trim().toUpperCase())
          .filter(t => t.length > 0);
      } else {
        body.ticker_universe = tickerUniverse;
      }

      // Add exit strategy parameters
      body.exit_strategy_type = exitStrategy;
      body.trailing_stop_pct = exitParams.trailing_stop_pct;
      body.profit_target_pct = exitParams.profit_target_pct;
      body.stop_loss_pct = exitParams.stop_loss_pct;
      body.hybrid_initial_target_pct = exitParams.hybrid_initial_target_pct;
      body.hybrid_trailing_pct = exitParams.hybrid_trailing_pct;
      body.max_hold_days = exitParams.max_hold_days;

      const response = await fetchWithAuth(
        `${API_URL}/api/admin/strategies/backtest`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        }
      );

      if (response.ok) {
        const data = await response.json();
        setResult(data);
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

  const runExitComparison = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setComparisonResult(null);

    try {
      const body = {
        start_date: startDate,
        end_date: endDate,
        lookback_days: Math.ceil((new Date(endDate) - new Date(startDate)) / (1000 * 60 * 60 * 24))
      };

      if (tickerUniverse === 'custom') {
        body.ticker_list = customTickers
          .split(/[,\s]+/)
          .map(t => t.trim().toUpperCase())
          .filter(t => t.length > 0);
      }

      const response = await fetchWithAuth(
        `${API_URL}/api/admin/strategies/backtest/compare-exits`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        }
      );

      if (response.ok) {
        const data = await response.json();
        setComparisonResult(data);
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

  // Prepare chart data
  const chartData = result?.equity_curve?.map(point => ({
    date: point.date,
    portfolio: point.equity,
    benchmark: point.benchmark_equity
  })) || [];

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-emerald-50 to-teal-50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-100 rounded-lg">
            <TrendingUp className="w-5 h-5 text-emerald-600" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">Flexible Backtest</h3>
            <p className="text-sm text-gray-600">Test strategies with custom date ranges and ticker lists</p>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Date Range */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar size={14} className="inline mr-1" />
              Start Date
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Calendar size={14} className="inline mr-1" />
              End Date
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
            />
          </div>
        </div>

        {/* Ticker Universe */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <List size={14} className="inline mr-1" />
            Ticker Universe
          </label>
          <div className="flex gap-2 mb-3">
            {Object.entries(PRESET_UNIVERSES).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setTickerUniverse(key)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  tickerUniverse === key
                    ? 'bg-emerald-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {tickerUniverse === 'custom' && (
            <textarea
              value={customTickers}
              onChange={(e) => setCustomTickers(e.target.value)}
              placeholder="Enter tickers separated by commas or spaces (e.g., AAPL, MSFT, GOOGL)"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 h-24"
            />
          )}
        </div>

        {/* Strategy Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <Settings size={14} className="inline mr-1" />
            Strategy
          </label>
          <div className="flex gap-2 mb-3">
            <button
              onClick={() => setUseCustomParams(false)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                !useCustomParams
                  ? 'bg-emerald-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Use Library Strategy
            </button>
            <button
              onClick={() => setUseCustomParams(true)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                useCustomParams
                  ? 'bg-emerald-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Custom Parameters
            </button>
          </div>

          {!useCustomParams ? (
            <select
              value={selectedStrategy || ''}
              onChange={(e) => setSelectedStrategy(e.target.value ? parseInt(e.target.value) : null)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
            >
              <option value="">Select a strategy...</option>
              {strategies.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.strategy_type})
                </option>
              ))}
            </select>
          ) : (
            <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
              {/* Strategy Type Toggle */}
              <div className="flex gap-2 mb-4">
                <button
                  onClick={() => setStrategyType('momentum')}
                  className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    strategyType === 'momentum'
                      ? 'bg-emerald-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Momentum
                </button>
                <button
                  onClick={() => setStrategyType('dwap')}
                  className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    strategyType === 'dwap'
                      ? 'bg-emerald-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  DWAP
                </button>
              </div>

              {/* Parameter Sliders */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(PARAM_CONFIG[strategyType]).map(([key, config]) => (
                  <div key={key}>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      {config.label}
                    </label>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min={config.min}
                        max={config.max}
                        step={config.step}
                        value={customParams[key] || config.min}
                        onChange={(e) => setCustomParams(prev => ({
                          ...prev,
                          [key]: parseFloat(e.target.value)
                        }))}
                        className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                      />
                      <span className="text-sm font-medium text-gray-900 w-12">
                        {customParams[key] || config.min}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Exit Strategy Selection */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-medium text-gray-700">
              <Settings size={14} className="inline mr-1" />
              Exit Strategy
            </label>
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={`flex items-center gap-1 px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                compareMode
                  ? 'bg-purple-600 text-white'
                  : 'bg-purple-100 text-purple-700 hover:bg-purple-200'
              }`}
            >
              <GitCompare size={12} />
              {compareMode ? 'Compare Mode ON' : 'Compare All'}
            </button>
          </div>

          {!compareMode ? (
            <>
              <div className="flex flex-wrap gap-2 mb-3">
                {Object.entries(EXIT_STRATEGIES).map(([key, { label }]) => (
                  <button
                    key={key}
                    onClick={() => setExitStrategy(key)}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                      exitStrategy === key
                        ? 'bg-emerald-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <p className="text-xs text-gray-500 mb-3">{EXIT_STRATEGIES[exitStrategy]?.description}</p>

              {/* Exit Strategy Parameters */}
              <div className="p-4 bg-gray-50 rounded-lg space-y-3">
                {(exitStrategy === 'trailing_stop' || exitStrategy === 'hybrid') && (
                  <div className="flex items-center gap-3">
                    <label className="text-sm text-gray-700 w-40">
                      {exitStrategy === 'hybrid' ? 'Trailing %' : 'Trailing Stop %'}
                    </label>
                    <input
                      type="range"
                      min="5"
                      max="25"
                      step="1"
                      value={exitStrategy === 'hybrid' ? exitParams.hybrid_trailing_pct : exitParams.trailing_stop_pct}
                      onChange={(e) => setExitParams(prev => ({
                        ...prev,
                        [exitStrategy === 'hybrid' ? 'hybrid_trailing_pct' : 'trailing_stop_pct']: parseInt(e.target.value)
                      }))}
                      className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                    />
                    <span className="text-sm font-medium w-10 text-right">
                      {exitStrategy === 'hybrid' ? exitParams.hybrid_trailing_pct : exitParams.trailing_stop_pct}%
                    </span>
                  </div>
                )}

                {(exitStrategy === 'fixed_target' || exitStrategy === 'stop_loss_target' || exitStrategy === 'hybrid') && (
                  <div className="flex items-center gap-3">
                    <label className="text-sm text-gray-700 w-40">
                      {exitStrategy === 'hybrid' ? 'Initial Target %' : 'Profit Target %'}
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="50"
                      step="5"
                      value={exitStrategy === 'hybrid' ? exitParams.hybrid_initial_target_pct : exitParams.profit_target_pct}
                      onChange={(e) => setExitParams(prev => ({
                        ...prev,
                        [exitStrategy === 'hybrid' ? 'hybrid_initial_target_pct' : 'profit_target_pct']: parseInt(e.target.value)
                      }))}
                      className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                    />
                    <span className="text-sm font-medium w-10 text-right">
                      {exitStrategy === 'hybrid' ? exitParams.hybrid_initial_target_pct : exitParams.profit_target_pct}%
                    </span>
                  </div>
                )}

                {exitStrategy === 'stop_loss_target' && (
                  <div className="flex items-center gap-3">
                    <label className="text-sm text-gray-700 w-40">Stop Loss %</label>
                    <input
                      type="range"
                      min="5"
                      max="15"
                      step="1"
                      value={exitParams.stop_loss_pct}
                      onChange={(e) => setExitParams(prev => ({ ...prev, stop_loss_pct: parseInt(e.target.value) }))}
                      className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-red-600"
                    />
                    <span className="text-sm font-medium w-10 text-right">{exitParams.stop_loss_pct}%</span>
                  </div>
                )}

                {exitStrategy === 'time_based' && (
                  <div className="flex items-center gap-3">
                    <label className="text-sm text-gray-700 w-40">Max Hold Days</label>
                    <input
                      type="range"
                      min="20"
                      max="120"
                      step="10"
                      value={exitParams.max_hold_days}
                      onChange={(e) => setExitParams(prev => ({ ...prev, max_hold_days: parseInt(e.target.value) }))}
                      className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                    />
                    <span className="text-sm font-medium w-10 text-right">{exitParams.max_hold_days}d</span>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
              <p className="text-sm text-purple-800">
                <strong>Compare Mode:</strong> Test all exit strategies head-to-head to find the best one for your selected date range.
              </p>
              <p className="text-xs text-purple-600 mt-2">
                Strategies compared: Trailing 10/15/20%, Fixed Target 20/30%, Hybrid, Stop+Target, Time-Based
              </p>
            </div>
          )}
        </div>

        {/* Run Button */}
        <button
          onClick={compareMode ? runExitComparison : runBacktest}
          disabled={loading || (!compareMode && !useCustomParams && !selectedStrategy)}
          className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-emerald-600 text-white rounded-xl font-medium hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <>
              <RefreshCw className="w-5 h-5 animate-spin" />
              {compareMode ? 'Comparing Exit Strategies...' : 'Running Backtest...'}
            </>
          ) : (
            <>
              {compareMode ? <GitCompare className="w-5 h-5" /> : <PlayCircle className="w-5 h-5" />}
              {compareMode ? 'Compare All Exit Strategies' : 'Run Backtest'}
            </>
          )}
        </button>

        {/* Error Display */}
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
            <div>
              <p className="font-medium text-red-800">Backtest Failed</p>
              <p className="text-sm text-red-600">{error}</p>
            </div>
          </div>
        )}

        {/* Comparison Results */}
        {comparisonResult && (
          <div className="space-y-6">
            {/* Winner Banner */}
            {comparisonResult.best_strategy && (
              <div className="p-4 bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-200 rounded-xl flex items-center gap-4">
                <div className="p-3 bg-yellow-100 rounded-full">
                  <Trophy className="w-6 h-6 text-yellow-600" />
                </div>
                <div>
                  <p className="text-sm text-yellow-800">Best Exit Strategy</p>
                  <p className="text-xl font-bold text-yellow-900">{comparisonResult.best_strategy}</p>
                </div>
              </div>
            )}

            {/* Ranking Table */}
            <div className="border border-gray-200 rounded-xl overflow-hidden">
              <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
                <h4 className="font-medium text-gray-900">Exit Strategy Comparison</h4>
                <p className="text-xs text-gray-500 mt-1">
                  {comparisonResult.start_date} to {comparisonResult.end_date} • {comparisonResult.tickers_used} tickers
                </p>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rank</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Strategy</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Return</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Sharpe</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Drawdown</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Win Rate</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {comparisonResult.ranking?.map((r, i) => (
                      <tr key={i} className={i === 0 ? 'bg-yellow-50' : 'hover:bg-gray-50'}>
                        <td className="px-4 py-3 text-sm">
                          {i === 0 ? (
                            <span className="inline-flex items-center justify-center w-6 h-6 bg-yellow-400 text-yellow-900 rounded-full font-bold text-xs">1</span>
                          ) : (
                            <span className="text-gray-500">{r.rank}</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-sm font-medium text-gray-900">{r.name}</td>
                        <td className={`px-4 py-3 text-sm font-medium text-right ${r.return >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                          {r.return >= 0 ? '+' : ''}{r.return?.toFixed(1)}%
                        </td>
                        <td className="px-4 py-3 text-sm text-right text-gray-900">{r.sharpe?.toFixed(2)}</td>
                        <td className="px-4 py-3 text-sm text-right text-red-600">-{r.drawdown?.toFixed(1)}%</td>
                        <td className="px-4 py-3 text-sm text-right text-gray-900">{r.win_rate?.toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Visual Comparison Chart */}
            {comparisonResult.ranking?.length > 0 && (
              <div className="p-4 border border-gray-200 rounded-xl">
                <h4 className="font-medium text-gray-900 mb-4">Return vs Sharpe Ratio</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={comparisonResult.ranking} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis type="number" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={120} />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const data = payload[0]?.payload;
                        return (
                          <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200 text-sm">
                            <p className="font-medium">{data?.name}</p>
                            <p className={data?.return >= 0 ? 'text-emerald-600' : 'text-red-600'}>
                              Return: {data?.return >= 0 ? '+' : ''}{data?.return?.toFixed(1)}%
                            </p>
                            <p className="text-blue-600">Sharpe: {data?.sharpe?.toFixed(2)}</p>
                            <p className="text-red-600">Max DD: -{data?.drawdown?.toFixed(1)}%</p>
                          </div>
                        );
                      }}
                    />
                    <Bar dataKey="return" name="Return %">
                      {comparisonResult.ranking?.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={index === 0 ? '#F59E0B' : entry.return >= 0 ? '#10B981' : '#EF4444'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Detailed Metrics */}
            <div className="border border-gray-200 rounded-xl overflow-hidden">
              <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
                <h4 className="font-medium text-gray-900">Detailed Metrics</h4>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 text-xs">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase">Strategy</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase">Trades</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase">Avg Win</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase">Avg Loss</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase">Profit Factor</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase">Avg Days</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase">Sortino</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase">Calmar</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {comparisonResult.results?.filter(r => r.metrics).map((r, i) => (
                      <tr key={i} className="hover:bg-gray-50">
                        <td className="px-3 py-2 font-medium text-gray-900">{r.name}</td>
                        <td className="px-3 py-2 text-right text-gray-600">{r.metrics?.total_trades}</td>
                        <td className="px-3 py-2 text-right text-emerald-600">+{r.metrics?.avg_win_pct?.toFixed(1)}%</td>
                        <td className="px-3 py-2 text-right text-red-600">-{r.metrics?.avg_loss_pct?.toFixed(1)}%</td>
                        <td className="px-3 py-2 text-right text-gray-900">{r.metrics?.profit_factor?.toFixed(2)}</td>
                        <td className="px-3 py-2 text-right text-gray-600">{r.metrics?.avg_days_held?.toFixed(0)}</td>
                        <td className="px-3 py-2 text-right text-gray-900">{r.metrics?.sortino_ratio?.toFixed(2)}</td>
                        <td className="px-3 py-2 text-right text-gray-900">{r.metrics?.calmar_ratio?.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            {/* Performance Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-gradient-to-br from-emerald-50 to-green-50 rounded-xl text-center">
                <p className="text-sm text-gray-600 mb-1">Total Return</p>
                <p className={`text-2xl font-bold ${result.total_return_pct >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                  {result.total_return_pct >= 0 ? '+' : ''}{result.total_return_pct?.toFixed(1)}%
                </p>
              </div>
              <div className="p-4 bg-gray-50 rounded-xl text-center">
                <p className="text-sm text-gray-600 mb-1">Sharpe Ratio</p>
                <p className="text-2xl font-bold text-gray-900">{result.sharpe_ratio?.toFixed(2)}</p>
              </div>
              <div className="p-4 bg-red-50 rounded-xl text-center">
                <p className="text-sm text-gray-600 mb-1">Max Drawdown</p>
                <p className="text-2xl font-bold text-red-600">-{result.max_drawdown_pct?.toFixed(1)}%</p>
              </div>
              <div className="p-4 bg-blue-50 rounded-xl text-center">
                <p className="text-sm text-gray-600 mb-1">Total Trades</p>
                <p className="text-2xl font-bold text-blue-600">{result.total_trades}</p>
              </div>
            </div>

            {/* Win Rate & Avg Trade */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 bg-gray-50 rounded-lg text-center">
                <p className="text-xs text-gray-500">Win Rate</p>
                <p className="text-lg font-semibold text-gray-900">{result.win_rate_pct?.toFixed(0)}%</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg text-center">
                <p className="text-xs text-gray-500">Avg Win</p>
                <p className="text-lg font-semibold text-emerald-600">+{result.avg_win_pct?.toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg text-center">
                <p className="text-xs text-gray-500">Avg Loss</p>
                <p className="text-lg font-semibold text-red-600">{result.avg_loss_pct?.toFixed(1)}%</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg text-center">
                <p className="text-xs text-gray-500">Profit Factor</p>
                <p className="text-lg font-semibold text-gray-900">{result.profit_factor?.toFixed(2)}</p>
              </div>
            </div>

            {/* Benchmark Comparison */}
            {result.benchmark_return_pct != null && (
              <div className={`p-4 rounded-xl ${
                result.total_return_pct > result.benchmark_return_pct
                  ? 'bg-emerald-50 border border-emerald-200'
                  : 'bg-red-50 border border-red-200'
              }`}>
                <p className="text-center">
                  <span className="font-medium">
                    {result.total_return_pct > result.benchmark_return_pct ? 'Outperformed' : 'Underperformed'}
                  </span>
                  <span className="text-gray-600"> SPY by </span>
                  <span className={`font-bold ${
                    result.total_return_pct > result.benchmark_return_pct ? 'text-emerald-700' : 'text-red-700'
                  }`}>
                    {Math.abs(result.total_return_pct - result.benchmark_return_pct).toFixed(1)}%
                  </span>
                </p>
              </div>
            )}

            {/* Equity Curve Chart */}
            {chartData.length > 0 && (
              <div className="p-4 border border-gray-200 rounded-xl">
                <h4 className="font-medium text-gray-900 mb-4">Equity Curve</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 11 }}
                      tickFormatter={(val) => formatChartDate(val)}
                    />
                    <YAxis
                      tick={{ fontSize: 11 }}
                      tickFormatter={(val) => `$${(val / 1000).toFixed(0)}k`}
                      domain={['auto', 'auto']}
                    />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        const data = payload[0]?.payload;
                        return (
                          <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200 text-sm">
                            <p className="font-medium">{formatDate(label)}</p>
                            <p className="text-emerald-600">Portfolio: ${data?.portfolio?.toLocaleString()}</p>
                            {data?.benchmark && (
                              <p className="text-gray-500">SPY: ${data?.benchmark?.toLocaleString()}</p>
                            )}
                          </div>
                        );
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="portfolio"
                      stroke="#10B981"
                      strokeWidth={2}
                      dot={false}
                      name="Portfolio"
                    />
                    {chartData[0]?.benchmark && (
                      <Line
                        type="monotone"
                        dataKey="benchmark"
                        stroke="#9CA3AF"
                        strokeWidth={1}
                        strokeDasharray="5 5"
                        dot={false}
                        name="SPY Benchmark"
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Recent Trades */}
            {result.trades && result.trades.length > 0 && (
              <div className="border border-gray-200 rounded-xl overflow-hidden">
                <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
                  <h4 className="font-medium text-gray-900">Recent Trades (Last 10)</h4>
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Ticker</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                        <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Entry</th>
                        <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Exit</th>
                        <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Return</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {result.trades.slice(-10).reverse().map((trade, i) => (
                        <tr key={i} className="hover:bg-gray-50">
                          <td className="px-4 py-2 text-sm text-gray-900">
                            {formatDate(trade.exit_date || trade.entry_date)}
                          </td>
                          <td className="px-4 py-2 text-sm font-medium text-gray-900">{trade.ticker}</td>
                          <td className="px-4 py-2 text-sm text-gray-500">{trade.exit_reason || 'Open'}</td>
                          <td className="px-4 py-2 text-sm text-gray-900 text-right">${trade.entry_price?.toFixed(2)}</td>
                          <td className="px-4 py-2 text-sm text-gray-900 text-right">
                            {trade.exit_price ? `$${trade.exit_price.toFixed(2)}` : '-'}
                          </td>
                          <td className={`px-4 py-2 text-sm font-medium text-right ${
                            trade.return_pct >= 0 ? 'text-emerald-600' : 'text-red-600'
                          }`}>
                            {trade.return_pct >= 0 ? '+' : ''}{trade.return_pct?.toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
