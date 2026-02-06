import React, { useState, useEffect } from 'react';
import { Zap, RefreshCw, AlertCircle, TrendingUp, ArrowUpRight, Sparkles } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const DoubleSignals = ({ onSymbolClick }) => {
  const [signals, setSignals] = useState([]);
  const [freshSignals, setFreshSignals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [stats, setStats] = useState({ dwapOnly: 0, momentumOnly: 0, freshCount: 0, staleCount: 0 });
  const [marketFilterActive, setMarketFilterActive] = useState(false);

  const fetchSignals = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/signals/double-signals?momentum_top_n=20&fresh_days=5`);
      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }
      const data = await res.json();
      setSignals(data.signals || []);
      setFreshSignals(data.fresh_signals || []);
      setStats({
        dwapOnly: data.dwap_only_count || 0,
        momentumOnly: data.momentum_only_count || 0,
        freshCount: data.fresh_count || 0,
        staleCount: data.stale_count || 0
      });
      setMarketFilterActive(data.market_filter_active);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      console.error('Failed to fetch double signals:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSignals();
    // Refresh every 5 minutes
    const interval = setInterval(fetchSignals, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center gap-2 text-red-600 mb-2">
          <AlertCircle className="w-5 h-5" />
          <span className="font-medium">Failed to load signals</span>
        </div>
        <p className="text-gray-600 text-sm mb-4">{error}</p>
        <button
          onClick={fetchSignals}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Header */}
      <div className="px-4 py-3 border-b flex justify-between items-center">
        <div>
          <h3 className="font-semibold flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-500" />
            Ensemble Signals
            {signals.length > 0 && (
              <span className="bg-yellow-100 text-yellow-800 text-xs px-2 py-0.5 rounded-full">
                {signals.length}
              </span>
            )}
          </h3>
          <p className="text-xs text-gray-500 mt-1">
            DWAP +5% trigger + Top 20 momentum (2.5x avg returns)
          </p>
        </div>
        <button
          onClick={fetchSignals}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          disabled={loading}
        >
          <RefreshCw className={`w-4 h-4 text-gray-600 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Fresh signals banner */}
      {stats.freshCount > 0 && !loading && (
        <div className="px-4 py-2.5 bg-gradient-to-r from-green-50 to-emerald-50 border-b border-green-100">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-green-600" />
            <span className="text-sm font-medium text-green-800">
              {stats.freshCount} Fresh Signal{stats.freshCount > 1 ? 's' : ''} - BUY
            </span>
            <span className="text-xs text-green-600">
              (crossed DWAP +5% in last 5 days)
            </span>
          </div>
        </div>
      )}

      {/* Stats bar */}
      {!loading && (
        <div className="px-4 py-2 bg-gray-50 border-b text-xs text-gray-600 flex flex-wrap gap-3">
          {stats.freshCount > 0 && (
            <>
              <span className="flex items-center gap-1 text-green-700 font-medium">
                <Sparkles className="w-3 h-3" />
                {stats.freshCount} fresh
              </span>
              <span className="text-gray-400">|</span>
            </>
          )}
          <span className="text-gray-500">{stats.staleCount} stale</span>
          <span className="text-gray-400">|</span>
          <span>{stats.dwapOnly} DWAP-only</span>
          <span className="text-gray-400">|</span>
          <span>{stats.momentumOnly} momentum-only</span>
        </div>
      )}

      {/* Market filter warning */}
      {marketFilterActive && signals.length === 0 && !loading && (
        <div className="px-4 py-3 bg-yellow-50 border-b text-sm text-yellow-800">
          Market filter active: SPY is below 200-day MA. No signals available.
        </div>
      )}

      {/* Loading state */}
      {loading && signals.length === 0 && (
        <div className="p-8 text-center text-gray-500">
          <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
          Loading signals...
        </div>
      )}

      {/* Signals table */}
      {signals.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-gray-600">
              <tr>
                <th className="px-3 py-2 text-left font-medium">Symbol</th>
                <th className="px-3 py-2 text-right font-medium">Price</th>
                <th className="px-3 py-2 text-right font-medium">DWAP%</th>
                <th className="px-3 py-2 text-center font-medium">Trigger</th>
                <th className="px-3 py-2 text-right font-medium">Mom#</th>
                <th className="px-3 py-2 text-right font-medium hidden sm:table-cell">10d</th>
                <th className="px-3 py-2 text-right font-medium hidden sm:table-cell">60d</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {signals.map((s) => (
                <tr
                  key={s.symbol}
                  className={`cursor-pointer transition-colors ${
                    s.is_fresh
                      ? 'bg-green-50 hover:bg-green-100 border-l-4 border-l-green-500'
                      : 'bg-yellow-50/30 hover:bg-yellow-50'
                  }`}
                  onClick={() => onSymbolClick?.(s.symbol)}
                >
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-1.5">
                      {s.is_fresh ? (
                        <Sparkles className="w-4 h-4 text-green-600" title="Fresh signal - BUY" />
                      ) : (
                        <Zap className="w-4 h-4 text-yellow-500" title="Double signal (stale)" />
                      )}
                      <span className={`font-semibold ${s.is_fresh ? 'text-green-900' : 'text-gray-900'}`}>
                        {s.symbol}
                      </span>
                      {s.is_fresh && (
                        <span className="text-xs bg-green-600 text-white px-1.5 py-0.5 rounded font-medium">
                          BUY
                        </span>
                      )}
                      {s.is_strong && (
                        <ArrowUpRight className="w-3 h-3 text-green-500" title="Strong signal" />
                      )}
                    </div>
                  </td>
                  <td className="px-3 py-2.5 text-right">${s.price?.toFixed(2)}</td>
                  <td className={`px-3 py-2.5 text-right font-medium ${
                    s.pct_above_dwap > 10 ? 'text-green-600' :
                    s.pct_above_dwap > 7 ? 'text-green-500' :
                    'text-gray-700'
                  }`}>
                    +{s.pct_above_dwap?.toFixed(1)}%
                  </td>
                  <td className="px-3 py-2.5 text-center">
                    {s.dwap_crossover_date ? (
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        s.is_fresh ? 'bg-green-200 text-green-800 font-semibold' :
                        s.days_since_crossover <= 14 ? 'bg-yellow-100 text-yellow-700' :
                        'bg-gray-100 text-gray-500'
                      }`} title={`Crossed DWAP +5% on ${s.dwap_crossover_date}`}>
                        {s.days_since_crossover}d ago
                      </span>
                    ) : (
                      <span className="text-xs text-gray-400">60d+</span>
                    )}
                  </td>
                  <td className={`px-3 py-2.5 text-right font-medium ${
                    s.momentum_rank <= 5 ? 'text-green-600' :
                    s.momentum_rank <= 10 ? 'text-green-500' :
                    'text-gray-600'
                  }`}>
                    #{s.momentum_rank}
                  </td>
                  <td className={`px-3 py-2.5 text-right hidden sm:table-cell ${
                    s.short_momentum > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {s.short_momentum > 0 ? '+' : ''}{s.short_momentum?.toFixed(1)}%
                  </td>
                  <td className={`px-3 py-2.5 text-right hidden sm:table-cell ${
                    s.long_momentum > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {s.long_momentum > 0 ? '+' : ''}{s.long_momentum?.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* No signals */}
      {!loading && signals.length === 0 && !marketFilterActive && (
        <div className="p-8 text-center text-gray-500">
          <TrendingUp className="w-8 h-8 mx-auto mb-2 text-gray-300" />
          No stocks currently meet both DWAP and momentum criteria.
        </div>
      )}

      {/* Footer */}
      <div className="px-4 py-2 text-xs text-gray-500 border-t flex justify-between items-center">
        <span>
          Fresh = crossed +5% in last 5 days (actionable BUY)
        </span>
        {lastUpdated && (
          <span>Updated: {lastUpdated}</span>
        )}
      </div>
    </div>
  );
};

export default DoubleSignals;
