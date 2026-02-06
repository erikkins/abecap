import React, { useState, useEffect } from 'react';
import { TrendingUp, RefreshCw, AlertCircle, Zap } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const MomentumRankings = ({ onSymbolClick }) => {
  const [rankings, setRankings] = useState([]);
  const [doubleSignals, setDoubleSignals] = useState(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [marketFilterActive, setMarketFilterActive] = useState(false);

  const fetchRankings = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch both momentum rankings and double signals in parallel
      const [rankingsRes, doubleRes] = await Promise.all([
        fetch(`${API_BASE}/api/signals/momentum-rankings?top_n=20`),
        fetch(`${API_BASE}/api/signals/double-signals?momentum_top_n=20`)
      ]);

      if (!rankingsRes.ok) {
        throw new Error(`API error: ${rankingsRes.status}`);
      }

      const rankingsData = await rankingsRes.json();
      setRankings(rankingsData.rankings || []);
      setMarketFilterActive(rankingsData.market_filter_active);

      // Extract double signal symbols
      if (doubleRes.ok) {
        const doubleData = await doubleRes.json();
        const doubleSymbols = new Set((doubleData.signals || []).map(s => s.symbol));
        setDoubleSignals(doubleSymbols);
      }

      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      console.error('Failed to fetch momentum rankings:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRankings();
    // Refresh every 5 minutes
    const interval = setInterval(fetchRankings, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center gap-2 text-red-600 mb-2">
          <AlertCircle className="w-5 h-5" />
          <span className="font-medium">Failed to load rankings</span>
        </div>
        <p className="text-gray-600 text-sm mb-4">{error}</p>
        <button
          onClick={fetchRankings}
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
            <TrendingUp className="w-5 h-5 text-green-600" />
            Top Momentum Stocks
          </h3>
          <p className="text-xs text-gray-500 mt-1">
            Ranked by momentum score (higher = better)
          </p>
        </div>
        <button
          onClick={fetchRankings}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          disabled={loading}
        >
          <RefreshCw className={`w-4 h-4 text-gray-600 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Double signal legend */}
      {doubleSignals.size > 0 && (
        <div className="px-4 py-2 bg-yellow-50 border-b text-xs text-yellow-800 flex items-center gap-1">
          <Zap className="w-3 h-3" />
          <span>{doubleSignals.size} stocks also have DWAP +5% trigger (double signal)</span>
        </div>
      )}

      {/* Market filter warning */}
      {marketFilterActive && rankings.length === 0 && !loading && (
        <div className="px-4 py-3 bg-yellow-50 border-b text-sm text-yellow-800">
          Market filter active: SPY is below 200-day MA. Rankings may be limited.
        </div>
      )}

      {/* Loading state */}
      {loading && rankings.length === 0 && (
        <div className="p-8 text-center text-gray-500">
          <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
          Loading rankings...
        </div>
      )}

      {/* Rankings table */}
      {rankings.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-gray-600">
              <tr>
                <th className="px-3 py-2 text-left font-medium">#</th>
                <th className="px-3 py-2 text-left font-medium">Symbol</th>
                <th className="px-3 py-2 text-right font-medium">Price</th>
                <th className="px-3 py-2 text-right font-medium">Score</th>
                <th className="px-3 py-2 text-right font-medium">10d</th>
                <th className="px-3 py-2 text-right font-medium">60d</th>
                <th className="px-3 py-2 text-right font-medium hidden sm:table-cell">Vol</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {rankings.map((r) => {
                const isDouble = doubleSignals.has(r.symbol);
                return (
                  <tr
                    key={r.symbol}
                    className={`hover:bg-blue-50 cursor-pointer transition-colors ${isDouble ? 'bg-yellow-50' : ''}`}
                    onClick={() => onSymbolClick?.(r.symbol)}
                  >
                    <td className="px-3 py-2.5 text-gray-400 font-medium">{r.rank}</td>
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-1">
                        <span className="font-semibold text-gray-900">{r.symbol}</span>
                        {isDouble && (
                          <Zap className="w-4 h-4 text-yellow-500" title="Double signal: DWAP +5% trigger active" />
                        )}
                      </div>
                    </td>
                    <td className="px-3 py-2.5 text-right">${r.price?.toFixed(2)}</td>
                    <td className={`px-3 py-2.5 text-right font-semibold ${
                      r.momentum_score > 25 ? 'text-green-600' :
                      r.momentum_score > 15 ? 'text-green-500' :
                      r.momentum_score > 5 ? 'text-gray-700' :
                      'text-gray-500'
                    }`}>
                      {r.momentum_score?.toFixed(1)}
                    </td>
                    <td className={`px-3 py-2.5 text-right ${
                      r.short_momentum > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {r.short_momentum > 0 ? '+' : ''}{r.short_momentum?.toFixed(1)}%
                    </td>
                    <td className={`px-3 py-2.5 text-right ${
                      r.long_momentum > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {r.long_momentum > 0 ? '+' : ''}{r.long_momentum?.toFixed(1)}%
                    </td>
                    <td className="px-3 py-2.5 text-right text-gray-500 hidden sm:table-cell">
                      {r.volatility?.toFixed(0)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* No rankings */}
      {!loading && rankings.length === 0 && (
        <div className="p-8 text-center text-gray-500">
          No stocks currently pass the momentum quality filter.
        </div>
      )}

      {/* Footer */}
      <div className="px-4 py-2 text-xs text-gray-500 border-t flex justify-between items-center">
        <span>
          Score = 10d × 0.5 + 60d × 0.3 - Vol × 0.2
        </span>
        {lastUpdated && (
          <span>Updated: {lastUpdated}</span>
        )}
      </div>
    </div>
  );
};

export default MomentumRankings;
