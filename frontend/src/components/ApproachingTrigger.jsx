import React, { useState, useEffect } from 'react';
import { Eye, RefreshCw, AlertCircle, TrendingUp } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const ApproachingTrigger = ({ onSymbolClick }) => {
  const [approaching, setApproaching] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchApproaching = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/signals/approaching-trigger?momentum_top_n=20&min_pct=3&max_pct=5`);
      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }
      const data = await res.json();
      setApproaching(data.approaching || []);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      console.error('Failed to fetch approaching trigger:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchApproaching();
    // Refresh every 5 minutes
    const interval = setInterval(fetchApproaching, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex items-center gap-2 text-red-600 mb-2">
          <AlertCircle className="w-4 h-4" />
          <span className="text-sm font-medium">Failed to load</span>
        </div>
        <button
          onClick={fetchApproaching}
          className="px-3 py-1.5 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (approaching.length === 0 && !loading) {
    return null; // Don't render if no approaching stocks
  }

  return (
    <div className="bg-amber-50 border border-amber-200 rounded-lg">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-amber-200 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-amber-600" />
          <h3 className="font-medium text-amber-800 text-sm">
            Approaching Trigger
            {approaching.length > 0 && (
              <span className="ml-1.5 bg-amber-200 text-amber-800 text-xs px-1.5 py-0.5 rounded-full">
                {approaching.length}
              </span>
            )}
          </h3>
        </div>
        <button
          onClick={fetchApproaching}
          className="p-1 hover:bg-amber-100 rounded transition-colors"
          disabled={loading}
        >
          <RefreshCw className={`w-3.5 h-3.5 text-amber-600 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Description */}
      <div className="px-4 py-2 text-xs text-amber-700 border-b border-amber-100">
        Momentum stocks at +3-4% DWAP (close to +5% trigger)
      </div>

      {/* Loading state */}
      {loading && approaching.length === 0 && (
        <div className="p-4 text-center text-amber-600 text-sm">
          <RefreshCw className="w-4 h-4 animate-spin mx-auto mb-1" />
          Loading...
        </div>
      )}

      {/* Approaching list */}
      {approaching.length > 0 && (
        <div className="divide-y divide-amber-100">
          {approaching.map((s) => (
            <div
              key={s.symbol}
              className="px-4 py-2.5 hover:bg-amber-100 cursor-pointer transition-colors flex items-center justify-between"
              onClick={() => onSymbolClick?.(s.symbol)}
            >
              <div className="flex items-center gap-3">
                <div>
                  <span className="font-semibold text-gray-900 text-sm">{s.symbol}</span>
                  <span className="text-xs text-amber-600 ml-1.5">
                    #{s.momentum_rank}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-4 text-right">
                <div>
                  <div className="text-sm font-medium text-gray-900">${s.price?.toFixed(2)}</div>
                  <div className="text-xs text-amber-700">+{s.pct_above_dwap?.toFixed(1)}%</div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-amber-700 font-medium">
                    +{s.distance_to_trigger?.toFixed(1)}%
                  </div>
                  <div className="text-xs text-amber-600">to trigger</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      {lastUpdated && (
        <div className="px-4 py-1.5 text-xs text-amber-600 border-t border-amber-100">
          Updated: {lastUpdated}
        </div>
      )}
    </div>
  );
};

export default ApproachingTrigger;
