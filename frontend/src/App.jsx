import React, { useState, useEffect, useCallback } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ComposedChart, Bar, ReferenceLine, ReferenceDot, Legend
} from 'recharts';
import {
  TrendingUp, TrendingDown, RefreshCw, Settings, Bell, User, LogOut,
  DollarSign, Target, Shield, Activity, PieChart as PieIcon, History,
  ArrowUpRight, ArrowDownRight, Clock, Zap, X, ChevronRight, Eye,
  Calendar, BarChart3, Wallet, LogIn, AlertCircle, Loader2
} from 'lucide-react';
import LandingPage from './LandingPage';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import LoginModal from './components/LoginModal';
import AdminDashboard from './components/AdminDashboard';
import SubscriptionBanner from './components/SubscriptionBanner';
import MomentumRankings from './components/MomentumRankings';
import DoubleSignals from './components/DoubleSignals';

// ============================================================================
// API Configuration
// ============================================================================

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// CDN URL for static signals (same bucket as price data, publicly accessible)
const SIGNALS_CDN_URL = 'https://rigacap-prod-price-data-149218244179.s3.amazonaws.com/signals/latest.json';

// localStorage cache keys
const CACHE_KEYS = {
  SIGNALS: 'rigacap_signals_cache',
  POSITIONS: 'rigacap_positions_cache',
  MISSED: 'rigacap_missed_cache',
  BACKTEST: 'rigacap_backtest_cache',
  CACHE_TIME: 'rigacap_cache_time'
};

// Cache duration: 5 minutes for signals, 1 hour for user data
const CACHE_DURATION = {
  SIGNALS: 5 * 60 * 1000,  // 5 minutes
  USER_DATA: 60 * 60 * 1000  // 1 hour
};

// Helper to get cached data
const getCache = (key) => {
  try {
    const cached = localStorage.getItem(key);
    if (cached) return JSON.parse(cached);
  } catch (e) {
    console.log('Cache read error:', e);
  }
  return null;
};

// Helper to set cached data
const setCache = (key, data) => {
  try {
    localStorage.setItem(key, JSON.stringify(data));
    localStorage.setItem(CACHE_KEYS.CACHE_TIME + '_' + key, Date.now().toString());
  } catch (e) {
    console.log('Cache write error:', e);
  }
};

// Helper to check if cache is still valid
const isCacheValid = (key, maxAge) => {
  try {
    const cacheTime = localStorage.getItem(CACHE_KEYS.CACHE_TIME + '_' + key);
    if (!cacheTime) return false;
    return (Date.now() - parseInt(cacheTime)) < maxAge;
  } catch (e) {
    return false;
  }
};

const api = {
  async get(endpoint) {
    const res = await fetch(`${API_BASE}${endpoint}`);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
  },
  async post(endpoint, data) {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
  },
  async delete(endpoint) {
    const res = await fetch(`${API_BASE}${endpoint}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
  }
};

// Normalize signal data types (S3 JSON may have strings instead of numbers/booleans)
const normalizeSignal = (signal) => ({
  ...signal,
  signal_strength: typeof signal.signal_strength === 'string' ? parseFloat(signal.signal_strength) : (signal.signal_strength || 0),
  is_strong: signal.is_strong === true || signal.is_strong === 'True' || signal.is_strong === 'true',
  price: typeof signal.price === 'string' ? parseFloat(signal.price) : signal.price,
  dwap: typeof signal.dwap === 'string' ? parseFloat(signal.dwap) : signal.dwap,
  pct_above_dwap: typeof signal.pct_above_dwap === 'string' ? parseFloat(signal.pct_above_dwap) : signal.pct_above_dwap,
  volume: typeof signal.volume === 'string' ? parseInt(signal.volume, 10) : signal.volume,
});

// Fetch signals from CDN (static JSON, instant load)
const fetchSignalsFromCDN = async () => {
  try {
    const res = await fetch(SIGNALS_CDN_URL, { cache: 'default' });
    if (!res.ok) throw new Error(`CDN error: ${res.status}`);
    const data = await res.json();
    return (data.signals || []).map(normalizeSignal);
  } catch (e) {
    console.log('CDN signals fetch failed, falling back to API:', e);
    return null; // Will trigger API fallback
  }
};

// Note: AuthContext, useAuth, LoginModal, AdminDashboard, SubscriptionBanner
// are now imported from separate files

// ============================================================================
// Components
// ============================================================================

// Custom triangle markers for buy/sell points on charts
const BuyMarker = ({ cx, cy, payload }) => (
  <svg x={cx - 10} y={cy - 20} width={20} height={20} viewBox="0 0 20 20" style={{ cursor: 'pointer' }}>
    <title>BUY POINT: {payload?.date} @ ${payload?.close?.toFixed(2)}</title>
    <polygon
      points="10,2 18,18 2,18"
      fill="#10B981"
      stroke="#059669"
      strokeWidth="1"
    />
    <text x="10" y="14" textAnchor="middle" fontSize="8" fill="white" fontWeight="bold">B</text>
  </svg>
);

const SellMarker = ({ cx, cy, payload }) => (
  <svg x={cx - 10} y={cy} width={20} height={20} viewBox="0 0 20 20" style={{ cursor: 'pointer' }}>
    <title>SELL POINT (+20%): {payload?.date} @ ${payload?.close?.toFixed(2)}</title>
    <polygon
      points="10,18 18,2 2,2"
      fill="#EF4444"
      stroke="#DC2626"
      strokeWidth="1"
    />
    <text x="10" y="12" textAnchor="middle" fontSize="8" fill="white" fontWeight="bold">S</text>
  </svg>
);

// Loading Spinner
const LoadingSpinner = ({ message = "Loading..." }) => (
  <div className="flex flex-col items-center justify-center py-12">
    <Loader2 className="w-8 h-8 text-blue-600 animate-spin mb-3" />
    <p className="text-gray-500">{message}</p>
  </div>
);

// Error Display
const ErrorDisplay = ({ message, onRetry }) => (
  <div className="flex flex-col items-center justify-center py-12">
    <AlertCircle className="w-12 h-12 text-red-400 mb-3" />
    <p className="text-red-600 mb-4">{message}</p>
    {onRetry && (
      <button onClick={onRetry} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
        Retry
      </button>
    )}
  </div>
);

// LoginModal is now imported from ./components/LoginModal

// Buy Modal Component
const BuyModal = ({ symbol, price, stockInfo, onClose, onBuy }) => {
  const [shares, setShares] = useState(Math.floor(10000 / price)); // Default ~$10k position
  const [entryPrice, setEntryPrice] = useState(price);
  const [submitting, setSubmitting] = useState(false);

  const totalCost = shares * entryPrice;
  const stopLoss = entryPrice * 0.92; // 8% stop
  const profitTarget = entryPrice * 1.20; // 20% target

  const handleBuy = async () => {
    setSubmitting(true);
    try {
      await api.post('/api/portfolio/positions', {
        symbol,
        shares,
        price: entryPrice
      });
      onBuy();
      onClose();
    } catch (err) {
      console.error('Buy failed:', err);
      alert('Failed to create position. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-[60] p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 bg-gradient-to-r from-emerald-500 to-green-600">
          <h2 className="text-xl font-bold text-white">Buy {symbol}</h2>
          {stockInfo?.name && <p className="text-emerald-100 text-sm">{stockInfo.name}</p>}
        </div>

        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Number of Shares</label>
            <input
              type="number"
              value={shares}
              onChange={(e) => setShares(Math.max(1, parseInt(e.target.value) || 0))}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              min="1"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Entry Price</label>
            <input
              type="number"
              step="0.01"
              value={entryPrice}
              onChange={(e) => setEntryPrice(parseFloat(e.target.value) || 0)}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
            />
          </div>

          <div className="bg-gray-50 rounded-xl p-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Total Cost</span>
              <span className="font-semibold">${totalCost.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Stop Loss (8%)</span>
              <span className="text-red-500 font-medium">${stopLoss.toFixed(2)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Profit Target (20%)</span>
              <span className="text-emerald-600 font-medium">${profitTarget.toFixed(2)}</span>
            </div>
          </div>
        </div>

        <div className="px-6 py-4 border-t border-gray-100 flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-3 text-gray-600 hover:bg-gray-100 rounded-xl font-medium"
          >
            Cancel
          </button>
          <button
            onClick={handleBuy}
            disabled={submitting || shares < 1 || entryPrice <= 0}
            className="flex-1 px-4 py-3 bg-emerald-600 text-white rounded-xl font-medium hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <DollarSign size={18} />}
            {submitting ? 'Saving...' : 'Track Position'}
          </button>
        </div>
      </div>
    </div>
  );
};

// Sell Modal Component (for closing positions)
const SellModal = ({ symbol, position, currentPrice, stockInfo, onClose, onSell }) => {
  const [shares, setShares] = useState(position?.shares || 0);
  const [exitPrice, setExitPrice] = useState(currentPrice);
  const [submitting, setSubmitting] = useState(false);

  const entryPrice = position?.entry_price || 0;
  const totalProceeds = shares * exitPrice;
  const totalCost = shares * entryPrice;
  const pnl = totalProceeds - totalCost;
  const pnlPct = entryPrice > 0 ? ((exitPrice - entryPrice) / entryPrice) * 100 : 0;

  const handleSell = async () => {
    setSubmitting(true);
    try {
      await api.delete(`/api/portfolio/positions/${position.id}?exit_price=${exitPrice}`);
      onSell();
      onClose();
    } catch (err) {
      console.error('Sell failed:', err);
      alert('Failed to close position. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-[60] p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full overflow-hidden">
        <div className={`px-6 py-4 border-b border-gray-100 ${pnl >= 0 ? 'bg-gradient-to-r from-emerald-500 to-green-600' : 'bg-gradient-to-r from-red-500 to-rose-600'}`}>
          <h2 className="text-xl font-bold text-white">Sell {symbol}</h2>
          {stockInfo?.name && <p className="text-white/80 text-sm">{stockInfo.name}</p>}
        </div>

        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Number of Shares</label>
            <input
              type="number"
              value={shares}
              onChange={(e) => setShares(Math.max(0, Math.min(position?.shares || 0, parseFloat(e.target.value) || 0)))}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              max={position?.shares || 0}
            />
            <p className="text-xs text-gray-400 mt-1">Max: {position?.shares || 0} shares</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Exit Price</label>
            <input
              type="number"
              step="0.01"
              value={exitPrice}
              onChange={(e) => setExitPrice(parseFloat(e.target.value) || 0)}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
            />
          </div>

          <div className="bg-gray-50 rounded-xl p-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Entry Price</span>
              <span className="font-medium">${entryPrice.toFixed(2)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Total Proceeds</span>
              <span className="font-semibold">${totalProceeds.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
            </div>
            <div className="border-t border-gray-200 my-2"></div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Profit/Loss</span>
              <span className={`font-bold ${pnl >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                {pnl >= 0 ? '+' : ''}{pnl.toLocaleString(undefined, { maximumFractionDigits: 2 })} ({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%)
              </span>
            </div>
          </div>
        </div>

        <div className="px-6 py-4 border-t border-gray-100 flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-3 text-gray-600 hover:bg-gray-100 rounded-xl font-medium"
          >
            Cancel
          </button>
          <button
            onClick={handleSell}
            disabled={submitting || shares <= 0 || exitPrice <= 0}
            className={`flex-1 px-4 py-3 text-white rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 ${pnl >= 0 ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-red-600 hover:bg-red-700'}`}
          >
            {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <DollarSign size={18} />}
            {submitting ? 'Selling...' : 'Confirm Sale'}
          </button>
        </div>
      </div>
    </div>
  );
};

// Stock Chart Modal
const StockChartModal = ({ symbol, type, data, onClose, onAction, liveQuote }) => {
  const [timeRange, setTimeRange] = useState('1Y');
  const [priceData, setPriceData] = useState([]);
  const [stockInfo, setStockInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showBuyModal, setShowBuyModal] = useState(false);
  const [showSellModal, setShowSellModal] = useState(false);
  const [currentLiveQuote, setCurrentLiveQuote] = useState(liveQuote);

  // Poll for live quote updates while modal is open
  useEffect(() => {
    const fetchLiveQuote = async () => {
      try {
        const response = await api.get(`/api/quotes/live?symbols=${symbol}`);
        if (response.quotes?.[symbol]) {
          setCurrentLiveQuote(response.quotes[symbol]);
        }
      } catch (err) {
        // Silently fail - live quotes are optional
      }
    };

    // Initial fetch and poll every 15 seconds while modal is open
    fetchLiveQuote();
    const interval = setInterval(fetchLiveQuote, 15000);
    return () => clearInterval(interval);
  }, [symbol]);

  // Fetch company info once when modal opens
  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const info = await api.get(`/api/signals/info/${symbol}`);
        setStockInfo(info);
      } catch (err) {
        console.log('Could not fetch stock info');
      }
    };
    fetchInfo();
  }, [symbol]);

  useEffect(() => {
    const fetchHistory = async () => {
      setLoading(true);
      setError(null);
      try {
        // For missed opportunities, fetch enough data to show the transaction window
        let days = { '1M': 30, '3M': 90, '6M': 180, '1Y': 252, '2Y': 504 }[timeRange] || 252;

        // For missed opportunities, we need enough data to cover entry_date - 30 days
        if (type === 'missed' && data?.entry_date) {
          const entryDate = new Date(data.entry_date);
          const today = new Date();
          const daysSinceEntry = Math.ceil((today - entryDate) / (1000 * 60 * 60 * 24));
          days = Math.max(days, daysSinceEntry + 60); // Extra buffer for 30 days before entry
        }

        const response = await api.get(`/api/stock/${symbol}/history?days=${days}`);
        let chartData = response.data || [];

        // For missed opportunities, filter to show transaction window (30 days before buy, 30 days after sell)
        if (type === 'missed' && data?.entry_date && data?.sell_date) {
          const entryDate = new Date(data.entry_date);
          const sellDate = new Date(data.sell_date);
          const windowStart = new Date(entryDate);
          windowStart.setDate(windowStart.getDate() - 30);
          const windowEnd = new Date(sellDate);
          windowEnd.setDate(windowEnd.getDate() + 30);

          chartData = chartData.filter(d => {
            const date = new Date(d.date);
            return date >= windowStart && date <= windowEnd;
          });
        }

        setPriceData(chartData);
      } catch (err) {
        setError('Failed to load chart data');
        setPriceData([]);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [symbol, timeRange, type, data?.entry_date, data?.sell_date]);

  // Format market cap for display
  const formatMarketCap = (mcap) => {
    if (!mcap) return '';
    const num = parseFloat(mcap.replace(/,/g, ''));
    if (isNaN(num)) return mcap;
    if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(1)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(0)}M`;
    return `$${num.toLocaleString()}`;
  };

  // Use live quote if available, otherwise fall back to chart data
  const livePrice = currentLiveQuote?.price;
  const currentPrice = livePrice || priceData[priceData.length - 1]?.close || data?.current_price || data?.price || 0;
  const startPrice = priceData[0]?.close || currentPrice;
  const changePct = startPrice > 0 ? ((currentPrice - startPrice) / startPrice * 100).toFixed(1) : 0;
  const isPositive = changePct >= 0;

  // Add live price point to chart data if available
  const chartDataWithLive = livePrice && priceData.length > 0
    ? [...priceData, {
        date: new Date().toISOString().split('T')[0],
        close: livePrice,
        open: livePrice,
        high: livePrice,
        low: livePrice,
        isLive: true, // Flag for special rendering
      }]
    : priceData;

  // Find entry point index for positions
  const entryPointIndex = type === 'position' && data?.entry_date
    ? chartDataWithLive.findIndex(d => d.date === data.entry_date)
    : -1;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-100 relative flex-shrink-0">
          {/* Close button - top right */}
          <button onClick={onClose} className="absolute top-4 right-4 p-2 hover:bg-gray-100 rounded-lg z-10">
            <X size={24} className="text-gray-400" />
          </button>

          <div className="pr-12">
            <div className="flex items-center gap-3 flex-wrap">
              <h2 className="text-2xl font-bold text-gray-900">{symbol}</h2>
              {data?.is_strong && (
                <span className="px-2 py-1 bg-emerald-100 text-emerald-700 text-xs font-semibold rounded-full flex items-center gap-1">
                  <Zap size={12} /> STRONG SIGNAL
                </span>
              )}
              {type === 'position' && (
                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs font-semibold rounded-full">
                  OPEN POSITION
                </span>
              )}
              {type === 'missed' && (
                <span className="px-2 py-1 bg-amber-100 text-amber-700 text-xs font-semibold rounded-full flex items-center gap-1">
                  <Clock size={12} /> MISSED +20%
                </span>
              )}
              {data?.signal_strength > 0 && (
                <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                  data.signal_strength >= 70 ? 'bg-emerald-100 text-emerald-700' :
                  data.signal_strength >= 50 ? 'bg-blue-100 text-blue-700' :
                  'bg-yellow-100 text-yellow-700'
                }`}>
                  Strength: {data.signal_strength.toFixed(0)}
                </span>
              )}
            </div>
            {/* Company Name & Sector */}
            {stockInfo?.name && (
              <div className="mt-1">
                <p className="text-gray-600 text-sm">{stockInfo.name}</p>
                {stockInfo?.sector && (
                  <span className="inline-block mt-1 px-2 py-0.5 bg-indigo-100 text-indigo-700 text-xs font-medium rounded-full">
                    {stockInfo.sector}{stockInfo?.industry ? ` - ${stockInfo.industry}` : ''}
                  </span>
                )}
              </div>
            )}
            <div className="flex items-center gap-4 mt-2">
              <span className="text-2xl font-semibold">${currentPrice.toFixed(2)}</span>
              {currentLiveQuote && (
                <span className={`flex items-center text-sm font-medium ${currentLiveQuote.change_pct >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                  {currentLiveQuote.change_pct >= 0 ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
                  {currentLiveQuote.change_pct >= 0 ? '+' : ''}{currentLiveQuote.change_pct?.toFixed(2)}% today
                </span>
              )}
              <span className={`flex items-center text-sm font-medium ${isPositive ? 'text-emerald-600' : 'text-red-500'}`}>
                {isPositive ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
                {isPositive ? '+' : ''}{changePct}% ({timeRange})
              </span>
              {currentLiveQuote && (
                <span className="flex items-center gap-1 text-xs text-blue-500">
                  <Activity size={12} className="animate-pulse" /> Live
                </span>
              )}
              {stockInfo?.market_cap && (
                <span className="text-sm text-gray-500">
                  Market Cap: {formatMarketCap(stockInfo.market_cap)}
                </span>
              )}
            </div>
            {/* Company Description - scrollable */}
            {stockInfo?.description && (
              <div className="mt-2 max-h-20 overflow-y-auto">
                <p className="text-sm text-gray-500">{stockInfo.description}</p>
              </div>
            )}
          </div>
        </div>

        {/* Scrollable content area */}
        <div className="flex-1 overflow-y-auto">
          {/* Time Range */}
          <div className="px-6 py-3 border-b border-gray-100 flex gap-2 items-center">
          {type === 'missed' ? (
            <div className="flex items-center gap-2">
              <span className="px-4 py-1.5 rounded-lg text-sm font-medium bg-amber-100 text-amber-700">
                Transaction Window
              </span>
              <span className="text-sm text-gray-500">
                {data?.entry_date} → {data?.sell_date} (±30 days)
              </span>
            </div>
          ) : (
            ['1M', '3M', '6M', '1Y', '2Y'].map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  timeRange === range ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {range}
              </button>
            ))
          )}
        </div>

        {/* Chart */}
        <div className="p-6">
          {loading ? (
            <LoadingSpinner message="Loading chart data..." />
          ) : error ? (
            <ErrorDisplay message={error} />
          ) : priceData.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <BarChart3 className="w-12 h-12 mx-auto text-gray-300 mb-3" />
              <p>No price data available</p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={chartDataWithLive}>
                <defs>
                  <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.15}/>
                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" vertical={false} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11 }}
                  stroke="#9CA3AF"
                  tickFormatter={(val) => new Date(val).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  interval={Math.floor(priceData.length / 6)}
                />
                <YAxis
                  yAxisId="price"
                  tick={{ fontSize: 11 }}
                  stroke="#9CA3AF"
                  domain={['dataMin - 10', 'dataMax + 10']}
                  tickFormatter={(val) => `$${val.toFixed(0)}`}
                />
                <YAxis
                  yAxisId="volume"
                  orientation="right"
                  tick={{ fontSize: 10 }}
                  stroke="#D1D5DB"
                  tickFormatter={(val) => `${(val / 1000000).toFixed(0)}M`}
                />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0]?.payload;
                    const isEntryDay = d?.date === data?.entry_date;
                    const isSellDay = d?.date === data?.sell_date;
                    const borderClass = isEntryDay ? 'border-emerald-400 border-2' :
                                       isSellDay ? 'border-amber-400 border-2' : 'border-gray-200';
                    return (
                      <div className={`bg-white p-3 rounded-lg shadow-lg border ${borderClass} text-sm`}>
                        <p className="font-medium text-gray-900 mb-1">
                          {new Date(label).toLocaleDateString()}
                          {isEntryDay && <span className="ml-2 text-emerald-600 font-bold">BUY POINT</span>}
                          {isSellDay && <span className="ml-2 text-amber-600 font-bold">SELL POINT (+20%)</span>}
                        </p>
                        <p className="text-blue-600">Price: ${d?.close?.toFixed(2)}</p>
                        {isEntryDay && data?.entry_price && (
                          <p className="text-emerald-600 font-medium">Entry: ${data.entry_price.toFixed(2)}</p>
                        )}
                        {isSellDay && data?.sell_price && (
                          <p className="text-amber-600 font-medium">Exit: ${data.sell_price.toFixed(2)}</p>
                        )}
                        {d?.dwap && <p className="text-purple-600">DWAP: ${d.dwap.toFixed(2)}</p>}
                        {d?.ma_50 && <p className="text-orange-500">MA50: ${d.ma_50.toFixed(2)}</p>}
                        {d?.volume > 0 && <p className="text-gray-400">Vol: {(d.volume / 1000000).toFixed(1)}M</p>}
                        {d?.isLive && <p className="text-blue-500 font-medium">Live Price</p>}
                      </div>
                    );
                  }}
                />
                <Bar yAxisId="volume" dataKey="volume" fill="#E5E7EB" opacity={0.5} />
                {chartDataWithLive.some(d => d.dwap) && (
                  <Line yAxisId="price" type="monotone" dataKey="dwap" stroke="#8B5CF6" strokeWidth={2} dot={false} name="DWAP" />
                )}
                {chartDataWithLive.some(d => d.ma_50) && (
                  <Line yAxisId="price" type="monotone" dataKey="ma_50" stroke="#F97316" strokeWidth={1.5} dot={false} strokeDasharray="5 5" name="MA50" />
                )}
                <Area yAxisId="price" type="monotone" dataKey="close" stroke="#3B82F6" strokeWidth={2} fill="url(#priceGradient)" name="Price" />

                {/* Entry price marker for positions and missed opportunities */}
                {(type === 'position' || type === 'missed') && data?.entry_price && (
                  <ReferenceLine
                    yAxisId="price"
                    y={data.entry_price}
                    stroke="#10B981"
                    strokeWidth={2}
                    strokeDasharray="8 4"
                    label={{
                      value: `Buy $${data.entry_price.toFixed(2)}`,
                      fill: '#10B981',
                      fontWeight: 'bold',
                      fontSize: 12,
                      position: 'insideLeft'
                    }}
                  />
                )}

                {/* Exit/Sell price marker for missed opportunities */}
                {type === 'missed' && data?.sell_price && (
                  <ReferenceLine
                    yAxisId="price"
                    y={data.sell_price}
                    stroke="#F59E0B"
                    strokeWidth={2}
                    strokeDasharray="8 4"
                    label={{
                      value: `Sell $${data.sell_price.toFixed(2)} (+20%)`,
                      fill: '#F59E0B',
                      fontWeight: 'bold',
                      fontSize: 12,
                      position: 'insideRight'
                    }}
                  />
                )}

                {/* Stop loss line */}
                {data?.stop_loss && (
                  <ReferenceLine
                    yAxisId="price"
                    y={data.stop_loss}
                    stroke="#EF4444"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    label={{
                      value: `Stop $${data.stop_loss.toFixed(2)}`,
                      fill: '#EF4444',
                      fontSize: 10,
                      position: 'insideBottomRight'
                    }}
                  />
                )}

                {/* Profit target line */}
                {data?.profit_target && (
                  <ReferenceLine
                    yAxisId="price"
                    y={data.profit_target}
                    stroke="#10B981"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    label={{
                      value: `Target $${data.profit_target.toFixed(2)}`,
                      fill: '#10B981',
                      fontSize: 10,
                      position: 'insideTopRight'
                    }}
                  />
                )}

                {/* Buy point marker - triangle at entry date */}
                {(() => {
                  if (!data?.entry_date || chartDataWithLive.length === 0) return null;

                  // Normalize entry date to YYYY-MM-DD format for comparison
                  const entryDateStr = data.entry_date.split('T')[0];

                  // Find exact match or closest date on/after entry date
                  let entryMatch = chartDataWithLive.find(d => d.date === entryDateStr);
                  if (!entryMatch) {
                    // Find closest date on or after entry_date (entry might be on weekend/holiday)
                    entryMatch = chartDataWithLive.find(d => d.date >= entryDateStr);
                  }
                  if (!entryMatch) {
                    // If entry is before all chart data, don't show marker (it's out of view)
                    return null;
                  }

                  // Use actual entry_price for y position (not close price which may differ)
                  const yPrice = data.entry_price || entryMatch.close;
                  if (!yPrice || !entryMatch.date) return null;

                  return (
                    <ReferenceDot
                      yAxisId="price"
                      x={entryMatch.date}
                      y={yPrice}
                      shape={(props) => <BuyMarker {...props} payload={{...entryMatch, close: yPrice}} />}
                    />
                  );
                })()}

                {/* Sell point marker - triangle at sell date (for trades) */}
                {(() => {
                  if (!data?.sell_date || chartDataWithLive.length === 0) return null;

                  // Normalize sell date to YYYY-MM-DD format for comparison
                  const sellDateStr = data.sell_date.split('T')[0];

                  // Find exact match or closest date on/after sell date
                  let sellMatch = chartDataWithLive.find(d => d.date === sellDateStr);
                  if (!sellMatch) {
                    sellMatch = chartDataWithLive.find(d => d.date >= sellDateStr);
                  }
                  if (!sellMatch) {
                    // If sell date is after all chart data, use last point
                    sellMatch = chartDataWithLive[chartDataWithLive.length - 1];
                  }
                  if (!sellMatch?.date) return null;
                  return sellMatch ? (
                    <ReferenceDot
                      yAxisId="price"
                      x={sellMatch.date}
                      y={sellMatch.close || data.sell_price}
                      shape={(props) => <SellMarker {...props} payload={sellMatch} />}
                    />
                  ) : null;
                })()}

                {/* Signal point marker - triangle at current date for NEW signals only (not missed opportunities) */}
                {type === 'signal' && !data?.exit_date && chartDataWithLive.length > 0 && !livePrice && (
                  <ReferenceDot
                    yAxisId="price"
                    x={chartDataWithLive[chartDataWithLive.length - 1]?.date}
                    y={chartDataWithLive[chartDataWithLive.length - 1]?.close}
                    shape={BuyMarker}
                  />
                )}

                {/* Live price marker - pulsing dot at current live price */}
                {livePrice && chartDataWithLive.length > 0 && (
                  <ReferenceDot
                    yAxisId="price"
                    x={chartDataWithLive[chartDataWithLive.length - 1]?.date}
                    y={livePrice}
                    r={8}
                    fill="#3B82F6"
                    stroke="#fff"
                    strokeWidth={2}
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Details */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-100">
          {/* Recommendation banner */}
          {data?.recommendation && (
            <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800">
              <strong>Recommendation:</strong> {data.recommendation}
            </div>
          )}

          <div className="grid grid-cols-4 gap-4">
            {type === 'signal' ? (
              <>
                <div className="text-center">
                  <p className="text-sm text-gray-500">DWAP Signal</p>
                  <p className="text-lg font-semibold text-emerald-600">+{data?.pct_above_dwap}%</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Volume</p>
                  <p className="text-lg font-semibold">{data?.volume_ratio}x avg</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Stop Loss</p>
                  <p className="text-lg font-semibold text-red-500">${data?.stop_loss?.toFixed(2)}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Profit Target</p>
                  <p className="text-lg font-semibold text-emerald-600">${data?.profit_target?.toFixed(2)}</p>
                </div>
              </>
            ) : type === 'missed' ? (
              <>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Buy Date</p>
                  <p className="text-lg font-semibold">{data?.entry_date}</p>
                  <p className="text-xs text-emerald-600">${data?.entry_price?.toFixed(2)}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Sell Date</p>
                  <p className="text-lg font-semibold">{data?.sell_date}</p>
                  <p className="text-xs text-emerald-600">${data?.sell_price?.toFixed(2)}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Return</p>
                  <p className="text-lg font-semibold text-emerald-600">
                    +{data?.would_be_return?.toFixed(1)}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Days Held</p>
                  <p className="text-lg font-semibold">{data?.days_held || '-'}</p>
                </div>
              </>
            ) : (
              <>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Entry Price</p>
                  <p className="text-lg font-semibold">${data?.entry_price?.toFixed(2)}</p>
                  <p className="text-xs text-gray-400">{data?.entry_date}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Current P&L</p>
                  <p className={`text-lg font-semibold ${data?.pnl_pct >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                    {data?.pnl_pct >= 0 ? '+' : ''}{data?.pnl_pct?.toFixed(1)}%
                  </p>
                  <p className={`text-xs ${data?.pnl_dollars >= 0 ? 'text-emerald-500' : 'text-red-400'}`}>
                    ${data?.pnl_dollars?.toFixed(0) || ((data?.current_price - data?.entry_price) * data?.shares).toFixed(0)}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Shares</p>
                  <p className="text-lg font-semibold">{data?.shares?.toFixed(2)}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Days Held</p>
                  <p className="text-lg font-semibold">{data?.days_held}</p>
                </div>
              </>
            )}
          </div>

          {/* Technical Indicators - Signal only */}
          {type === 'signal' && (data?.ma_50 || stockInfo?.ma_50) && (
            <div className="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-gray-200">
              <div className="text-center">
                <p className="text-sm text-gray-500">50-Day MA</p>
                <p className="text-lg font-semibold">${(data?.ma_50 || stockInfo?.ma_50)?.toFixed(2)}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-500">200-Day MA</p>
                <p className="text-lg font-semibold">${(data?.ma_200 || stockInfo?.ma_200)?.toFixed(2)}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-500">52-Week High</p>
                <p className="text-lg font-semibold">${(data?.high_52w || stockInfo?.high_52w)?.toFixed(2)}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-500">DWAP Price</p>
                <p className="text-lg font-semibold">${data?.dwap?.toFixed(2)}</p>
              </div>
            </div>
          )}
        </div>
        </div>

        {/* Actions - fixed footer */}
        <div className="px-6 py-4 border-t border-gray-100 flex justify-end gap-3 flex-shrink-0 bg-white">
          <button onClick={onClose} className="px-6 py-2.5 text-gray-600 hover:bg-gray-100 rounded-xl font-medium">
            Close
          </button>
          {type === 'signal' && !data?.exit_date && (
            <button
              onClick={() => setShowBuyModal(true)}
              className="px-6 py-2.5 bg-emerald-600 text-white rounded-xl font-medium hover:bg-emerald-700 flex items-center gap-2"
            >
              <DollarSign size={18} />
              Track Position
            </button>
          )}
          {type === 'position' && (
            <button
              onClick={() => setShowSellModal(true)}
              className="px-6 py-2.5 bg-red-600 text-white rounded-xl font-medium hover:bg-red-700 flex items-center gap-2"
            >
              <DollarSign size={18} />
              Mark as Sold
            </button>
          )}
          {type === 'missed' && (
            <div className="px-4 py-2 bg-amber-50 text-amber-700 rounded-xl text-sm">
              This opportunity has already passed
            </div>
          )}
        </div>
      </div>

      {/* Buy Modal */}
      {showBuyModal && (
        <BuyModal
          symbol={symbol}
          price={currentPrice}
          stockInfo={stockInfo}
          onClose={() => setShowBuyModal(false)}
          onBuy={() => {
            onAction && onAction();
          }}
        />
      )}

      {/* Sell Modal */}
      {showSellModal && (
        <SellModal
          symbol={symbol}
          position={data}
          currentPrice={currentPrice}
          stockInfo={stockInfo}
          onClose={() => setShowSellModal(false)}
          onSell={() => {
            onAction && onAction();
          }}
        />
      )}
    </div>
  );
};

// Metric Card
const MetricCard = ({ title, value, subtitle, trend, icon: Icon }) => (
  <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 hover:shadow-md transition-all">
    <div className="flex items-start justify-between">
      <div>
        <p className="text-sm text-gray-500 font-medium">{title}</p>
        <p className={`text-2xl font-bold mt-1 ${trend === 'up' ? 'text-emerald-600' : trend === 'down' ? 'text-red-500' : 'text-gray-900'}`}>{value}</p>
        {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
      </div>
      {Icon && (
        <div className={`p-2 rounded-lg ${trend === 'up' ? 'bg-emerald-50 text-emerald-600' : trend === 'down' ? 'bg-red-50 text-red-500' : 'bg-gray-50 text-gray-400'}`}>
          <Icon size={20} />
        </div>
      )}
    </div>
  </div>
);

// Signal Strength indicator
const SignalStrengthBar = ({ strength }) => {
  const numStrength = typeof strength === 'string' ? parseFloat(strength) : (strength || 0);
  const color = numStrength >= 70 ? 'bg-emerald-500' : numStrength >= 50 ? 'bg-blue-500' : numStrength >= 30 ? 'bg-yellow-500' : 'bg-gray-400';
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${numStrength}%` }} />
      </div>
      <span className="text-xs font-semibold text-gray-600">{Math.round(numStrength)}</span>
    </div>
  );
};

// Signal Card
const SignalCard = ({ signal, onClick }) => {
  const displayPrice = signal.live_price || signal.price;
  const hasLiveData = !!signal.live_price;

  return (
    <div onClick={() => onClick(signal)} className={`bg-white rounded-lg border-l-4 ${signal.is_strong ? 'border-emerald-500' : 'border-blue-500'} shadow-sm p-4 hover:shadow-md transition-all cursor-pointer group`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-gray-900">{signal.symbol}</span>
          {signal.is_strong && <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 text-xs font-semibold rounded-full flex items-center gap-1"><Zap size={12} /> STRONG</span>}
        </div>
        <div className="flex items-center gap-2">
          <div className="text-right">
            <span className="text-lg font-semibold text-gray-900">${displayPrice?.toFixed(2)}</span>
            {hasLiveData && signal.live_change_pct !== undefined && (
              <span className={`ml-2 text-sm ${signal.live_change_pct >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                {signal.live_change_pct >= 0 ? '+' : ''}{signal.live_change_pct?.toFixed(2)}%
              </span>
            )}
          </div>
          <ChevronRight size={18} className="text-gray-400 group-hover:text-blue-600 transition-colors" />
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 text-sm">
        <div className="flex items-center gap-1">
          <TrendingUp size={14} className="text-emerald-500" />
          <span className="text-gray-500">DWAP:</span>
          <span className="font-medium text-emerald-600">+{signal.pct_above_dwap}%</span>
        </div>
        <div className="flex items-center gap-1">
          <Activity size={14} className="text-blue-500" />
          <span className="text-gray-500">Vol:</span>
          <span className="font-medium">{signal.volume_ratio}x</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-gray-500">Str:</span>
          <SignalStrengthBar strength={signal.signal_strength || 0} />
        </div>
      </div>
      {signal.recommendation && (
        <div className="mt-2 text-xs text-gray-500 italic truncate">{signal.recommendation}</div>
      )}
      {hasLiveData && (
        <div className="mt-1 text-xs text-blue-500 flex items-center gap-1">
          <Activity size={10} className="animate-pulse" /> Live
        </div>
      )}
    </div>
  );
};

// Position Row
const PositionRow = ({ position, onClick }) => {
  const pnlColor = position.pnl_pct >= 0 ? 'text-emerald-600' : 'text-red-500';
  const pnlBg = position.pnl_pct >= 0 ? 'bg-emerald-50' : 'bg-red-50';
  const hasLiveData = position.live_change !== undefined;
  const dayChangeColor = (position.live_change_pct || 0) >= 0 ? 'text-emerald-600' : 'text-red-500';

  return (
    <tr onClick={() => onClick(position)} className="hover:bg-blue-50 transition-colors cursor-pointer group">
      <td className="py-3 px-4">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-gray-900">{position.symbol}</span>
          {hasLiveData && <Activity size={10} className="text-blue-500 animate-pulse" />}
          <Eye size={14} className="text-gray-300 group-hover:text-blue-500" />
        </div>
      </td>
      <td className="py-3 px-4 text-gray-600">{position.shares?.toFixed(2)}</td>
      <td className="py-3 px-4 text-gray-600">${position.entry_price?.toFixed(2)}</td>
      <td className="py-3 px-4">
        <div className="flex flex-col">
          <span className="font-medium text-gray-900">${position.current_price?.toFixed(2)}</span>
          {hasLiveData && (
            <span className={`text-xs ${dayChangeColor}`}>
              {position.live_change_pct >= 0 ? '+' : ''}{position.live_change_pct?.toFixed(2)}% today
            </span>
          )}
        </div>
      </td>
      <td className="py-3 px-4"><span className={`inline-flex items-center gap-1 px-2 py-1 rounded-md font-semibold text-sm ${pnlBg} ${pnlColor}`}>{position.pnl_pct >= 0 ? '+' : ''}{position.pnl_pct?.toFixed(1)}%</span></td>
      <td className="py-3 px-4 text-gray-500"><Clock size={14} className="inline mr-1" />{position.days_held}d</td>
    </tr>
  );
};

// ============================================================================
// Main Dashboard
// ============================================================================

function Dashboard() {
  const { user, logout, isAdmin, isAuthenticated, loading: authLoading } = useAuth();
  const [signals, setSignals] = useState([]);
  const [positions, setPositions] = useState([]);
  const [trades, setTrades] = useState([]);
  const [missedOpportunities, setMissedOpportunities] = useState([]);
  const [backtest, setBacktest] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastScan, setLastScan] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showEnsembleView, setShowEnsembleView] = useState(true); // true = ensemble (double), false = separate lists
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [chartModal, setChartModal] = useState(null);
  const [dataStatus, setDataStatus] = useState({ loaded: 0, status: 'loading' });
  const [marketRegime, setMarketRegime] = useState(null);
  const [liveQuotes, setLiveQuotes] = useState({});
  const [quotesLastUpdate, setQuotesLastUpdate] = useState(null);

  // Live quotes polling - updates prices every 30 seconds during market hours
  useEffect(() => {
    const fetchLiveQuotes = async () => {
      // Get symbols from positions and signals
      const positionSymbols = positions.map(p => p.symbol);
      const signalSymbols = signals.slice(0, 10).map(s => s.symbol); // Top 10 signals
      const allSymbols = [...new Set([...positionSymbols, ...signalSymbols])];

      if (allSymbols.length === 0) return;

      try {
        const response = await api.get(`/api/quotes/live?symbols=${allSymbols.join(',')}`);
        if (response.quotes) {
          setLiveQuotes(response.quotes);
          setQuotesLastUpdate(new Date(response.timestamp));
        }
      } catch (err) {
        console.log('Live quotes fetch failed:', err);
      }
    };

    // Initial fetch
    if (positions.length > 0 || signals.length > 0) {
      fetchLiveQuotes();
    }

    // Poll every 30 seconds
    const interval = setInterval(fetchLiveQuotes, 30000);

    return () => clearInterval(interval);
  }, [positions.length, signals.length]); // Re-run when positions or signals change

  // Merge live quotes into positions for display
  const positionsWithLiveQuotes = positions.map(p => {
    const quote = liveQuotes[p.symbol];
    if (quote) {
      const livePrice = quote.price;
      const pnlPct = ((livePrice - p.entry_price) / p.entry_price) * 100;
      const pnlDollars = (livePrice - p.entry_price) * p.shares;
      return {
        ...p,
        current_price: livePrice,
        pnl_pct: pnlPct,
        pnl_dollars: pnlDollars,
        live_change: quote.change,
        live_change_pct: quote.change_pct,
      };
    }
    return p;
  });

  // Merge live quotes into signals for display
  const signalsWithLiveQuotes = signals.map(s => {
    const quote = liveQuotes[s.symbol];
    if (quote) {
      return {
        ...s,
        live_price: quote.price,
        live_change: quote.change,
        live_change_pct: quote.change_pct,
      };
    }
    return s;
  });

  // Initial data load - HYBRID APPROACH for instant dashboard display
  // 1. Show cached data immediately (no loading state for returning users)
  // 2. Fetch signals from CDN (same for all users, instant)
  // 3. Background refresh user-specific data from API
  useEffect(() => {
    const loadData = async () => {
      // Step 1: Load cached data IMMEDIATELY (no loading spinner for returning users)
      const cachedSignals = getCache(CACHE_KEYS.SIGNALS);
      const cachedBacktest = getCache(CACHE_KEYS.BACKTEST);
      const cachedPositions = getCache(CACHE_KEYS.POSITIONS);
      const cachedMissed = getCache(CACHE_KEYS.MISSED);

      // If we have any cached data, show the dashboard immediately
      if (cachedSignals || cachedBacktest) {
        if (cachedSignals) setSignals(cachedSignals);
        if (cachedBacktest) {
          // Check if cached data is walk-forward format or simple backtest
          if (cachedBacktest.available !== undefined) {
            // Walk-forward cached format
            const wf = cachedBacktest;
            setBacktest({
              total_return_pct: wf.total_return_pct?.toFixed(1) || '0.0',
              sharpe_ratio: wf.sharpe_ratio?.toFixed(2) || '0.00',
              max_drawdown_pct: Math.abs(wf.max_drawdown_pct || 0).toFixed(1),
              win_rate: '--',
              start_date: wf.start_date?.split('T')[0],
              end_date: wf.end_date?.split('T')[0],
              strategy: 'momentum',
              benchmark_return_pct: wf.benchmark_return_pct?.toFixed(1) || '0.0',
              num_strategy_switches: wf.num_strategy_switches || 0,
              is_walk_forward: true
            });
          } else if (cachedBacktest.backtest) {
            // Simple backtest format
            setBacktest({ ...cachedBacktest.backtest, strategy: cachedBacktest.strategy || 'momentum', is_walk_forward: false });
          }
          // Don't load positions/trades from backtest cache - only from user data
        }
        // Load user positions from cache (NOT backtest positions)
        if (cachedPositions) setPositions(cachedPositions);
        // Missed opportunities removed - was simulated data
        setLoading(false); // Dashboard visible immediately!
      }

      // Step 2: Fetch signals from CDN (static JSON, instant for ALL users)
      try {
        const cdnSignals = await fetchSignalsFromCDN();
        if (cdnSignals && cdnSignals.length > 0) {
          setSignals(cdnSignals);
          setCache(CACHE_KEYS.SIGNALS, cdnSignals);
          setLastScan(new Date()); // CDN signals are fresh
        }
      } catch (e) {
        console.log('CDN signals not available, will use API fallback');
      }

      // Step 3: Quick health check to show data status
      try {
        const health = await api.get('/health');
        setDataStatus({ loaded: health.symbols_loaded, status: 'ready' });
        if (health.last_scan) {
          // Ensure timestamp is parsed as UTC (backend returns UTC without Z suffix)
          const ts = health.last_scan.endsWith('Z') ? health.last_scan : health.last_scan + 'Z';
          setLastScan(new Date(ts));
        }
        setLoading(false); // Definitely show dashboard now
      } catch (err) {
        // If health check fails but we have cached data, still show dashboard
        if (cachedSignals || cachedBacktest) {
          setDataStatus({ loaded: 0, status: 'cached' });
          setLoading(false);
        } else {
          setError('Failed to connect to backend. Make sure the API is running.');
          setLoading(false);
          return;
        }
      }

      // Step 4: Background refresh - load fresh data from API (don't block UI)
      const refreshData = async () => {
        try {
          // Load all data in parallel - try cached walk-forward first, fallback to simple backtest
          const [walkForwardResult, signalsResult, marketResult, userPositionsResult, userTradesResult, missedResult] = await Promise.allSettled([
            api.get('/api/backtest/walk-forward-cached').catch(() => null),
            // Only fetch from API if CDN failed
            signals.length === 0 ? api.get('/api/signals/memory-scan?refresh=false&apply_market_filter=true').catch(() => null) : Promise.resolve(null),
            api.get('/api/market/regime').catch(() => null),
            api.get('/api/portfolio/positions').catch(() => null),
            api.get('/api/portfolio/trades?limit=50').catch(() => null),
            api.get('/api/signals/missed?days=90&limit=5').catch(() => null)
          ]);

          // Process walk-forward or fallback to simple backtest (for stats display only, NOT for positions/trades)
          if (walkForwardResult.status === 'fulfilled' && walkForwardResult.value?.available) {
            // Use cached walk-forward results (more accurate)
            const wf = walkForwardResult.value;
            setBacktest({
              total_return_pct: wf.total_return_pct?.toFixed(1) || '0.0',
              sharpe_ratio: wf.sharpe_ratio?.toFixed(2) || '0.00',
              max_drawdown_pct: Math.abs(wf.max_drawdown_pct || 0).toFixed(1),
              win_rate: '--',  // Walk-forward doesn't track win rate
              start_date: wf.start_date?.split('T')[0],
              end_date: wf.end_date?.split('T')[0],
              strategy: 'momentum',
              benchmark_return_pct: wf.benchmark_return_pct?.toFixed(1) || '0.0',
              num_strategy_switches: wf.num_strategy_switches || 0,
              is_walk_forward: true
            });
            setCache(CACHE_KEYS.BACKTEST, walkForwardResult.value);
          } else {
            // Fallback to simple backtest
            try {
              const simpleBacktest = await api.get('/api/backtest/run?days=252');
              if (simpleBacktest?.success) {
                setBacktest({ ...simpleBacktest.backtest, strategy: simpleBacktest.strategy || 'momentum', is_walk_forward: false });
                setCache(CACHE_KEYS.BACKTEST, simpleBacktest);
              }
            } catch (e) {
              console.log('Simple backtest fallback failed:', e);
            }
          }

          // Process user positions ONLY - no backtest fallback
          let userPositions = [];
          if (userPositionsResult.status === 'fulfilled' && userPositionsResult.value?.positions) {
            userPositions = userPositionsResult.value.positions;
            setPositions(userPositions);
            setCache(CACHE_KEYS.POSITIONS, userPositions);
          } else {
            setPositions([]);
          }

          // Process user trades ONLY - no backtest fallback
          if (userTradesResult.status === 'fulfilled' && userTradesResult.value?.trades) {
            setTrades(userTradesResult.value.trades);
          } else {
            setTrades([]);
          }

          // Process signals result (only if CDN didn't work)
          // Filter out signals for stocks user already has positions in
          if (signalsResult.status === 'fulfilled' && signalsResult.value?.signals) {
            const positionSymbols = new Set(userPositions.map(p => p.symbol));
            const filteredSignals = signalsResult.value.signals.filter(s => !positionSymbols.has(s.symbol));
            setSignals(filteredSignals);
            setCache(CACHE_KEYS.SIGNALS, filteredSignals);
            if (signalsResult.value.timestamp) {
              setLastScan(new Date(signalsResult.value.timestamp));
            }
          }

          // Process market regime
          if (marketResult.status === 'fulfilled' && marketResult.value) {
            setMarketRegime(marketResult.value);
          }

          // Process missed opportunities - signals that hit +20% profit target
          if (missedResult.status === 'fulfilled' && Array.isArray(missedResult.value)) {
            setMissedOpportunities(missedResult.value);
            setCache(CACHE_KEYS.MISSED, missedResult.value);
          }
        } catch (err) {
          console.log('Background refresh failed:', err);
        }
      };

      // Run background refresh
      refreshData();
    };

    loadData();
  }, []);

  const runScan = async () => {
    setScanning(true);
    try {
      // Refresh signals from API (memory-scan exports to CDN automatically)
      const signalsResult = await api.get('/api/signals/memory-scan?refresh=true&apply_market_filter=true&export_to_cdn=true');

      // Filter out signals for stocks user already has positions in
      const positionSymbols = new Set(positions.map(p => p.symbol));
      const filteredSignals = (signalsResult.signals || []).filter(s => !positionSymbols.has(s.symbol));
      setSignals(filteredSignals);
      setCache(CACHE_KEYS.SIGNALS, filteredSignals);
      setLastScan(new Date(signalsResult.timestamp));

      // Update market regime
      const marketResult = await api.get('/api/market/regime');
      setMarketRegime(marketResult);

      // Re-run backtest for stats only (NOT for positions/trades)
      // Note: We use simple backtest here for speed; daily walk-forward runs in background
      try {
        const backtestResult = await api.get('/api/backtest/run?days=252');
        setBacktest({ ...backtestResult.backtest, strategy: backtestResult.strategy || 'momentum', is_walk_forward: false });
        setCache(CACHE_KEYS.BACKTEST, backtestResult);
        // Don't set positions/trades from backtest - use real user data only
      } catch (btErr) {
        console.log('Backtest failed - data may still be loading');
      }

    } catch (err) {
      console.error('Scan failed:', err);
    } finally {
      setScanning(false);
    }
  };

  // Reload positions after a buy/sell - user data only, no backtest fallback
  const reloadPositions = async () => {
    try {
      // Get real positions from database only
      const posResult = await api.get('/api/portfolio/positions');
      const userPositions = posResult.positions || [];
      setPositions(userPositions);
      setCache(CACHE_KEYS.POSITIONS, userPositions);

      // Also reload trades
      const tradesResult = await api.get('/api/portfolio/trades?limit=50');
      if (tradesResult.trades) {
        setTrades(tradesResult.trades);
      }

      // Update signals to exclude stocks user now has positions in
      const positionSymbols = new Set(userPositions.map(p => p.symbol));
      setSignals(prev => prev.filter(s => !positionSymbols.has(s.symbol)));
    } catch (err) {
      console.log('Could not reload positions:', err);
    }
  };

  // Use live-quoted positions for calculations
  const totalValue = positionsWithLiveQuotes.reduce((sum, p) => sum + (p.shares || 0) * (p.current_price || 0), 0);
  const totalCost = positionsWithLiveQuotes.reduce((sum, p) => sum + (p.shares || 0) * (p.entry_price || 0), 0);
  const totalPnlPct = totalCost > 0 ? ((totalValue - totalCost) / totalCost) * 100 : 0;
  const wins = trades.filter(t => t.pnl > 0);
  const winRate = trades.length > 0 ? (wins.length / trades.length * 100) : 0;
  const totalHistoricalPnl = trades.reduce((sum, t) => sum + (t.pnl || 0), 0);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Loading RigaCap</h2>
          <p className="text-gray-500">Initializing your dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Connection Error</h2>
          <p className="text-gray-500 mb-4">{error}</p>
          <p className="text-sm text-gray-400 mb-4">
            Backend: {API_BASE}
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Show data loading state if backend is up but no data yet
  const noDataAvailable = positions.length === 0 && signals.length === 0 && trades.length === 0;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
              <TrendingUp className="text-white" size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">RigaCap</h1>
              <p className="text-xs text-gray-500">DWAP Trading System</p>
            </div>
          </div>

          <nav className="flex items-center gap-1 bg-gray-100 p-1 rounded-xl">
            <button onClick={() => setActiveTab('dashboard')} className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'dashboard' ? 'bg-white shadow text-gray-900' : 'text-gray-600 hover:text-gray-900'}`}>
              <BarChart3 size={16} className="inline mr-2" />Dashboard
            </button>
            <button onClick={() => setActiveTab('momentum')} className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'momentum' ? 'bg-white shadow text-gray-900' : 'text-gray-600 hover:text-gray-900'}`}>
              <TrendingUp size={16} className="inline mr-2" />Momentum
            </button>
            <button onClick={() => setActiveTab('history')} className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'history' ? 'bg-white shadow text-gray-900' : 'text-gray-600 hover:text-gray-900'}`}>
              <History size={16} className="inline mr-2" />Trade History
            </button>
            {isAdmin && (
              <button onClick={() => setActiveTab('admin')} className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'admin' ? 'bg-white shadow text-gray-900' : 'text-gray-600 hover:text-gray-900'}`}>
                <Settings size={16} className="inline mr-2" />Admin
              </button>
            )}
          </nav>

          <div className="flex items-center gap-4">
            <div className="text-right text-sm">
              <span className="text-gray-500">Last scan: </span>
              <span className="text-gray-700 font-medium">{lastScan ? lastScan.toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' }) : 'Never'}</span>
              <div className="text-xs text-gray-400">{dataStatus.loaded} symbols loaded</div>
            </div>
            <button
              onClick={runScan}
              disabled={scanning}
              className={`px-4 py-2 rounded-lg font-medium text-sm flex items-center gap-2 ${
                scanning ? 'bg-gray-100 text-gray-400' : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-500/20'
              }`}
            >
              <RefreshCw size={16} className={scanning ? 'animate-spin' : ''} />
              {scanning ? 'Scanning...' : 'Scan'}
            </button>
            {user ? (
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white font-medium">
                  {(user.name || user.email || 'U')[0].toUpperCase()}
                </div>
                <button onClick={logout} className="p-2 text-gray-400 hover:text-gray-600"><LogOut size={18} /></button>
              </div>
            ) : (
              <button onClick={() => setShowLoginModal(true)} className="px-4 py-2 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800 flex items-center gap-2">
                <LogIn size={16} />Sign In
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6">
        {/* Subscription Banner */}
        {isAuthenticated && <SubscriptionBanner />}

        {/* Admin Dashboard */}
        {activeTab === 'admin' && isAdmin && (
          <AdminDashboard />
        )}

        {/* No data warning banner */}
        {noDataAvailable && (
          <div className="mb-4 bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-center gap-3">
            <AlertCircle className="text-amber-500 flex-shrink-0" size={24} />
            <div>
              <h3 className="font-semibold text-amber-800">Market Data Loading</h3>
              <p className="text-sm text-amber-700">
                Historical data is being fetched. This may take a moment.
                Click "Scan" to retry, or wait for the automatic refresh.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'dashboard' ? (
          <>
            {/* Backtest/Walk-Forward summary banner */}
            {backtest && (
              <div className={`mb-4 bg-gradient-to-r ${backtest.is_walk_forward ? 'from-purple-50 to-indigo-50 border-purple-200' : 'from-blue-50 to-indigo-50 border-blue-200'} border rounded-xl p-4`}>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-900">
                      Simulated Portfolio {backtest.is_walk_forward ? '(Walk-Forward)' : '(Backtest)'}
                    </h3>
                    <p className="text-sm text-gray-500">
                      {backtest.is_walk_forward
                        ? `Adaptive strategy with ${backtest.num_strategy_switches || 0} switches • ${backtest.start_date} to ${backtest.end_date}`
                        : `Based on ${backtest.strategy === 'momentum' ? 'Momentum' : 'DWAP'} strategy from ${backtest.start_date} to ${backtest.end_date}`
                      }
                    </p>
                  </div>
                  <div className="flex gap-6 text-sm">
                    <div className="text-center">
                      <p className="text-gray-500">Return</p>
                      <p className={`font-bold ${parseFloat(backtest.total_return_pct) >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                        {parseFloat(backtest.total_return_pct) >= 0 ? '+' : ''}{backtest.total_return_pct}%
                      </p>
                    </div>
                    {backtest.is_walk_forward ? (
                      <div className="text-center">
                        <p className="text-gray-500">vs SPY</p>
                        <p className={`font-bold ${parseFloat(backtest.total_return_pct) > parseFloat(backtest.benchmark_return_pct) ? 'text-emerald-600' : 'text-gray-900'}`}>
                          {parseFloat(backtest.total_return_pct) > parseFloat(backtest.benchmark_return_pct) ? '+' : ''}{(parseFloat(backtest.total_return_pct) - parseFloat(backtest.benchmark_return_pct)).toFixed(1)}%
                        </p>
                      </div>
                    ) : (
                      <div className="text-center">
                        <p className="text-gray-500">Win Rate</p>
                        <p className="font-bold text-gray-900">{backtest.win_rate}%</p>
                      </div>
                    )}
                    <div className="text-center">
                      <p className="text-gray-500">Sharpe</p>
                      <p className="font-bold text-gray-900">{backtest.sharpe_ratio}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-500">Max DD</p>
                      <p className="font-bold text-red-500">-{backtest.max_drawdown_pct}%</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Market Regime Banner */}
            {marketRegime && (
              <div className={`mb-4 p-3 rounded-lg flex items-center justify-between ${
                marketRegime.regime === 'strong_bull' ? 'bg-emerald-50 border border-emerald-200' :
                marketRegime.regime === 'weak_bull' ? 'bg-green-50 border border-green-200' :
                marketRegime.regime === 'rotating_bull' ? 'bg-violet-50 border border-violet-200' :
                marketRegime.regime === 'range_bound' ? 'bg-amber-50 border border-amber-200' :
                marketRegime.regime === 'recovery' ? 'bg-cyan-50 border border-cyan-200' :
                marketRegime.regime === 'weak_bear' ? 'bg-orange-50 border border-orange-200' :
                marketRegime.regime === 'panic_crash' ? 'bg-red-50 border border-red-200' :
                'bg-gray-50 border border-gray-200'
              }`}>
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-full ${
                    marketRegime.regime === 'strong_bull' ? 'bg-emerald-100' :
                    marketRegime.regime === 'weak_bull' ? 'bg-green-100' :
                    marketRegime.regime === 'rotating_bull' ? 'bg-violet-100' :
                    marketRegime.regime === 'range_bound' ? 'bg-amber-100' :
                    marketRegime.regime === 'recovery' ? 'bg-cyan-100' :
                    marketRegime.regime === 'weak_bear' ? 'bg-orange-100' :
                    'bg-red-100'
                  }`}>
                    {marketRegime.regime === 'strong_bull' ? <TrendingUp className="w-5 h-5 text-emerald-600" /> :
                     marketRegime.regime === 'weak_bull' ? <TrendingUp className="w-5 h-5 text-green-600" /> :
                     marketRegime.regime === 'rotating_bull' ? <RefreshCw className="w-5 h-5 text-violet-600" /> :
                     marketRegime.regime === 'range_bound' ? <Activity className="w-5 h-5 text-amber-600" /> :
                     marketRegime.regime === 'recovery' ? <TrendingUp className="w-5 h-5 text-cyan-600" /> :
                     marketRegime.regime === 'weak_bear' ? <TrendingDown className="w-5 h-5 text-orange-600" /> :
                     <TrendingDown className="w-5 h-5 text-red-600" />}
                  </div>
                  <div>
                    <span className="font-semibold text-gray-900 capitalize">
                      {marketRegime.regime_name || marketRegime.regime?.replace('_', ' ')} Market
                    </span>
                    <span className="mx-2 text-gray-400">|</span>
                    <span className="text-gray-600">SPY ${marketRegime.spy_price?.toFixed(2)}</span>
                    <span className="mx-2 text-gray-400">|</span>
                    <span className="text-gray-600">VIX {marketRegime.vix_level?.toFixed(1)}</span>
                    {marketRegime.risk_level && (
                      <>
                        <span className="mx-2 text-gray-400">|</span>
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          marketRegime.risk_level === 'low' ? 'bg-green-100 text-green-700' :
                          marketRegime.risk_level === 'extreme' ? 'bg-red-100 text-red-700' :
                          marketRegime.risk_level === 'high' ? 'bg-orange-100 text-orange-700' :
                          'bg-yellow-100 text-yellow-700'
                        }`}>
                          {marketRegime.risk_level} risk
                        </span>
                      </>
                    )}
                  </div>
                </div>
                <div className="text-sm text-gray-600 max-w-sm text-right leading-tight">
                  {marketRegime.recommendation}
                </div>
              </div>
            )}

            <div className="grid grid-cols-5 gap-4 mb-6">
              <MetricCard title="Portfolio Value" value={`$${totalValue.toLocaleString(undefined, {maximumFractionDigits: 0})}`} icon={Wallet} trend="up" />
              <MetricCard title="Open P&L" value={`${totalPnlPct >= 0 ? '+' : ''}${totalPnlPct.toFixed(1)}%`} icon={totalPnlPct >= 0 ? TrendingUp : TrendingDown} trend={totalPnlPct >= 0 ? 'up' : 'down'} />
              <MetricCard title="Positions" value={`${positions.length}/15`} icon={PieIcon} />
              <MetricCard title="Signals" value={signalsWithLiveQuotes.length} subtitle={`${signalsWithLiveQuotes.filter(s => s.is_strong).length} strong`} icon={Zap} />
              <MetricCard title="Win Rate" value={`${winRate.toFixed(0)}%`} subtitle={`${trades.length} trades`} icon={Target} />
            </div>

            <div className="grid grid-cols-3 gap-6">
              <div className="col-span-1">
                <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                  <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-gray-900">Live Signals</h2>
                    <span className="text-xs text-gray-500">Click to view chart</span>
                  </div>
                  <div className="p-4 space-y-3 max-h-[600px] overflow-y-auto">
                    {signalsWithLiveQuotes.length > 0 ? signalsWithLiveQuotes.map((s, i) => (
                      <SignalCard key={i} signal={s} onClick={(sig) => setChartModal({ type: 'signal', data: sig, symbol: sig.symbol })} />
                    )) : (
                      <div className="text-center py-8 text-gray-500">
                        <Activity className="w-12 h-12 mx-auto text-gray-300 mb-3" />
                        <p>No signals found</p>
                        <p className="text-xs mt-1">Run a scan to check for opportunities</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="col-span-2">
                <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                  <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <h2 className="text-lg font-semibold text-gray-900">Open Positions</h2>
                      {positions.length > 0 && positions[0]?.source === 'backtest' && (
                        <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">Simulated</span>
                      )}
                    </div>
                    <span className="text-xs text-gray-500">Click row to view chart with entry point</span>
                  </div>
                  {positions.length > 0 ? (
                    <table className="w-full">
                      <thead className="bg-gray-50 border-b border-gray-100">
                        <tr>{['Symbol', 'Shares', 'Entry', 'Current', 'P&L', 'Days'].map(h => <th key={h} className="py-3 px-4 text-left text-xs font-semibold text-gray-500 uppercase">{h}</th>)}</tr>
                      </thead>
                      <tbody>
                        {positionsWithLiveQuotes.map(p => <PositionRow key={p.id} position={p} onClick={(pos) => setChartModal({ type: 'position', data: pos, symbol: pos.symbol })} />)}
                      </tbody>
                    </table>
                  ) : (
                    <div className="text-center py-12 text-gray-500">
                      <PieIcon className="w-12 h-12 mx-auto text-gray-300 mb-3" />
                      <p>No open positions</p>
                      <p className="text-xs mt-1">Positions will appear when signals meet buy criteria</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Missed Opportunities Section */}
            {missedOpportunities.length > 0 && (
              <div className="mt-6 bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-semibold text-gray-900">Missed Opportunities</h2>
                    <span className="px-2 py-0.5 bg-amber-100 text-amber-700 text-xs font-medium rounded-full">Hit +20% Target</span>
                  </div>
                  <span className="text-xs text-gray-500">Completed winning trades you could have made</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50 border-b border-gray-100">
                      <tr>
                        {['Symbol', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Return', 'P&L', 'Days'].map(h => (
                          <th key={h} className="py-3 px-4 text-left text-xs font-semibold text-gray-500 uppercase">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {missedOpportunities.map((opp, idx) => (
                        <tr
                          key={`${opp.symbol}-${idx}`}
                          className="hover:bg-amber-50 border-b border-gray-50 cursor-pointer"
                          onClick={() => setChartModal({
                            type: 'missed',
                            data: {
                              ...opp,
                              entry_date: opp.signal_date,
                              entry_price: opp.signal_price,
                              sell_date: opp.exit_date,
                              sell_price: opp.exit_price
                            },
                            symbol: opp.symbol
                          })}
                        >
                          <td className="py-3 px-4 font-semibold text-gray-900">{opp.symbol}</td>
                          <td className="py-3 px-4 text-gray-500 text-sm">{opp.signal_date}</td>
                          <td className="py-3 px-4">${opp.signal_price?.toFixed(2)}</td>
                          <td className="py-3 px-4 text-gray-500 text-sm">{opp.exit_date}</td>
                          <td className="py-3 px-4">${opp.exit_price?.toFixed(2)}</td>
                          <td className="py-3 px-4">
                            <span className="px-2 py-1 rounded text-sm font-semibold bg-green-50 text-green-600">
                              +{opp.would_be_return?.toFixed(0)}%
                            </span>
                          </td>
                          <td className="py-3 px-4 font-medium text-green-600">
                            +${opp.would_be_pnl?.toFixed(0)}
                          </td>
                          <td className="py-3 px-4 text-sm text-gray-500">{opp.days_held || '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="px-5 py-3 bg-green-50 border-t border-green-100 text-sm text-green-700">
                  <span className="font-medium">Total missed gains:</span> $
                  {missedOpportunities.reduce((sum, o) => sum + (o.would_be_pnl || 0), 0).toLocaleString(undefined, {maximumFractionDigits: 0})}
                  {' '}(based on $6k position size)
                </div>
              </div>
            )}
          </>
        ) : activeTab === 'momentum' ? (
          <div className="space-y-6">
            {/* View Toggle */}
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-600">View:</span>
                <button
                  onClick={() => setShowEnsembleView(true)}
                  className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    showEnsembleView ? 'bg-yellow-100 text-yellow-800 font-medium' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  Ensemble Signals
                </button>
                <button
                  onClick={() => setShowEnsembleView(false)}
                  className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    !showEnsembleView ? 'bg-blue-100 text-blue-800 font-medium' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  Separate Lists
                </button>
              </div>
            </div>

            {showEnsembleView ? (
              /* Ensemble View - Double Signals Only */
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <DoubleSignals onSymbolClick={(symbol) => setChartModal({ type: 'signal', data: { symbol }, symbol })} />
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-yellow-500" />
                    Why Ensemble Signals?
                  </h3>
                  <div className="space-y-3 text-sm text-gray-600">
                    <div className="p-3 bg-yellow-50 rounded-lg text-yellow-800">
                      <strong>2.5x Higher Returns:</strong> Stocks with BOTH DWAP trigger AND top momentum ranking
                      averaged +2.91% over 20 days vs +1.16% for DWAP-only signals.
                    </div>
                    <p>
                      <strong>Entry Criteria:</strong>
                    </p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>Price crossed +5% above 200-day DWAP (buy trigger)</li>
                      <li>Stock ranks in top 20 by momentum score</li>
                      <li>Volume {">"} 500,000 and Price {">"} $20</li>
                    </ul>
                    <p>
                      <strong>Buy Point:</strong> When the DWAP +5% crossover occurs AND the stock is already
                      showing momentum strength. These are high-conviction entries.
                    </p>
                    <p className="text-gray-500 text-xs mt-4">
                      Switch to "Separate Lists" to see DWAP signals and momentum rankings independently.
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              /* Separate Lists View */
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <MomentumRankings onSymbolClick={(symbol) => setChartModal({ type: 'signal', data: { symbol }, symbol })} />
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-blue-600" />
                    How Momentum Ranking Works
                  </h3>
                  <div className="space-y-3 text-sm text-gray-600">
                    <p>
                      <strong>Score Formula:</strong> short_momentum × 0.5 + long_momentum × 0.3 - volatility × 0.2
                    </p>
                    <p>
                      <strong>Quality Filters:</strong>
                    </p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>Price above 20-day and 50-day moving averages</li>
                      <li>Within 5% of 50-day high (breakout potential)</li>
                      <li>Volume {">"} 500,000</li>
                      <li>Price {">"} $20</li>
                    </ul>
                    <p>
                      <strong>Trading Strategy:</strong> The momentum strategy buys the top-ranked stocks each week (Friday rebalance).
                      Higher scores indicate stronger momentum with controlled volatility.
                    </p>
                    <div className="mt-4 p-3 bg-blue-50 rounded-lg text-blue-800">
                      <strong>Tip:</strong> Switch to "Ensemble Signals" to see only stocks that appear in BOTH lists.
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            <div className="grid grid-cols-4 gap-4">
              <MetricCard title="Total Trades" value={trades.length} icon={History} />
              <MetricCard title="Win Rate" value={`${winRate.toFixed(0)}%`} subtitle={`${wins.length}W / ${trades.length - wins.length}L`} icon={Target} trend={winRate > 50 ? 'up' : 'down'} />
              <MetricCard title="Total P&L" value={`$${totalHistoricalPnl.toLocaleString(undefined, {maximumFractionDigits: 0})}`} icon={Wallet} trend={totalHistoricalPnl >= 0 ? 'up' : 'down'} />
              <MetricCard title="Avg Return" value={`${trades.length ? (trades.reduce((s,t) => s + (t.pnl_pct || 0), 0) / trades.length).toFixed(1) : 0}%`} icon={BarChart3} />
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
              <div className="px-5 py-4 border-b border-gray-100"><h2 className="text-lg font-semibold text-gray-900">Trade History (1 Year Backtest)</h2></div>
              <div className="overflow-x-auto max-h-[600px]">
                {trades.length > 0 ? (
                  <table className="w-full">
                    <thead className="bg-gray-50 border-b border-gray-100 sticky top-0">
                      <tr>{['Symbol', 'Entry', 'Exit', 'Entry $', 'Exit $', 'Return', 'P&L', 'Reason', 'Days'].map(h => <th key={h} className="py-3 px-4 text-left text-xs font-semibold text-gray-500 uppercase">{h}</th>)}</tr>
                    </thead>
                    <tbody>
                      {trades.map(t => (
                        <tr key={t.id} className="hover:bg-gray-50 border-b border-gray-50">
                          <td className="py-3 px-4 font-medium">{t.symbol}</td>
                          <td className="py-3 px-4 text-gray-500 text-sm">{t.entry_date}</td>
                          <td className="py-3 px-4 text-gray-500 text-sm">{t.exit_date}</td>
                          <td className="py-3 px-4">${t.entry_price?.toFixed(2)}</td>
                          <td className="py-3 px-4">${t.exit_price?.toFixed(2)}</td>
                          <td className="py-3 px-4"><span className={`px-2 py-1 rounded text-sm font-semibold ${t.pnl_pct >= 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-red-50 text-red-500'}`}>{t.pnl_pct >= 0 ? '+' : ''}{t.pnl_pct?.toFixed(1)}%</span></td>
                          <td className={`py-3 px-4 font-medium ${t.pnl >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>${t.pnl?.toFixed(0)}</td>
                          <td className="py-3 px-4"><span className={`px-2 py-1 rounded text-xs font-medium ${t.exit_reason === 'profit_target' ? 'bg-emerald-100 text-emerald-700' : t.exit_reason === 'stop_loss' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'}`}>{t.exit_reason?.toUpperCase()}</span></td>
                          <td className="py-3 px-4 text-gray-500">{t.days_held}d</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="text-center py-12 text-gray-500">
                    <History className="w-12 h-12 mx-auto text-gray-300 mb-3" />
                    <p>No trades in backtest period</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      {showLoginModal && <LoginModal isOpen={showLoginModal} onClose={() => setShowLoginModal(false)} />}
      {chartModal && <StockChartModal {...chartModal} liveQuote={liveQuotes[chartModal.symbol]} onClose={() => setChartModal(null)} onAction={() => { setChartModal(null); reloadPositions(); }} />}
    </div>
  );
}

// Protected Route wrapper
function ProtectedRoute({ children }) {
  const { isAuthenticated, loading, user } = useAuth();

  console.log('ProtectedRoute: loading=', loading, 'isAuthenticated=', isAuthenticated, 'user=', user?.email);

  if (loading) {
    console.log('ProtectedRoute: Still loading, showing spinner');
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
      </div>
    );
  }

  if (!isAuthenticated) {
    console.log('ProtectedRoute: Not authenticated, redirecting to /');
    return <Navigate to="/" replace />;
  }

  console.log('ProtectedRoute: Authenticated, rendering children');
  return children;
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app" element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        } />
        <Route path="/dashboard" element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        } />
        <Route path="/admin" element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        } />
      </Routes>
    </AuthProvider>
  );
}
