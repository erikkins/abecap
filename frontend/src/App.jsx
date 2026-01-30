import React, { useState, useEffect, useCallback } from 'react';
import { 
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, ComposedChart, Bar, ReferenceLine, Legend
} from 'recharts';
import { 
  TrendingUp, TrendingDown, RefreshCw, Settings, Bell, User, LogOut,
  DollarSign, Target, Shield, Activity, PieChart as PieIcon, History,
  ArrowUpRight, ArrowDownRight, Clock, Zap, X, ChevronRight, Eye,
  Calendar, BarChart3, Wallet, LogIn
} from 'lucide-react';

// ============================================================================
// Authentication Context
// ============================================================================

const AuthContext = React.createContext(null);
const useAuth = () => React.useContext(AuthContext);

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem('stocker_user');
    if (stored) setUser(JSON.parse(stored));
    setLoading(false);
  }, []);

  const login = async (provider) => {
    const mockUser = { id: '1', name: 'Demo User', email: 'demo@stocker.app', provider };
    setUser(mockUser);
    localStorage.setItem('stocker_user', JSON.stringify(mockUser));
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('stocker_user');
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

// ============================================================================
// Mock Data Generators (2 years of history)
// ============================================================================

const generatePriceHistory = (symbol, days = 504) => {
  const data = [];
  let price = 100 + Math.random() * 200;
  const volatility = 0.02 + Math.random() * 0.02;
  
  for (let i = days; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    const change = (Math.random() - 0.48) * volatility * price;
    price = Math.max(20, price + change);
    const volume = Math.floor(5000000 + Math.random() * 20000000);
    
    data.push({
      date: date.toISOString().split('T')[0],
      price: Math.round(price * 100) / 100,
      volume,
      high: Math.round((price * (1 + Math.random() * 0.02)) * 100) / 100,
      low: Math.round((price * (1 - Math.random() * 0.02)) * 100) / 100,
    });
  }
  
  // Calculate indicators
  for (let i = 0; i < data.length; i++) {
    if (i >= 49) {
      data[i].ma50 = Math.round(data.slice(i - 49, i + 1).reduce((s, d) => s + d.price, 0) / 50 * 100) / 100;
    }
    if (i >= 199) {
      data[i].ma200 = Math.round(data.slice(i - 199, i + 1).reduce((s, d) => s + d.price, 0) / 200 * 100) / 100;
      const slice = data.slice(i - 199, i + 1);
      const pv = slice.reduce((s, d) => s + d.price * d.volume, 0);
      const v = slice.reduce((s, d) => s + d.volume, 0);
      data[i].dwap = Math.round((pv / v) * 100) / 100;
    }
  }
  return data;
};

const generateHistoricalTrades = () => {
  const trades = [];
  const symbols = ['AAPL', 'NVDA', 'MSFT', 'AMD', 'GOOGL', 'META', 'TSLA', 'AMZN', 'AVGO', 'CRM'];
  
  for (let i = 0; i < 50; i++) {
    const entryDate = new Date();
    entryDate.setDate(entryDate.getDate() - Math.floor(Math.random() * 700) - 30);
    const holdDays = Math.floor(Math.random() * 60) + 5;
    const exitDate = new Date(entryDate);
    exitDate.setDate(exitDate.getDate() + holdDays);
    
    const entryPrice = 50 + Math.random() * 400;
    const pnlPct = (Math.random() - 0.4) * 40;
    const exitPrice = entryPrice * (1 + pnlPct / 100);
    const shares = Math.floor(1000 / entryPrice) * 10;
    
    trades.push({
      id: i + 1,
      symbol: symbols[Math.floor(Math.random() * symbols.length)],
      entryDate: entryDate.toISOString().split('T')[0],
      exitDate: exitDate.toISOString().split('T')[0],
      entryPrice: Math.round(entryPrice * 100) / 100,
      exitPrice: Math.round(exitPrice * 100) / 100,
      shares,
      pnl: Math.round((exitPrice - entryPrice) * shares * 100) / 100,
      pnlPct: Math.round(pnlPct * 100) / 100,
      exitReason: pnlPct > 15 ? 'PROFIT_TARGET' : pnlPct < -7 ? 'STOP_LOSS' : 'MANUAL',
      daysHeld: holdDays
    });
  }
  return trades.sort((a, b) => new Date(b.exitDate) - new Date(a.exitDate));
};

const mockSignals = [
  { symbol: 'NVDA', price: 512.30, pct_above_dwap: 8.2, volume_ratio: 2.1, is_strong: true, stop_loss: 471.32, profit_target: 614.76, volume: 45000000 },
  { symbol: 'AMD', price: 158.90, pct_above_dwap: 6.5, volume_ratio: 1.8, is_strong: true, stop_loss: 146.19, profit_target: 190.68, volume: 32000000 },
  { symbol: 'AVGO', price: 1245.50, pct_above_dwap: 5.8, volume_ratio: 1.4, is_strong: false, stop_loss: 1145.86, profit_target: 1494.60, volume: 8000000 },
  { symbol: 'MSFT', price: 385.40, pct_above_dwap: 5.3, volume_ratio: 1.2, is_strong: false, stop_loss: 354.57, profit_target: 462.48, volume: 22000000 },
  { symbol: 'CRM', price: 298.75, pct_above_dwap: 5.1, volume_ratio: 1.1, is_strong: false, stop_loss: 274.85, profit_target: 358.50, volume: 6000000 },
];

const mockPositions = [
  { id: 1, symbol: 'AAPL', shares: 55, entry_price: 182.50, current_price: 195.90, pnl_pct: 7.34, days_held: 18, entry_date: '2025-01-12', stop_loss: 167.90, profit_target: 219.00 },
  { id: 2, symbol: 'META', shares: 22, entry_price: 485.00, current_price: 468.20, pnl_pct: -3.46, days_held: 8, entry_date: '2025-01-22', stop_loss: 446.20, profit_target: 582.00 },
  { id: 3, symbol: 'GOOGL', shares: 40, entry_price: 141.20, current_price: 152.80, pnl_pct: 8.22, days_held: 25, entry_date: '2025-01-05', stop_loss: 129.90, profit_target: 169.44 },
];

// ============================================================================
// Components
// ============================================================================

// Login Modal
const LoginModal = ({ onClose }) => {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async (provider) => {
    setLoading(true);
    await login(provider);
    setLoading(false);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-8 relative">
        <button onClick={onClose} className="absolute top-4 right-4 text-gray-400 hover:text-gray-600">
          <X size={24} />
        </button>
        
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
            <TrendingUp className="text-white" size={32} />
          </div>
          <h2 className="text-2xl font-bold text-gray-900">Welcome to Stocker</h2>
          <p className="text-gray-500 mt-2">Sign in to track your portfolio</p>
        </div>

        <button
          onClick={() => handleLogin('google')}
          disabled={loading}
          className="w-full flex items-center justify-center gap-3 px-4 py-3 border border-gray-300 rounded-xl hover:bg-gray-50 transition-colors mb-3"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
          </svg>
          <span className="font-medium text-gray-700">Continue with Google</span>
        </button>

        <button
          onClick={() => handleLogin('apple')}
          disabled={loading}
          className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-black text-white rounded-xl hover:bg-gray-800 transition-colors mb-6"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.81-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M13 3.5c.73-.83 1.94-1.46 2.94-1.5.13 1.17-.34 2.35-1.04 3.19-.69.85-1.83 1.51-2.95 1.42-.15-1.15.41-2.35 1.05-3.11z"/>
          </svg>
          <span className="font-medium">Continue with Apple</span>
        </button>

        <div className="relative mb-6">
          <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-gray-200"></div></div>
          <div className="relative flex justify-center text-sm"><span className="px-4 bg-white text-gray-500">or</span></div>
        </div>

        <input
          type="email"
          placeholder="Email address"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent mb-4"
        />
        <button
          onClick={() => handleLogin('email')}
          disabled={loading}
          className="w-full bg-blue-600 text-white py-3 rounded-xl font-medium hover:bg-blue-700 transition-colors"
        >
          {loading ? 'Signing in...' : 'Continue with Email'}
        </button>
      </div>
    </div>
  );
};

// Stock Chart Modal
const StockChartModal = ({ symbol, type, data, onClose, onAction }) => {
  const [timeRange, setTimeRange] = useState('1Y');
  const [priceData, setPriceData] = useState([]);
  
  useEffect(() => {
    const history = generatePriceHistory(symbol);
    const days = { '1M': 30, '3M': 90, '6M': 180, '1Y': 252, '2Y': 504 }[timeRange] || 252;
    setPriceData(history.slice(-days));
  }, [symbol, timeRange]);

  const currentPrice = priceData[priceData.length - 1]?.price || data?.price || 0;
  const startPrice = priceData[0]?.price || currentPrice;
  const changePct = ((currentPrice - startPrice) / startPrice * 100).toFixed(1);
  const isPositive = changePct >= 0;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-bold text-gray-900">{symbol}</h2>
              {data?.is_strong && (
                <span className="px-2 py-1 bg-emerald-100 text-emerald-700 text-xs font-semibold rounded-full flex items-center gap-1">
                  <Zap size={12} /> STRONG SIGNAL
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-2xl font-semibold">${currentPrice.toFixed(2)}</span>
              <span className={`flex items-center text-sm font-medium ${isPositive ? 'text-emerald-600' : 'text-red-500'}`}>
                {isPositive ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
                {isPositive ? '+' : ''}{changePct}%
              </span>
            </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-gray-100 rounded-lg">
            <X size={24} className="text-gray-400" />
          </button>
        </div>

        {/* Time Range */}
        <div className="px-6 py-3 border-b border-gray-100 flex gap-2">
          {['1M', '3M', '6M', '1Y', '2Y'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                timeRange === range ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {range}
            </button>
          ))}
        </div>

        {/* Chart */}
        <div className="p-6">
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={priceData}>
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
                tickFormatter={(val) => new Date(val).toLocaleDateString('en-US', { month: 'short' })}
                interval={Math.floor(priceData.length / 8)}
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
                  return (
                    <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200 text-sm">
                      <p className="font-medium text-gray-900 mb-1">{new Date(label).toLocaleDateString()}</p>
                      <p className="text-blue-600">Price: ${d?.price?.toFixed(2)}</p>
                      {d?.dwap && <p className="text-purple-600">DWAP: ${d.dwap.toFixed(2)}</p>}
                      {d?.ma50 && <p className="text-orange-500">MA50: ${d.ma50.toFixed(2)}</p>}
                      <p className="text-gray-400">Vol: {(d?.volume / 1000000).toFixed(1)}M</p>
                    </div>
                  );
                }}
              />
              <Bar yAxisId="volume" dataKey="volume" fill="#E5E7EB" opacity={0.5} />
              <Line yAxisId="price" type="monotone" dataKey="dwap" stroke="#8B5CF6" strokeWidth={2} dot={false} name="DWAP" />
              <Line yAxisId="price" type="monotone" dataKey="ma50" stroke="#F97316" strokeWidth={1.5} dot={false} strokeDasharray="5 5" name="MA50" />
              <Area yAxisId="price" type="monotone" dataKey="price" stroke="#3B82F6" strokeWidth={2} fill="url(#priceGradient)" name="Price" />
              
              {/* Buy signal marker */}
              {type === 'signal' && data?.price && (
                <ReferenceLine yAxisId="price" y={data.price} stroke="#10B981" strokeWidth={2} strokeDasharray="8 4"
                  label={{ value: '▶ BUY SIGNAL', fill: '#10B981', fontWeight: 'bold', fontSize: 12, position: 'insideRight' }}
                />
              )}
              
              {/* Entry price for positions */}
              {type === 'position' && data?.entry_price && (
                <ReferenceLine yAxisId="price" y={data.entry_price} stroke="#3B82F6" strokeWidth={2} strokeDasharray="8 4"
                  label={{ value: `Entry $${data.entry_price}`, fill: '#3B82F6', fontWeight: 'bold', fontSize: 12, position: 'insideLeft' }}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Details */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-100">
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
            ) : (
              <>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Entry Price</p>
                  <p className="text-lg font-semibold">${data?.entry_price?.toFixed(2)}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Current P&L</p>
                  <p className={`text-lg font-semibold ${data?.pnl_pct >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                    {data?.pnl_pct >= 0 ? '+' : ''}{data?.pnl_pct?.toFixed(1)}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Shares</p>
                  <p className="text-lg font-semibold">{data?.shares}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">Days Held</p>
                  <p className="text-lg font-semibold">{data?.days_held}</p>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="px-6 py-4 border-t border-gray-100 flex justify-end gap-3">
          <button onClick={onClose} className="px-6 py-2.5 text-gray-600 hover:bg-gray-100 rounded-xl font-medium">
            Cancel
          </button>
          <button 
            onClick={() => onAction(data)}
            className={`px-6 py-2.5 rounded-xl font-medium flex items-center gap-2 shadow-lg transition-all ${
              type === 'signal' 
                ? 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-500/20'
                : 'bg-red-600 hover:bg-red-700 text-white shadow-red-500/20'
            }`}
          >
            {type === 'signal' ? <><DollarSign size={18} /> Buy {symbol}</> : <><TrendingDown size={18} /> Sell Position</>}
          </button>
        </div>
      </div>
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

// Signal Card
const SignalCard = ({ signal, onClick }) => (
  <div onClick={() => onClick(signal)} className={`bg-white rounded-lg border-l-4 ${signal.is_strong ? 'border-emerald-500' : 'border-blue-500'} shadow-sm p-4 hover:shadow-md transition-all cursor-pointer group`}>
    <div className="flex items-center justify-between mb-2">
      <div className="flex items-center gap-2">
        <span className="text-lg font-bold text-gray-900">{signal.symbol}</span>
        {signal.is_strong && <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 text-xs font-semibold rounded-full flex items-center gap-1"><Zap size={12} /> STRONG</span>}
      </div>
      <div className="flex items-center gap-2">
        <span className="text-lg font-semibold text-gray-900">${signal.price.toFixed(2)}</span>
        <ChevronRight size={18} className="text-gray-400 group-hover:text-blue-600 transition-colors" />
      </div>
    </div>
    <div className="grid grid-cols-2 gap-2 text-sm">
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
    </div>
  </div>
);

// Position Row
const PositionRow = ({ position, onClick }) => {
  const pnlColor = position.pnl_pct >= 0 ? 'text-emerald-600' : 'text-red-500';
  const pnlBg = position.pnl_pct >= 0 ? 'bg-emerald-50' : 'bg-red-50';
  return (
    <tr onClick={() => onClick(position)} className="hover:bg-blue-50 transition-colors cursor-pointer group">
      <td className="py-3 px-4"><div className="flex items-center gap-2"><span className="font-semibold text-gray-900">{position.symbol}</span><Eye size={14} className="text-gray-300 group-hover:text-blue-500" /></div></td>
      <td className="py-3 px-4 text-gray-600">{position.shares}</td>
      <td className="py-3 px-4 text-gray-600">${position.entry_price.toFixed(2)}</td>
      <td className="py-3 px-4 font-medium text-gray-900">${position.current_price.toFixed(2)}</td>
      <td className="py-3 px-4"><span className={`inline-flex items-center gap-1 px-2 py-1 rounded-md font-semibold text-sm ${pnlBg} ${pnlColor}`}>{position.pnl_pct >= 0 ? '+' : ''}{position.pnl_pct.toFixed(1)}%</span></td>
      <td className="py-3 px-4 text-gray-500"><Clock size={14} className="inline mr-1" />{position.days_held}d</td>
    </tr>
  );
};

// ============================================================================
// Main Dashboard
// ============================================================================

function Dashboard() {
  const { user, logout } = useAuth();
  const [signals, setSignals] = useState(mockSignals);
  const [positions, setPositions] = useState(mockPositions);
  const [historicalTrades, setHistoricalTrades] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [lastScan, setLastScan] = useState(new Date());
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [chartModal, setChartModal] = useState(null);

  useEffect(() => { setHistoricalTrades(generateHistoricalTrades()); }, []);

  const runScan = async () => {
    setScanning(true);
    await new Promise(r => setTimeout(r, 1500));
    setLastScan(new Date());
    setScanning(false);
  };

  const handleBuy = (signal) => {
    if (!user) { setShowLoginModal(true); return; }
    const newPosition = {
      id: Date.now(), symbol: signal.symbol, shares: Math.floor(10000 / signal.price),
      entry_price: signal.price, current_price: signal.price, pnl_pct: 0, days_held: 0,
      entry_date: new Date().toISOString().split('T')[0], stop_loss: signal.stop_loss, profit_target: signal.profit_target
    };
    setPositions([newPosition, ...positions]);
    setSignals(signals.filter(s => s.symbol !== signal.symbol));
    setChartModal(null);
  };

  const handleSell = (position) => {
    if (!user) { setShowLoginModal(true); return; }
    const trade = {
      id: Date.now(), symbol: position.symbol, entryDate: position.entry_date,
      exitDate: new Date().toISOString().split('T')[0], entryPrice: position.entry_price,
      exitPrice: position.current_price, shares: position.shares,
      pnl: (position.current_price - position.entry_price) * position.shares,
      pnlPct: position.pnl_pct, exitReason: 'MANUAL', daysHeld: position.days_held
    };
    setHistoricalTrades([trade, ...historicalTrades]);
    setPositions(positions.filter(p => p.id !== position.id));
    setChartModal(null);
  };

  const totalValue = positions.reduce((sum, p) => sum + p.shares * p.current_price, 0);
  const totalCost = positions.reduce((sum, p) => sum + p.shares * p.entry_price, 0);
  const totalPnlPct = totalCost > 0 ? ((totalValue - totalCost) / totalCost) * 100 : 0;
  const wins = historicalTrades.filter(t => t.pnl > 0);
  const winRate = historicalTrades.length > 0 ? (wins.length / historicalTrades.length * 100) : 0;
  const totalHistoricalPnl = historicalTrades.reduce((sum, t) => sum + t.pnl, 0);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
              <TrendingUp className="text-white" size={24} />
            </div>
            <div><h1 className="text-xl font-bold text-gray-900">Stocker</h1><p className="text-xs text-gray-500">DWAP Trading System</p></div>
          </div>

          <nav className="flex items-center gap-1 bg-gray-100 p-1 rounded-xl">
            <button onClick={() => setActiveTab('dashboard')} className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'dashboard' ? 'bg-white shadow text-gray-900' : 'text-gray-600 hover:text-gray-900'}`}>
              <BarChart3 size={16} className="inline mr-2" />Dashboard
            </button>
            <button onClick={() => setActiveTab('history')} className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'history' ? 'bg-white shadow text-gray-900' : 'text-gray-600 hover:text-gray-900'}`}>
              <History size={16} className="inline mr-2" />Trade History
            </button>
          </nav>
          
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-500">Last: {lastScan.toLocaleTimeString()}</span>
            <button onClick={runScan} disabled={scanning} className={`px-4 py-2 rounded-lg font-medium text-sm flex items-center gap-2 ${scanning ? 'bg-gray-100 text-gray-400' : 'bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-500/20'}`}>
              <RefreshCw size={16} className={scanning ? 'animate-spin' : ''} />{scanning ? 'Scanning...' : 'Scan'}
            </button>
            {user ? (
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white font-medium">{user.name[0]}</div>
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
        {activeTab === 'dashboard' ? (
          <>
            <div className="grid grid-cols-5 gap-4 mb-6">
              <MetricCard title="Portfolio Value" value={`$${totalValue.toLocaleString(undefined, {maximumFractionDigits: 0})}`} icon={Wallet} trend="up" />
              <MetricCard title="Open P&L" value={`${totalPnlPct >= 0 ? '+' : ''}${totalPnlPct.toFixed(1)}%`} icon={totalPnlPct >= 0 ? TrendingUp : TrendingDown} trend={totalPnlPct >= 0 ? 'up' : 'down'} />
              <MetricCard title="Positions" value={`${positions.length}/15`} icon={PieIcon} />
              <MetricCard title="Signals" value={signals.length} subtitle={`${signals.filter(s => s.is_strong).length} strong`} icon={Zap} />
              <MetricCard title="Win Rate" value={`${winRate.toFixed(0)}%`} subtitle={`${historicalTrades.length} trades`} icon={Target} />
            </div>

            <div className="grid grid-cols-3 gap-6">
              <div className="col-span-1">
                <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                  <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-gray-900">Live Signals</h2>
                    <span className="text-xs text-gray-500">Click to view chart</span>
                  </div>
                  <div className="p-4 space-y-3 max-h-[600px] overflow-y-auto">
                    {signals.map((s, i) => <SignalCard key={i} signal={s} onClick={(sig) => setChartModal({ type: 'signal', data: sig, symbol: sig.symbol })} />)}
                    {signals.length === 0 && <div className="text-center py-8 text-gray-500"><Activity className="w-12 h-12 mx-auto text-gray-300 mb-3" /><p>No signals</p></div>}
                  </div>
                </div>
              </div>

              <div className="col-span-2">
                <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                  <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-gray-900">Open Positions</h2>
                    <span className="text-xs text-gray-500">Click row to view chart & sell</span>
                  </div>
                  <table className="w-full">
                    <thead className="bg-gray-50 border-b border-gray-100">
                      <tr>{['Symbol', 'Shares', 'Entry', 'Current', 'P&L', 'Days'].map(h => <th key={h} className="py-3 px-4 text-left text-xs font-semibold text-gray-500 uppercase">{h}</th>)}</tr>
                    </thead>
                    <tbody>
                      {positions.map(p => <PositionRow key={p.id} position={p} onClick={(pos) => setChartModal({ type: 'position', data: pos, symbol: pos.symbol })} />)}
                    </tbody>
                  </table>
                  {positions.length === 0 && <div className="text-center py-12 text-gray-500"><PieIcon className="w-12 h-12 mx-auto text-gray-300 mb-3" /><p>No positions - click a signal to buy</p></div>}
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="space-y-6">
            <div className="grid grid-cols-4 gap-4">
              <MetricCard title="Total Trades" value={historicalTrades.length} icon={History} />
              <MetricCard title="Win Rate" value={`${winRate.toFixed(0)}%`} subtitle={`${wins.length}W / ${historicalTrades.length - wins.length}L`} icon={Target} trend={winRate > 50 ? 'up' : 'down'} />
              <MetricCard title="Total P&L" value={`$${totalHistoricalPnl.toLocaleString(undefined, {maximumFractionDigits: 0})}`} icon={Wallet} trend={totalHistoricalPnl >= 0 ? 'up' : 'down'} />
              <MetricCard title="Avg Return" value={`${historicalTrades.length ? (historicalTrades.reduce((s,t) => s + t.pnlPct, 0) / historicalTrades.length).toFixed(1) : 0}%`} icon={BarChart3} />
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
              <div className="px-5 py-4 border-b border-gray-100"><h2 className="text-lg font-semibold text-gray-900">Trade History (2 Years)</h2></div>
              <div className="overflow-x-auto max-h-[600px]">
                <table className="w-full">
                  <thead className="bg-gray-50 border-b border-gray-100 sticky top-0">
                    <tr>{['Symbol', 'Entry', 'Exit', 'Entry $', 'Exit $', 'Return', 'P&L', 'Reason'].map(h => <th key={h} className="py-3 px-4 text-left text-xs font-semibold text-gray-500 uppercase">{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {historicalTrades.map(t => (
                      <tr key={t.id} className="hover:bg-gray-50 border-b border-gray-50">
                        <td className="py-3 px-4 font-medium">{t.symbol}</td>
                        <td className="py-3 px-4 text-gray-500 text-sm">{t.entryDate}</td>
                        <td className="py-3 px-4 text-gray-500 text-sm">{t.exitDate}</td>
                        <td className="py-3 px-4">${t.entryPrice.toFixed(2)}</td>
                        <td className="py-3 px-4">${t.exitPrice.toFixed(2)}</td>
                        <td className="py-3 px-4"><span className={`px-2 py-1 rounded text-sm font-semibold ${t.pnlPct >= 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-red-50 text-red-500'}`}>{t.pnlPct >= 0 ? '+' : ''}{t.pnlPct.toFixed(1)}%</span></td>
                        <td className={`py-3 px-4 font-medium ${t.pnl >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>${t.pnl.toFixed(0)}</td>
                        <td className="py-3 px-4"><span className={`px-2 py-1 rounded text-xs font-medium ${t.exitReason === 'PROFIT_TARGET' ? 'bg-emerald-100 text-emerald-700' : t.exitReason === 'STOP_LOSS' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'}`}>{t.exitReason}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>

      {showLoginModal && <LoginModal onClose={() => setShowLoginModal(false)} />}
      {chartModal && <StockChartModal {...chartModal} onClose={() => setChartModal(null)} onAction={chartModal.type === 'signal' ? handleBuy : handleSell} />}
    </div>
  );
}

export default function App() {
  return <AuthProvider><Dashboard /></AuthProvider>;
}
