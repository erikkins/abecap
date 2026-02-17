import React, { useState, useEffect } from 'react';
import { Users, Activity, DollarSign, Clock, Search, ChevronLeft, ChevronRight, ToggleLeft, ToggleRight, Plus, Zap, TrendingUp, AlertCircle, CheckCircle, PlayCircle, RefreshCw, Beaker, Bot, Settings, Share2, Server } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import StrategyGenerator from './StrategyGenerator';
import WalkForwardSimulator from './WalkForwardSimulator';
import AutoSwitchConfig from './AutoSwitchConfig';
import StrategyEditor from './StrategyEditor';
import FlexibleBacktest from './FlexibleBacktest';
import SocialTab from './SocialTab';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const TABS = [
  { id: 'overview', label: 'Overview', icon: Activity },
  { id: 'strategies', label: 'Strategies', icon: TrendingUp },
  { id: 'lab', label: 'Strategy Lab', icon: Beaker },
  { id: 'autopilot', label: 'Auto-Pilot', icon: Bot },
  { id: 'social', label: 'Social', icon: Share2 },
  { id: 'users', label: 'Users', icon: Users },
];

export default function AdminDashboard() {
  const { fetchWithAuth, isAdmin } = useAuth();
  const [activeTab, setActiveTab] = useState('overview');
  const [stats, setStats] = useState(null);
  const [serviceStatus, setServiceStatus] = useState(null);
  const [users, setUsers] = useState([]);
  const [usersPagination, setUsersPagination] = useState({ page: 1, total: 0, per_page: 20 });
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Strategy management state
  const [strategies, setStrategies] = useState([]);
  const [activeStrategy, setActiveStrategy] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [strategiesLoading, setStrategiesLoading] = useState(false);

  // AWS Health state
  const [awsHealth, setAwsHealth] = useState(null);

  // Strategy Lab state
  const [showStrategyEditor, setShowStrategyEditor] = useState(false);
  const [editingStrategy, setEditingStrategy] = useState(null);

  // Fetch admin stats
  const fetchStats = async () => {
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/stats`);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  // Fetch service status
  const fetchServiceStatus = async () => {
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/service-status`);
      if (response.ok) {
        const data = await response.json();
        setServiceStatus(data);
      }
    } catch (err) {
      console.error('Failed to fetch service status:', err);
    }
  };

  // Fetch AWS health
  const fetchAwsHealth = async () => {
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/aws-health`);
      if (response.ok) {
        const data = await response.json();
        setAwsHealth(data);
      }
    } catch (err) {
      console.error('Failed to fetch AWS health:', err);
    }
  };

  // Fetch users
  const fetchUsers = async (page = 1, search = '') => {
    try {
      let url = `${API_URL}/api/admin/users?page=${page}&per_page=20`;
      if (search) {
        url += `&search=${encodeURIComponent(search)}`;
      }
      const response = await fetchWithAuth(url);
      if (response.ok) {
        const data = await response.json();
        setUsers(data.users);
        setUsersPagination({
          page: data.page,
          total: data.total,
          per_page: data.per_page,
        });
      }
    } catch (err) {
      console.error('Failed to fetch users:', err);
    }
  };

  // Toggle user status
  const toggleUserStatus = async (userId, isActive) => {
    try {
      const endpoint = isActive ? 'disable' : 'enable';
      const response = await fetchWithAuth(`${API_URL}/api/admin/users/${userId}/${endpoint}`, {
        method: 'POST',
      });
      if (response.ok) {
        fetchUsers(usersPagination.page, searchQuery);
      }
    } catch (err) {
      console.error('Failed to toggle user status:', err);
    }
  };

  // Extend trial
  const extendTrial = async (userId, days = 7) => {
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/users/${userId}/extend-trial?days=${days}`, {
        method: 'POST',
      });
      if (response.ok) {
        fetchUsers(usersPagination.page, searchQuery);
        alert(`Trial extended by ${days} days`);
      }
    } catch (err) {
      console.error('Failed to extend trial:', err);
    }
  };

  // Comp subscription
  const compUser = async (userId) => {
    const days = prompt('Comp subscription for how many days?', '90');
    if (!days) return;
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/users/${userId}/comp?days=${days}`, {
        method: 'POST',
      });
      if (response.ok) {
        const data = await response.json();
        alert(data.message);
        fetchUsers(usersPagination.page, searchQuery);
      } else {
        const err = await response.json();
        alert(`Failed: ${err.detail}`);
      }
    } catch (err) {
      console.error('Failed to comp user:', err);
    }
  };

  // Revoke comp
  const revokeComp = async (userId) => {
    if (!confirm('Revoke this comp subscription?')) return;
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/users/${userId}/revoke-comp`, {
        method: 'POST',
      });
      if (response.ok) {
        alert('Comp revoked');
        fetchUsers(usersPagination.page, searchQuery);
      } else {
        const err = await response.json();
        alert(`Failed: ${err.detail}`);
      }
    } catch (err) {
      console.error('Failed to revoke comp:', err);
    }
  };

  // Fetch strategies
  const fetchStrategies = async () => {
    setStrategiesLoading(true);
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/strategies`);
      if (response.ok) {
        const data = await response.json();
        setStrategies(data);
        const active = data.find(s => s.is_active);
        if (active) setActiveStrategy(active);
      }
    } catch (err) {
      console.error('Failed to fetch strategies:', err);
    } finally {
      setStrategiesLoading(false);
    }
  };

  // Fetch latest analysis
  const fetchLatestAnalysis = async () => {
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/strategies/analysis`);
      if (response.ok) {
        const data = await response.json();
        setAnalysisResults(data);
      }
    } catch (err) {
      // 404 is expected if no analysis has been run yet
      if (err.message?.includes('404')) {
        console.log('No analysis results yet');
      } else {
        console.error('Failed to fetch analysis:', err);
      }
    }
  };

  // Run strategy analysis
  const runAnalysis = async (lookbackDays = 90) => {
    setAnalysisLoading(true);
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/strategies/analyze?lookback_days=${lookbackDays}`, {
        method: 'POST',
      });
      if (response.ok) {
        const data = await response.json();
        setAnalysisResults(data);
        await fetchStrategies(); // Refresh strategies to get updated evaluations
      } else {
        try {
          const error = await response.json();
          alert(`Analysis failed: ${error.detail || JSON.stringify(error)}`);
        } catch {
          const text = await response.text();
          alert(`Analysis failed (${response.status}): ${text.slice(0, 200)}`);
        }
      }
    } catch (err) {
      console.error('Failed to run analysis:', err);
      alert(`Failed to run analysis: ${err.message}`);
    } finally {
      setAnalysisLoading(false);
    }
  };

  // Activate a strategy
  const activateStrategy = async (strategyId) => {
    try {
      const response = await fetchWithAuth(`${API_URL}/api/admin/strategies/${strategyId}/activate`, {
        method: 'POST',
      });
      if (response.ok) {
        const data = await response.json();
        setActiveStrategy(data);
        await fetchStrategies(); // Refresh all strategies
        alert(`Strategy "${data.name}" is now active`);
      }
    } catch (err) {
      console.error('Failed to activate strategy:', err);
      alert('Failed to activate strategy');
    }
  };

  // Initial load
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchStats(),
        fetchServiceStatus(),
        fetchAwsHealth(),
        fetchUsers(),
        fetchStrategies(),
        fetchLatestAnalysis(),
      ]);
      setLoading(false);
    };
    loadData();

    // Refresh every 30 seconds
    const interval = setInterval(() => {
      fetchStats();
      fetchServiceStatus();
      fetchAwsHealth();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  // Handle search
  const handleSearch = (e) => {
    e.preventDefault();
    fetchUsers(1, searchQuery);
  };

  // Handle strategy created/updated
  const handleStrategyChange = () => {
    fetchStrategies();
    setShowStrategyEditor(false);
    setEditingStrategy(null);
  };

  if (!isAdmin) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
        <h3 className="text-lg font-semibold text-red-800">Access Denied</h3>
        <p className="text-red-600 mt-2">You don't have permission to view this page.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Admin Dashboard</h2>
        <button
          onClick={() => { fetchStats(); fetchServiceStatus(); fetchUsers(usersPagination.page, searchQuery); fetchStrategies(); }}
          className="px-4 py-2 text-sm bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="flex gap-4 -mb-px">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon size={18} />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <OverviewTab
          stats={stats}
          serviceStatus={serviceStatus}
          activeStrategy={activeStrategy}
          awsHealth={awsHealth}
        />
      )}

      {activeTab === 'strategies' && (
        <StrategiesTab
          strategies={strategies}
          activeStrategy={activeStrategy}
          analysisResults={analysisResults}
          analysisLoading={analysisLoading}
          runAnalysis={runAnalysis}
          activateStrategy={activateStrategy}
          onEditStrategy={(strategy) => {
            setEditingStrategy(strategy);
            setShowStrategyEditor(true);
            setActiveTab('lab');
          }}
        />
      )}

      {activeTab === 'lab' && (
        <StrategyLabTab
          fetchWithAuth={fetchWithAuth}
          strategies={strategies}
          onStrategyCreated={handleStrategyChange}
          showStrategyEditor={showStrategyEditor}
          setShowStrategyEditor={setShowStrategyEditor}
          editingStrategy={editingStrategy}
          setEditingStrategy={setEditingStrategy}
        />
      )}

      {activeTab === 'autopilot' && (
        <AutoPilotTab fetchWithAuth={fetchWithAuth} />
      )}

      {activeTab === 'social' && (
        <SocialTab fetchWithAuth={fetchWithAuth} />
      )}

      {activeTab === 'users' && (
        <UsersTab
          users={users}
          usersPagination={usersPagination}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          handleSearch={handleSearch}
          toggleUserStatus={toggleUserStatus}
          extendTrial={extendTrial}
          compUser={compUser}
          revokeComp={revokeComp}
          fetchUsers={fetchUsers}
        />
      )}
    </div>
  );
}

// Overview Tab Component
function OverviewTab({ stats, serviceStatus, activeStrategy, awsHealth }) {
  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<Users className="text-blue-600" />}
          label="Total Users"
          value={stats?.total_users || 0}
          subtext={`${stats?.new_users_week || 0} this week`}
        />
        <StatCard
          icon={<Clock className="text-yellow-600" />}
          label="Active Trials"
          value={stats?.active_trials || 0}
          subtext={`${stats?.expired_trials || 0} expired`}
        />
        <StatCard
          icon={<DollarSign className="text-green-600" />}
          label="Paid Subscribers"
          value={stats?.paid_subscribers || 0}
          subtext={`$${stats?.mrr?.toFixed(0) || 0} MRR`}
        />
        <StatCard
          icon={<Activity className="text-purple-600" />}
          label="System Status"
          value={serviceStatus?.overall_status === 'healthy' ? 'Healthy' : 'Degraded'}
          subtext={`${Object.keys(serviceStatus?.services || {}).length} services`}
          valueColor={serviceStatus?.overall_status === 'healthy' ? 'text-green-600' : 'text-yellow-600'}
        />
      </div>

      {/* Service Status */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Service Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {serviceStatus?.services && Object.entries(serviceStatus.services).map(([name, service]) => (
            <div key={name} className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-gray-700 capitalize">{name}</span>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  service.status === 'ok' ? 'bg-green-100 text-green-800' :
                  service.status === 'not_configured' ? 'bg-gray-100 text-gray-600' :
                  'bg-red-100 text-red-800'
                }`}>
                  {service.status}
                </span>
              </div>
              {service.latency_ms && (
                <p className="text-sm text-gray-500">{service.latency_ms}ms latency</p>
              )}
              {service.symbols_loaded !== undefined && (
                <p className="text-sm text-gray-500">{service.symbols_loaded} symbols</p>
              )}
              {service.signals_today !== undefined && (
                <p className="text-sm text-gray-500">{service.signals_today} signals today</p>
              )}
              {service.error && (
                <p className="text-sm text-red-500 truncate">{service.error}</p>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Infrastructure Health */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center gap-2 mb-4">
          <Server size={20} className="text-gray-700" />
          <h3 className="text-lg font-semibold text-gray-900">Infrastructure</h3>
        </div>

        {!awsHealth || awsHealth.local_dev || awsHealth.error ? (
          <div className="p-4 bg-gray-50 rounded-lg text-sm text-gray-500">
            {awsHealth?.error
              ? `Infrastructure metrics error: ${awsHealth.error}`
              : 'Infrastructure metrics unavailable (local dev)'}
          </div>
        ) : (
          <>
            {/* Alarm Status */}
            {awsHealth.alarms.length > 0 && (
              <div className="mb-4">
                <p className="text-sm font-medium text-gray-600 mb-2">Alarm Status</p>
                <div className="flex flex-wrap gap-2">
                  {awsHealth.alarms.map((alarm) => (
                    <span
                      key={alarm.name}
                      className={`px-2.5 py-1 rounded-full text-xs font-medium ${
                        alarm.state === 'OK'
                          ? 'bg-green-100 text-green-800'
                          : alarm.state === 'ALARM'
                          ? 'bg-red-100 text-red-800'
                          : 'bg-gray-100 text-gray-600'
                      }`}
                    >
                      {alarm.name.replace('rigacap-prod-', '')}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Lambda */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-sm font-medium text-gray-700 mb-2">Lambda</p>
                <div className="space-y-1 text-sm text-gray-600">
                  <p>Invocations (24h): <span className="font-medium text-gray-900">{awsHealth.metrics.lambda?.invocations_24h?.toLocaleString() ?? '—'}</span></p>
                  <p>Errors (24h): <span className={`font-medium ${(awsHealth.metrics.lambda?.errors_24h || 0) > 0 ? 'text-red-600' : 'text-gray-900'}`}>{awsHealth.metrics.lambda?.errors_24h ?? '—'}</span></p>
                  <p>Avg Duration: <span className="font-medium text-gray-900">{awsHealth.metrics.lambda?.avg_duration_ms != null ? `${awsHealth.metrics.lambda.avg_duration_ms}ms` : '—'}</span></p>
                </div>
              </div>

              {/* API Gateway */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-sm font-medium text-gray-700 mb-2">API Gateway</p>
                <div className="space-y-1 text-sm text-gray-600">
                  <p>Requests (24h): <span className="font-medium text-gray-900">{awsHealth.metrics.api_gateway?.requests_24h?.toLocaleString() ?? '—'}</span></p>
                </div>
              </div>

              {/* RDS */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-sm font-medium text-gray-700 mb-2">RDS</p>
                <div className="space-y-1 text-sm text-gray-600">
                  <p>CPU: <span className={`font-medium ${(awsHealth.metrics.rds?.cpu_percent || 0) > 80 ? 'text-red-600' : 'text-gray-900'}`}>{awsHealth.metrics.rds?.cpu_percent != null ? `${awsHealth.metrics.rds.cpu_percent}%` : '—'}</span></p>
                  <p>Free Storage: <span className="font-medium text-gray-900">{awsHealth.metrics.rds?.free_storage_gb != null ? `${awsHealth.metrics.rds.free_storage_gb} GB` : '—'}</span></p>
                  <p>Connections: <span className="font-medium text-gray-900">{awsHealth.metrics.rds?.connections ?? '—'}</span></p>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Active Strategy Summary */}
      {activeStrategy && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Active Strategy</h3>
          <div className="p-4 bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-200 rounded-xl">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <Zap size={18} className="text-emerald-600" />
                  <span className="text-sm font-medium text-emerald-700">Currently Active</span>
                </div>
                <h4 className="text-xl font-bold text-gray-900">{activeStrategy.name}</h4>
                <p className="text-sm text-gray-600 mt-1">{activeStrategy.description}</p>
              </div>
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                activeStrategy.strategy_type === 'momentum'
                  ? 'bg-purple-100 text-purple-800'
                  : 'bg-blue-100 text-blue-800'
              }`}>
                {activeStrategy.strategy_type.toUpperCase()}
              </span>
            </div>
            {activeStrategy.latest_evaluation && (
              <div className="mt-4 pt-4 border-t border-emerald-200 grid grid-cols-4 gap-4">
                <div>
                  <p className="text-xs text-gray-500">Sharpe</p>
                  <p className="font-semibold text-emerald-700">{activeStrategy.latest_evaluation.sharpe_ratio?.toFixed(2) || '-'}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Return</p>
                  <p className={`font-semibold ${(activeStrategy.latest_evaluation.total_return_pct || 0) >= 0 ? 'text-emerald-700' : 'text-red-600'}`}>
                    {(activeStrategy.latest_evaluation.total_return_pct || 0) >= 0 ? '+' : ''}{activeStrategy.latest_evaluation.total_return_pct?.toFixed(1) || '-'}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Max DD</p>
                  <p className="font-semibold text-red-600">-{activeStrategy.latest_evaluation.max_drawdown_pct?.toFixed(1) || '-'}%</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Score</p>
                  <p className="font-semibold">{activeStrategy.latest_evaluation.recommendation_score?.toFixed(0) || '-'}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Strategies Tab Component
function StrategiesTab({ strategies, activeStrategy, analysisResults, analysisLoading, runAnalysis, activateStrategy, onEditStrategy }) {
  return (
    <div className="space-y-6">
      {/* Header with Run Analysis button */}
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-900">Strategy Library</h3>
        <button
          onClick={() => runAnalysis(90)}
          disabled={analysisLoading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {analysisLoading ? (
            <>
              <RefreshCw size={16} className="animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <PlayCircle size={16} />
              Run Analysis
            </>
          )}
        </button>
      </div>

      {/* Active Strategy Card */}
      {activeStrategy && (
        <div className="p-4 bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-200 rounded-xl">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Zap size={18} className="text-emerald-600" />
                <span className="text-sm font-medium text-emerald-700">Active Strategy</span>
              </div>
              <h4 className="text-xl font-bold text-gray-900">{activeStrategy.name}</h4>
              <p className="text-sm text-gray-600 mt-1">{activeStrategy.description}</p>
            </div>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
              activeStrategy.strategy_type === 'momentum'
                ? 'bg-purple-100 text-purple-800'
                : 'bg-blue-100 text-blue-800'
            }`}>
              {activeStrategy.strategy_type.toUpperCase()}
            </span>
          </div>
          <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-gray-500">Max Positions</p>
              <p className="font-semibold">{activeStrategy.parameters?.max_positions || '-'}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Position Size</p>
              <p className="font-semibold">{activeStrategy.parameters?.position_size_pct || '-'}%</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Stop Type</p>
              <p className="font-semibold">
                {activeStrategy.parameters?.trailing_stop_pct
                  ? `${activeStrategy.parameters.trailing_stop_pct}% Trailing`
                  : `${activeStrategy.parameters?.stop_loss_pct || '-'}% Fixed`}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Activated</p>
              <p className="font-semibold">
                {activeStrategy.activated_at
                  ? new Date(activeStrategy.activated_at).toLocaleDateString()
                  : '-'}
              </p>
            </div>
          </div>
          {activeStrategy.latest_evaluation && (
            <div className="mt-4 pt-4 border-t border-emerald-200 grid grid-cols-4 gap-4">
              <div>
                <p className="text-xs text-gray-500">Sharpe</p>
                <p className="font-semibold text-emerald-700">{activeStrategy.latest_evaluation.sharpe_ratio?.toFixed(2) || '-'}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Return</p>
                <p className={`font-semibold ${(activeStrategy.latest_evaluation.total_return_pct || 0) >= 0 ? 'text-emerald-700' : 'text-red-600'}`}>
                  {(activeStrategy.latest_evaluation.total_return_pct || 0) >= 0 ? '+' : ''}{activeStrategy.latest_evaluation.total_return_pct?.toFixed(1) || '-'}%
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Max DD</p>
                <p className="font-semibold text-red-600">-{activeStrategy.latest_evaluation.max_drawdown_pct?.toFixed(1) || '-'}%</p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Score</p>
                <p className="font-semibold">{activeStrategy.latest_evaluation.recommendation_score?.toFixed(0) || '-'}</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Strategy Library Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Strategy</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Sharpe</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Return</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Max DD</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Score</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {strategies.map((strategy) => (
                <tr key={strategy.id} className={strategy.is_active ? 'bg-emerald-50' : 'hover:bg-gray-50'}>
                  <td className="px-4 py-3">
                    <div className="font-medium text-gray-900">{strategy.name}</div>
                    <div className="text-xs text-gray-500 truncate max-w-[200px]">{strategy.description}</div>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      strategy.strategy_type === 'momentum'
                        ? 'bg-purple-100 text-purple-800'
                        : 'bg-blue-100 text-blue-800'
                    }`}>
                      {strategy.strategy_type}
                    </span>
                    {strategy.source === 'ai_generated' && (
                      <span className="ml-1 px-1.5 py-0.5 rounded text-xs bg-amber-100 text-amber-700">AI</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right font-medium">
                    {strategy.latest_evaluation?.sharpe_ratio?.toFixed(2) || '-'}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className={`font-medium ${(strategy.latest_evaluation?.total_return_pct || 0) >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                      {strategy.latest_evaluation?.total_return_pct != null
                        ? `${strategy.latest_evaluation.total_return_pct >= 0 ? '+' : ''}${strategy.latest_evaluation.total_return_pct.toFixed(1)}%`
                        : '-'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right text-red-600 font-medium">
                    {strategy.latest_evaluation?.max_drawdown_pct != null
                      ? `-${strategy.latest_evaluation.max_drawdown_pct.toFixed(1)}%`
                      : '-'}
                  </td>
                  <td className="px-4 py-3 text-right font-bold">
                    {strategy.latest_evaluation?.recommendation_score?.toFixed(0) || '-'}
                  </td>
                  <td className="px-4 py-3 text-center">
                    {strategy.is_active ? (
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">
                        <CheckCircle size={12} /> Active
                      </span>
                    ) : (
                      <span className="text-xs text-gray-500">Inactive</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-center">
                    <div className="flex items-center justify-center gap-2">
                      {strategy.is_custom && (
                        <button
                          onClick={() => onEditStrategy(strategy)}
                          className="px-2 py-1 text-xs font-medium text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded"
                        >
                          Edit
                        </button>
                      )}
                      {!strategy.is_active && (
                        <button
                          onClick={() => activateStrategy(strategy.id)}
                          className="px-3 py-1 text-xs font-medium text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded"
                        >
                          Activate
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Analysis Results Panel */}
      {analysisResults && (
        <div className="p-4 bg-gray-50 rounded-xl border border-gray-200">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-gray-900">Latest Analysis</h4>
            <span className="text-xs text-gray-500">
              {analysisResults.analysis_date
                ? new Date(analysisResults.analysis_date).toLocaleString()
                : '-'}
              {' '}&bull;{' '}{analysisResults.lookback_days} day lookback
            </span>
          </div>
          <div className="p-3 bg-white rounded-lg border border-gray-200">
            <div className="flex items-start gap-3">
              {analysisResults.recommended_strategy_id === analysisResults.current_active_strategy_id ? (
                <CheckCircle size={20} className="text-emerald-600 mt-0.5" />
              ) : (
                <AlertCircle size={20} className="text-amber-500 mt-0.5" />
              )}
              <div>
                <p className="text-sm text-gray-700 whitespace-pre-line">
                  {analysisResults.recommendation_notes}
                </p>
                {analysisResults.recommended_strategy_id !== analysisResults.current_active_strategy_id && (
                  <button
                    onClick={() => activateStrategy(analysisResults.recommended_strategy_id)}
                    className="mt-3 px-4 py-2 bg-amber-500 text-white text-sm font-medium rounded-lg hover:bg-amber-600 transition-colors"
                  >
                    Accept Recommendation
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Strategy Lab Tab Component
function StrategyLabTab({ fetchWithAuth, strategies, onStrategyCreated, showStrategyEditor, setShowStrategyEditor, editingStrategy, setEditingStrategy }) {
  return (
    <div className="space-y-6">
      {/* Create Strategy Button */}
      <div className="flex justify-end">
        <button
          onClick={() => {
            setEditingStrategy(null);
            setShowStrategyEditor(true);
          }}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
        >
          <Plus size={16} />
          Create Custom Strategy
        </button>
      </div>

      {/* Strategy Editor */}
      {showStrategyEditor && (
        <StrategyEditor
          fetchWithAuth={fetchWithAuth}
          strategy={editingStrategy}
          strategies={strategies}
          onSave={onStrategyCreated}
          onCancel={() => {
            setShowStrategyEditor(false);
            setEditingStrategy(null);
          }}
        />
      )}

      {/* AI Strategy Generator */}
      <StrategyGenerator
        fetchWithAuth={fetchWithAuth}
        onStrategyCreated={onStrategyCreated}
      />

      {/* Flexible Backtest */}
      <FlexibleBacktest
        fetchWithAuth={fetchWithAuth}
        strategies={strategies}
      />
    </div>
  );
}

// Auto-Pilot Tab Component
function AutoPilotTab({ fetchWithAuth }) {
  return (
    <div className="space-y-6">
      {/* Walk-Forward Simulator */}
      <WalkForwardSimulator fetchWithAuth={fetchWithAuth} />

      {/* Auto-Switch Configuration */}
      <AutoSwitchConfig fetchWithAuth={fetchWithAuth} />
    </div>
  );
}

// Users Tab Component
function UsersTab({ users, usersPagination, searchQuery, setSearchQuery, handleSearch, toggleUserStatus, extendTrial, compUser, revokeComp, fetchUsers }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200">
      <div className="p-6 border-b border-gray-200">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <h3 className="text-lg font-semibold text-gray-900">Users</h3>
          <form onSubmit={handleSearch} className="flex gap-2">
            <div className="relative">
              <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search users..."
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 transition-colors"
            >
              Search
            </button>
          </form>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Subscription</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Login</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {users.map((user) => (
              <tr key={user.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div>
                    <div className="text-sm font-medium text-gray-900">{user.name || 'No name'}</div>
                    <div className="text-sm text-gray-500">{user.email}</div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    user.is_active ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                  }`}>
                    {user.is_active ? 'Active' : 'Disabled'}
                  </span>
                  {user.role === 'admin' && (
                    <span className="ml-2 px-2 py-1 rounded text-xs font-medium bg-purple-100 text-purple-800">
                      Admin
                    </span>
                  )}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900 capitalize">{user.subscription_status || 'None'}</div>
                  {user.subscription_status === 'trial' && user.trial_days_remaining !== null && (
                    <div className="text-xs text-gray-500">
                      {user.trial_days_remaining} days left
                    </div>
                  )}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {user.created_at ? new Date(user.created_at).toLocaleDateString() : '-'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => toggleUserStatus(user.id, user.is_active)}
                      className={`p-1 rounded ${user.is_active ? 'text-red-600 hover:bg-red-50' : 'text-green-600 hover:bg-green-50'}`}
                      title={user.is_active ? 'Disable user' : 'Enable user'}
                    >
                      {user.is_active ? <ToggleRight size={20} /> : <ToggleLeft size={20} />}
                    </button>
                    {user.subscription_status === 'trial' && (
                      <button
                        onClick={() => extendTrial(user.id)}
                        className="p-1 rounded text-blue-600 hover:bg-blue-50"
                        title="Extend trial by 7 days"
                      >
                        <Plus size={20} />
                      </button>
                    )}
                    {user.subscription_status !== 'active' ? (
                      <button
                        onClick={() => compUser(user.id)}
                        className="px-2 py-1 rounded text-xs font-medium text-emerald-700 bg-emerald-50 hover:bg-emerald-100"
                        title="Grant comp subscription"
                      >
                        Comp
                      </button>
                    ) : (
                      <button
                        onClick={() => revokeComp(user.id)}
                        className="px-2 py-1 rounded text-xs font-medium text-orange-700 bg-orange-50 hover:bg-orange-100"
                        title="Revoke comp subscription"
                      >
                        Revoke
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="px-6 py-4 border-t border-gray-200 flex justify-between items-center">
        <p className="text-sm text-gray-500">
          Showing {((usersPagination.page - 1) * usersPagination.per_page) + 1} to{' '}
          {Math.min(usersPagination.page * usersPagination.per_page, usersPagination.total)} of{' '}
          {usersPagination.total} users
        </p>
        <div className="flex gap-2">
          <button
            onClick={() => fetchUsers(usersPagination.page - 1, searchQuery)}
            disabled={usersPagination.page <= 1}
            className="p-2 rounded border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft size={16} />
          </button>
          <button
            onClick={() => fetchUsers(usersPagination.page + 1, searchQuery)}
            disabled={usersPagination.page * usersPagination.per_page >= usersPagination.total}
            className="p-2 rounded border border-gray-300 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, subtext, valueColor = 'text-gray-900' }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-3 mb-2">
        {icon}
        <span className="text-sm font-medium text-gray-500">{label}</span>
      </div>
      <div className={`text-2xl font-bold ${valueColor}`}>{value}</div>
      {subtext && <p className="text-sm text-gray-500 mt-1">{subtext}</p>}
    </div>
  );
}
