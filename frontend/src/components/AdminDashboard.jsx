import React, { useState, useEffect } from 'react';
import { Users, Activity, DollarSign, Clock, Search, ChevronLeft, ChevronRight, ToggleLeft, ToggleRight, Plus } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function AdminDashboard() {
  const { fetchWithAuth, isAdmin } = useAuth();
  const [stats, setStats] = useState(null);
  const [serviceStatus, setServiceStatus] = useState(null);
  const [users, setUsers] = useState([]);
  const [usersPagination, setUsersPagination] = useState({ page: 1, total: 0, per_page: 20 });
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

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

  // Initial load
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchStats(),
        fetchServiceStatus(),
        fetchUsers(),
      ]);
      setLoading(false);
    };
    loadData();

    // Refresh every 30 seconds
    const interval = setInterval(() => {
      fetchStats();
      fetchServiceStatus();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  // Handle search
  const handleSearch = (e) => {
    e.preventDefault();
    fetchUsers(1, searchQuery);
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
          onClick={() => { fetchStats(); fetchServiceStatus(); fetchUsers(usersPagination.page, searchQuery); }}
          className="px-4 py-2 text-sm bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
        >
          Refresh
        </button>
      </div>

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

      {/* Users Table */}
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
