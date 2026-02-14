import React, { useState, useEffect, useCallback } from 'react';
import { Share2, Check, X, RefreshCw, Trash2, Image, MessageSquare, TrendingUp, BarChart3, Globe } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const PLATFORMS = [
  { id: 'all', label: 'All' },
  { id: 'twitter', label: 'Twitter/X' },
  { id: 'instagram', label: 'Instagram' },
];

const STATUSES = [
  { id: 'all', label: 'All' },
  { id: 'draft', label: 'Draft' },
  { id: 'approved', label: 'Approved' },
  { id: 'rejected', label: 'Rejected' },
  { id: 'posted', label: 'Posted' },
];

const POST_TYPES = [
  { id: 'all', label: 'All' },
  { id: 'trade_result', label: 'Trade Result' },
  { id: 'missed_opportunity', label: 'Missed Opportunity' },
  { id: 'weekly_recap', label: 'Weekly Recap' },
  { id: 'regime_commentary', label: 'Regime' },
];

const STATUS_COLORS = {
  draft: 'bg-gray-100 text-gray-700',
  approved: 'bg-green-100 text-green-700',
  rejected: 'bg-red-100 text-red-700',
  posted: 'bg-blue-100 text-blue-700',
};

const TYPE_COLORS = {
  trade_result: 'bg-emerald-100 text-emerald-700',
  missed_opportunity: 'bg-amber-100 text-amber-700',
  weekly_recap: 'bg-purple-100 text-purple-700',
  regime_commentary: 'bg-sky-100 text-sky-700',
};

const TYPE_LABELS = {
  trade_result: 'Trade Result',
  missed_opportunity: 'Missed Opportunity',
  weekly_recap: 'Weekly Recap',
  regime_commentary: 'Regime',
};

export default function SocialTab({ fetchWithAuth }) {
  const [stats, setStats] = useState(null);
  const [posts, setPosts] = useState([]);
  const [previews, setPreviews] = useState({});
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState({});

  // Filters
  const [platform, setPlatform] = useState('all');
  const [status, setStatus] = useState('all');
  const [postType, setPostType] = useState('all');

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetchWithAuth(`${API_URL}/api/admin/social/stats`);
      if (res.ok) setStats(await res.json());
    } catch (err) {
      console.error('Failed to fetch social stats:', err);
    }
  }, [fetchWithAuth]);

  const fetchPosts = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: '50' });
      if (platform !== 'all') params.set('platform', platform);
      if (status !== 'all') params.set('status', status);
      if (postType !== 'all') params.set('post_type', postType);

      const res = await fetchWithAuth(`${API_URL}/api/admin/social/posts?${params}`);
      if (res.ok) {
        const data = await res.json();
        setPosts(data.posts || []);
      }
    } catch (err) {
      console.error('Failed to fetch social posts:', err);
    } finally {
      setLoading(false);
    }
  }, [fetchWithAuth, platform, status, postType]);

  const fetchPreview = useCallback(async (postId) => {
    try {
      const res = await fetchWithAuth(`${API_URL}/api/admin/social/posts/${postId}/preview`);
      if (res.ok) {
        const data = await res.json();
        setPreviews(prev => ({ ...prev, [postId]: data }));
      }
    } catch (err) {
      console.error(`Failed to fetch preview for post ${postId}:`, err);
    }
  }, [fetchWithAuth]);

  useEffect(() => {
    fetchStats();
    fetchPosts();
  }, [fetchStats, fetchPosts]);

  // Fetch previews for posts that have images
  useEffect(() => {
    posts.forEach(post => {
      if (post.image_s3_key && !previews[post.id]) {
        fetchPreview(post.id);
      }
    });
  }, [posts, previews, fetchPreview]);

  const handleAction = async (postId, action, method = 'POST', confirmMsg = null) => {
    if (confirmMsg && !window.confirm(confirmMsg)) return;
    setActionLoading(prev => ({ ...prev, [postId]: action }));
    try {
      const res = await fetchWithAuth(`${API_URL}/api/admin/social/${action}`, { method });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(`Action failed: ${err.detail || res.statusText}`);
      }
      await Promise.all([fetchStats(), fetchPosts()]);
    } catch (err) {
      console.error(`Action ${action} failed:`, err);
      alert(`Action failed: ${err.message}`);
    } finally {
      setActionLoading(prev => ({ ...prev, [postId]: null }));
    }
  };

  const approve = (id) => handleAction(id, `posts/${id}/approve`);
  const reject = (id) => handleAction(id, `posts/${id}/reject`);
  const regenerate = (id) => handleAction(id, `posts/${id}/regenerate`);
  const deletePost = (id) => handleAction(id, `posts/${id}`, 'DELETE', 'Delete this post? This cannot be undone.');

  const generateChart = async (id) => {
    setActionLoading(prev => ({ ...prev, [id]: 'generate-chart' }));
    try {
      const res = await fetchWithAuth(`${API_URL}/api/admin/social/generate-chart/${id}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setPreviews(prev => ({ ...prev, [id]: { ...prev[id], image_url: data.image_url } }));
      } else {
        const err = await res.json().catch(() => ({}));
        alert(`Chart generation failed: ${err.detail || res.statusText}`);
      }
      await Promise.all([fetchStats(), fetchPosts()]);
    } catch (err) {
      console.error('Chart generation failed:', err);
      alert(`Chart generation failed: ${err.message}`);
    } finally {
      setActionLoading(prev => ({ ...prev, [id]: null }));
    }
  };

  const twitterPosts = posts.filter(p => p.platform === 'twitter');
  const instagramPosts = posts.filter(p => p.platform === 'instagram');

  return (
    <div className="space-y-6">
      {/* Stats Bar */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<Share2 className="text-blue-600" />}
          label="Total Posts"
          value={stats?.total ?? '-'}
        />
        <StatCard
          icon={<MessageSquare className="text-gray-600" />}
          label="Drafts"
          value={stats?.by_status?.draft ?? '-'}
          subtext="Pending review"
        />
        <StatCard
          icon={<Check className="text-green-600" />}
          label="Approved"
          value={stats?.by_status?.approved ?? '-'}
        />
        <StatCard
          icon={<Globe className="text-purple-600" />}
          label="Posted"
          value={stats?.by_status?.posted ?? '-'}
        />
      </div>

      {/* Filters Row */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
        <div className="flex flex-wrap gap-6">
          <FilterGroup label="Platform" options={PLATFORMS} value={platform} onChange={setPlatform} />
          <FilterGroup label="Status" options={STATUSES} value={status} onChange={setStatus} />
          <FilterGroup label="Post Type" options={POST_TYPES} value={postType} onChange={setPostType} />
        </div>
      </div>

      {/* Post Feed */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
        </div>
      ) : posts.length === 0 ? (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
          <Share2 size={48} className="mx-auto text-gray-300 mb-4" />
          <h3 className="text-lg font-semibold text-gray-700 mb-2">No social posts yet</h3>
          <p className="text-gray-500">Posts are auto-generated nightly at 8 PM ET from walk-forward simulation results.</p>
        </div>
      ) : platform === 'all' ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Twitter/X ({twitterPosts.length})</h3>
            {twitterPosts.length === 0 ? (
              <p className="text-sm text-gray-400 py-4">No Twitter posts match filters.</p>
            ) : (
              twitterPosts.map(post => (
                <TwitterCard
                  key={post.id}
                  post={post}
                  preview={previews[post.id]}
                  actionLoading={actionLoading[post.id]}
                  onApprove={approve}
                  onReject={reject}
                  onRegenerate={regenerate}
                  onDelete={deletePost}
                />
              ))
            )}
          </div>
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Instagram ({instagramPosts.length})</h3>
            {instagramPosts.length === 0 ? (
              <p className="text-sm text-gray-400 py-4">No Instagram posts match filters.</p>
            ) : (
              instagramPosts.map(post => (
                <InstagramCard
                  key={post.id}
                  post={post}
                  preview={previews[post.id]}
                  actionLoading={actionLoading[post.id]}
                  onApprove={approve}
                  onReject={reject}
                  onRegenerate={regenerate}
                  onDelete={deletePost}
                  onGenerateChart={generateChart}
                />
              ))
            )}
          </div>
        </div>
      ) : platform === 'twitter' ? (
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Twitter/X ({twitterPosts.length})</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {twitterPosts.map(post => (
              <TwitterCard
                key={post.id}
                post={post}
                preview={previews[post.id]}
                actionLoading={actionLoading[post.id]}
                onApprove={approve}
                onReject={reject}
                onRegenerate={regenerate}
                onDelete={deletePost}
              />
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Instagram ({instagramPosts.length})</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {instagramPosts.map(post => (
              <InstagramCard
                key={post.id}
                post={post}
                preview={previews[post.id]}
                actionLoading={actionLoading[post.id]}
                onApprove={approve}
                onReject={reject}
                onRegenerate={regenerate}
                onDelete={deletePost}
                onGenerateChart={generateChart}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function FilterGroup({ label, options, value, onChange }) {
  return (
    <div>
      <label className="block text-xs font-medium text-gray-500 mb-1.5">{label}</label>
      <div className="flex gap-1">
        {options.map(opt => (
          <button
            key={opt.id}
            onClick={() => onChange(opt.id)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
              value === opt.id
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, subtext }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-3 mb-2">
        {icon}
        <span className="text-sm font-medium text-gray-500">{label}</span>
      </div>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
      {subtext && <p className="text-sm text-gray-500 mt-1">{subtext}</p>}
    </div>
  );
}

function PostBadges({ post }) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className={`px-2 py-0.5 rounded text-xs font-medium ${TYPE_COLORS[post.post_type] || 'bg-gray-100 text-gray-600'}`}>
        {TYPE_LABELS[post.post_type] || post.post_type}
      </span>
      <span className={`px-2 py-0.5 rounded text-xs font-medium capitalize ${STATUS_COLORS[post.status] || 'bg-gray-100 text-gray-600'}`}>
        {post.status}
      </span>
      <span className="text-xs text-gray-400">
        {new Date(post.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}
      </span>
    </div>
  );
}

function ActionButtons({ post, actionLoading, onApprove, onReject, onRegenerate, onDelete, extraButtons }) {
  const isLoading = !!actionLoading;
  const canModify = post.status === 'draft' || post.status === 'rejected';

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {canModify && (
        <button
          onClick={() => onApprove(post.id)}
          disabled={isLoading}
          className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-green-700 bg-green-50 hover:bg-green-100 rounded-lg transition-colors disabled:opacity-50"
          title="Approve"
        >
          {actionLoading === 'approve' ? <RefreshCw size={13} className="animate-spin" /> : <Check size={13} />}
          Approve
        </button>
      )}
      {(post.status === 'draft' || post.status === 'approved') && (
        <button
          onClick={() => onReject(post.id)}
          disabled={isLoading}
          className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-red-700 bg-red-50 hover:bg-red-100 rounded-lg transition-colors disabled:opacity-50"
          title="Reject"
        >
          {actionLoading === 'reject' ? <RefreshCw size={13} className="animate-spin" /> : <X size={13} />}
          Reject
        </button>
      )}
      {extraButtons}
      <button
        onClick={() => onRegenerate(post.id)}
        disabled={isLoading}
        className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
        title="Regenerate"
      >
        {actionLoading === 'regenerate' ? <RefreshCw size={13} className="animate-spin" /> : <RefreshCw size={13} />}
        Regen
      </button>
      {(post.status === 'draft' || post.status === 'rejected') && (
        <button
          onClick={() => onDelete(post.id)}
          disabled={isLoading}
          className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
          title="Delete"
        >
          {actionLoading === 'delete' ? <RefreshCw size={13} className="animate-spin" /> : <Trash2 size={13} />}
        </button>
      )}
    </div>
  );
}

function splitTextAndHashtags(text, hashtags) {
  const mainText = text || '';
  const tags = hashtags || '';
  return { mainText, tags };
}

function TwitterCard({ post, preview, actionLoading, onApprove, onReject, onRegenerate, onDelete }) {
  const { mainText, tags } = splitTextAndHashtags(post.text_content, post.hashtags);
  const fullText = tags ? `${mainText}\n\n${tags}` : mainText;
  const charCount = preview?.char_count ?? fullText.length;
  const overLimit = preview?.over_limit ?? charCount > 280;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="p-4 space-y-3">
        <PostBadges post={post} />

        {/* Mock Tweet */}
        <div className="border border-gray-200 rounded-xl p-4">
          <div className="flex gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center flex-shrink-0">
              <span className="text-white font-bold text-sm">R</span>
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1">
                <span className="font-bold text-sm text-gray-900">RigaCap</span>
                <span className="text-sm text-gray-500">@rigacap</span>
              </div>
              <p className="text-sm text-gray-900 mt-1 whitespace-pre-wrap break-words">{mainText}</p>
              {tags && (
                <p className="text-sm text-blue-500 mt-1 break-words">{tags}</p>
              )}
            </div>
          </div>
        </div>

        {/* Char Count */}
        <div className="flex items-center justify-between">
          <span className={`text-xs font-medium ${overLimit ? 'text-red-600' : 'text-gray-400'}`}>
            {charCount}/280
          </span>
          <ActionButtons
            post={post}
            actionLoading={actionLoading}
            onApprove={onApprove}
            onReject={onReject}
            onRegenerate={onRegenerate}
            onDelete={onDelete}
          />
        </div>
      </div>
    </div>
  );
}

function InstagramCard({ post, preview, actionLoading, onApprove, onReject, onRegenerate, onDelete, onGenerateChart }) {
  const { mainText, tags } = splitTextAndHashtags(post.text_content, post.hashtags);
  const imageUrl = preview?.image_url || null;
  const hasImage = !!post.image_s3_key || !!imageUrl;
  const chartLoading = actionLoading === 'generate-chart';

  const generateChartButton = !hasImage && (
    <button
      onClick={() => onGenerateChart(post.id)}
      disabled={!!actionLoading}
      className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-indigo-700 bg-indigo-50 hover:bg-indigo-100 rounded-lg transition-colors disabled:opacity-50"
      title="Generate Chart"
    >
      {chartLoading ? <RefreshCw size={13} className="animate-spin" /> : <BarChart3 size={13} />}
      Chart
    </button>
  );

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="p-4 space-y-3">
        <PostBadges post={post} />

        {/* Mock Instagram Post */}
        <div className="border border-gray-200 rounded-xl overflow-hidden">
          {/* IG Header */}
          <div className="flex items-center gap-2.5 px-3 py-2.5 border-b border-gray-100">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-pink-500 via-red-500 to-yellow-500 flex items-center justify-center flex-shrink-0">
              <span className="text-white font-bold text-xs">R</span>
            </div>
            <span className="font-semibold text-sm text-gray-900">rigacap</span>
          </div>

          {/* Image Area */}
          {imageUrl ? (
            <img
              src={imageUrl}
              alt="Chart card"
              className="w-full aspect-square object-cover bg-gray-100"
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          ) : (
            <div className="w-full aspect-square bg-gray-50 flex flex-col items-center justify-center gap-3">
              <Image size={40} className="text-gray-300" />
              {post.image_s3_key ? (
                <span className="text-xs text-gray-400">Image loading...</span>
              ) : (
                <button
                  onClick={() => onGenerateChart(post.id)}
                  disabled={!!actionLoading}
                  className="px-4 py-2 text-sm font-medium text-indigo-700 bg-indigo-100 hover:bg-indigo-200 rounded-lg transition-colors disabled:opacity-50"
                >
                  {chartLoading ? (
                    <span className="flex items-center gap-2"><RefreshCw size={14} className="animate-spin" /> Generating...</span>
                  ) : (
                    'Generate Chart'
                  )}
                </button>
              )}
            </div>
          )}

          {/* Caption */}
          <div className="px-3 py-2.5">
            <p className="text-sm text-gray-900">
              <span className="font-semibold">rigacap</span>{' '}
              <span className="whitespace-pre-wrap break-words">{mainText}</span>
            </p>
            {tags && (
              <p className="text-sm text-blue-500 mt-1 break-words">{tags}</p>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex justify-end">
          <ActionButtons
            post={post}
            actionLoading={actionLoading}
            onApprove={onApprove}
            onReject={onReject}
            onRegenerate={onRegenerate}
            onDelete={onDelete}
            extraButtons={generateChartButton}
          />
        </div>
      </div>
    </div>
  );
}
