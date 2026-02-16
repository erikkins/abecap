import React, { useState, useEffect, useCallback } from 'react';
import { Share2, Check, X, RefreshCw, Trash2, Image, MessageSquare, TrendingUp, BarChart3, Globe, Send, Plus, Edit3, Save, Power, Rocket, ChevronDown, ChevronUp } from 'lucide-react';

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
  { id: 'manual', label: 'Manual' },
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
  manual: 'bg-indigo-100 text-indigo-700',
};

const TYPE_LABELS = {
  trade_result: 'Trade Result',
  missed_opportunity: 'Missed Opportunity',
  weekly_recap: 'Weekly Recap',
  regime_commentary: 'Regime',
  manual: 'Manual',
};

// Navy+Gold logo SVG as inline component
function RigaCapLogo({ size = 40, className = '' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" className={className}>
      <defs>
        <linearGradient id="social-logo-bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#172554"/>
          <stop offset="100%" stopColor="#1e3a5f"/>
        </linearGradient>
      </defs>
      <circle cx="50" cy="50" r="50" fill="url(#social-logo-bg)"/>
      <path d="M 50 15 L 46.5 38 Q 46.5 40, 44 40 L 41 40 Q 39 40, 39 42 L 39 44 Q 39 46, 41 46 L 44 46 Q 46.5 46, 46.5 48 L 43 69 Q 43 71, 40 71 L 36 71 Q 34 71, 34 73 L 34 76 Q 34 78, 36 78 L 40 78 Q 42 78, 42 80 L 42 92 L 58 92 L 58 80 Q 58 78, 60 78 L 64 78 Q 66 78, 66 76 L 66 73 Q 66 71, 64 71 L 60 71 Q 57 71, 57 69 L 53.5 48 Q 53.5 46, 56 46 L 59 46 Q 61 46, 61 44 L 61 42 Q 61 40, 59 40 L 56 40 Q 53.5 40, 53.5 38 Z" fill="#ffffff"/>
      <circle cx="50" cy="15" r="3" fill="#f59e0b"/>
    </svg>
  );
}

export default function SocialTab({ fetchWithAuth }) {
  const [stats, setStats] = useState(null);
  const [posts, setPosts] = useState([]);
  const [previews, setPreviews] = useState({});
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState({});
  const [showCompose, setShowCompose] = useState(false);
  const [publishingLive, setPublishingLive] = useState(() => localStorage.getItem('social_live') === 'true');

  const toggleLive = () => {
    const next = !publishingLive;
    if (next && !window.confirm('Enable live publishing? Posts will be sent to Twitter/Instagram when you click Publish.')) return;
    setPublishingLive(next);
    localStorage.setItem('social_live', String(next));
  };

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

  const publish = async (id) => {
    if (!window.confirm('Publish this post live? This will post to the platform immediately.')) return;
    setActionLoading(prev => ({ ...prev, [id]: 'publish' }));
    try {
      const res = await fetchWithAuth(`${API_URL}/api/admin/social/posts/${id}/publish`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        const url = data.tweet_url || data.permalink || '';
        alert(`Published successfully!${url ? `\n${url}` : ''}`);
      } else {
        const err = await res.json().catch(() => ({}));
        alert(`Publish failed: ${err.detail || res.statusText}`);
      }
      await Promise.all([fetchStats(), fetchPosts()]);
    } catch (err) {
      console.error('Publish failed:', err);
      alert(`Publish failed: ${err.message}`);
    } finally {
      setActionLoading(prev => ({ ...prev, [id]: null }));
    }
  };

  const editPost = async (id, textContent, hashtags) => {
    try {
      const res = await fetchWithAuth(`${API_URL}/api/admin/social/posts/${id}/edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text_content: textContent, hashtags }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(`Edit failed: ${err.detail || res.statusText}`);
        return false;
      }
      await fetchPosts();
      return true;
    } catch (err) {
      console.error('Edit failed:', err);
      alert(`Edit failed: ${err.message}`);
      return false;
    }
  };

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

  const handleCompose = async (composeData) => {
    try {
      const res = await fetchWithAuth(`${API_URL}/api/admin/social/posts/compose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(composeData),
      });
      if (res.ok) {
        setShowCompose(false);
        await Promise.all([fetchStats(), fetchPosts()]);
      } else {
        const err = await res.json().catch(() => ({}));
        alert(`Compose failed: ${err.detail || res.statusText}`);
      }
    } catch (err) {
      console.error('Compose failed:', err);
      alert(`Compose failed: ${err.message}`);
    }
  };

  const twitterPosts = posts.filter(p => p.platform === 'twitter');
  const instagramPosts = posts.filter(p => p.platform === 'instagram');

  return (
    <div className="space-y-6">
      {/* Live Switch */}
      <div className={`flex items-center justify-between rounded-xl border-2 px-5 py-3 transition-colors ${
        publishingLive
          ? 'bg-green-50 border-green-300'
          : 'bg-amber-50 border-amber-300'
      }`}>
        <div className="flex items-center gap-3">
          <Power size={18} className={publishingLive ? 'text-green-600' : 'text-amber-600'} />
          <div>
            <span className={`text-sm font-semibold ${publishingLive ? 'text-green-800' : 'text-amber-800'}`}>
              {publishingLive ? 'Publishing is LIVE' : 'Publishing is OFF'}
            </span>
            <p className="text-xs text-gray-500">
              {publishingLive
                ? 'Publish buttons are active — posts will go live to Twitter/Instagram.'
                : 'Publish buttons are hidden. Enable when ready to go live.'}
            </p>
          </div>
        </div>
        <button
          onClick={toggleLive}
          className={`relative inline-flex h-7 w-12 items-center rounded-full transition-colors ${
            publishingLive ? 'bg-green-500' : 'bg-gray-300'
          }`}
        >
          <span className={`inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform ${
            publishingLive ? 'translate-x-6' : 'translate-x-1'
          }`} />
        </button>
      </div>

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

      {/* Filters Row + Compose Button */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
        <div className="flex flex-wrap items-end gap-6">
          <FilterGroup label="Platform" options={PLATFORMS} value={platform} onChange={setPlatform} />
          <FilterGroup label="Status" options={STATUSES} value={status} onChange={setStatus} />
          <FilterGroup label="Post Type" options={POST_TYPES} value={postType} onChange={setPostType} />
          <div className="ml-auto">
            <button
              onClick={() => setShowCompose(true)}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
            >
              <Plus size={16} />
              New Post
            </button>
          </div>
        </div>
      </div>

      {/* Compose Modal */}
      {showCompose && (
        <ComposeModal onClose={() => setShowCompose(false)} onSubmit={handleCompose} />
      )}

      {/* Post Feed */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
        </div>
      ) : posts.length === 0 ? (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
          <Share2 size={48} className="mx-auto text-gray-300 mb-4" />
          <h3 className="text-lg font-semibold text-gray-700 mb-2">No social posts yet</h3>
          <p className="text-gray-500 mb-4">Posts are auto-generated nightly at 8 PM ET from walk-forward simulation results.</p>
          <button
            onClick={() => setShowCompose(true)}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
          >
            <Plus size={16} />
            Create Your First Post
          </button>
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
                  onPublish={publish}
                  onEdit={editPost}
                  publishingLive={publishingLive}
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
                  onPublish={publish}
                  onEdit={editPost}
                  onGenerateChart={generateChart}
                  publishingLive={publishingLive}
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
                onPublish={publish}
                onEdit={editPost}
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
                onPublish={publish}
                onEdit={editPost}
                onGenerateChart={generateChart}
              />
            ))}
          </div>
        </div>
      )}

      {/* Launch Queue Preview */}
      <LaunchQueueSection fetchWithAuth={fetchWithAuth} onQueued={() => Promise.all([fetchStats(), fetchPosts()])} />
    </div>
  );
}

const LAUNCH_POSTS = [
  {
    id: 'launch-1',
    label: 'Launch Announcement',
    twitter: {
      text: "We're live.\n\nRigaCap is an AI-powered ensemble trading system built on 5 years of walk-forward testing.\n\n3 factors. 7 market regimes. Zero hindsight bias.\n\nThe signal is clear.",
      hashtags: '#trading #algotrading #stockmarket #fintech',
    },
    instagram: {
      text: "We're live.\n\nRigaCap is an AI-powered ensemble trading system that combines DWAP timing, momentum quality, and adaptive regime detection into one institutional-grade platform.\n\nBuilt on 15 years of development. Validated across 5 years of walk-forward testing with zero hindsight bias.\n\nThe signal is clear.",
      hashtags: '#trading #algotrading #stockmarket #fintech #investing #daytrading #momentum #rigacap',
    },
  },
  {
    id: 'launch-2',
    label: 'Performance Stats',
    twitter: {
      text: "5-year walk-forward results (no hindsight bias):\n\n2021-22: +62.0%\n2022-23: -13.2%\n2023-24: +22.2%\n2024-25: +20.7%\n2025-26: +87.5%\n\nTotal: +289% | 31% annualized\n80% win rate across all market conditions.",
      hashtags: '#trading #performance #walkforward',
    },
    instagram: {
      text: "5 years. No hindsight bias. Real walk-forward results.\n\n2021-2022: +62.0% (Sharpe 1.21)\n2022-2023: -13.2% (the bear year)\n2023-2024: +22.2% (Sharpe 1.02)\n2024-2025: +20.7% (Sharpe 0.89)\n2025-2026: +87.5% (Sharpe 2.32)\n\n5-Year Total: +289%\n31% annualized return\n80% win rate (4 of 5 years positive)\nMax drawdown: -15.1%\n\nEvery period tested independently. No curve-fitting. No cherry-picking.",
      hashtags: '#trading #algotrading #performance #walkforward #stockmarket #investing #returns #rigacap',
    },
  },
  {
    id: 'launch-3',
    label: 'How It Works',
    twitter: {
      text: "How our Ensemble finds trades:\n\n1. DWAP Timing \u2014 catches breakouts early\n2. Momentum Quality \u2014 only top-ranked stocks\n3. Regime Filter \u2014 adapts across 7 market conditions\n\nAll 3 must align = high-conviction entry.\n12% trailing stops protect gains.",
      hashtags: '#trading #ensemble #momentum #riskmanagement',
    },
    instagram: {
      text: "The 3-factor Ensemble approach.\n\nFactor 1: DWAP Timing\nPrice > Daily Weighted Average Price catches early breakouts before the crowd sees them.\n\nFactor 2: Momentum Quality\nTop composite momentum ranking (10-day + 60-day) filtered for low volatility and volume confirmation.\n\nFactor 3: Regime Detection\n7 distinct market regimes detected daily. The system adapts position sizing and entry rules from Strong Bull all the way to Panic/Crash.\n\nAll 3 factors must align for entry. 12% trailing stops protect gains while letting winners run.",
      hashtags: '#trading #algotrading #ensemble #momentum #dwap #riskmanagement #marketregime #rigacap',
    },
  },
  {
    id: 'launch-4',
    label: '7 Market Regimes',
    twitter: {
      text: "Most strategies have one mode. Ours has 7.\n\nStrong Bull \u2192 full exposure\nWeak Bull \u2192 selective entries\nRotating Bull \u2192 follow momentum\nRange Bound \u2192 reduce size\nWeak Bear \u2192 tighten stops\nPanic/Crash \u2192 exit all\nRecovery \u2192 scale back in\n\nAdapt or get left behind.",
      hashtags: '#trading #marketregime #riskmanagement',
    },
    instagram: {
      text: "Most strategies have one mode. Ours detects 7.\n\nOur AI analyzes SPY price action, market breadth, and volatility daily to classify the current environment:\n\nStrong Bull \u2014 Broad rally, high breadth. Full exposure.\nWeak Bull \u2014 Narrow leadership. Selective entries only.\nRotating Bull \u2014 Sector rotation active. Follow momentum leaders.\nRange Bound \u2014 Choppy, no trend. Reduce position size.\nWeak Bear \u2014 Drifting lower. Tighten trailing stops.\nPanic / Crash \u2014 Sharp sell-off. Exit all positions.\nRecovery \u2014 Bottom reversal confirmed. Scale back in.\n\nThe market changes. Your strategy should too.",
      hashtags: '#trading #algotrading #marketregime #riskmanagement #investing #stockmarket #rigacap',
    },
  },
  {
    id: 'launch-5',
    label: 'Signal Teaser',
    twitter: {
      text: "Today's top Ensemble signals are in.\n\nReal-time buy signals. Momentum scores. Regime detection. Trailing stop alerts.\n\nSubscribers see the full list every evening.\n\nrigacap.com",
      hashtags: '#trading #signals #stockmarket',
    },
    instagram: {
      text: "Today's top Ensemble signals are in.\n\nEvery evening, our system scans 6,500+ stocks and identifies the highest-conviction opportunities using our 3-factor ensemble approach.\n\nSubscribers get:\n\u2022 Real-time buy signals with entry levels\n\u2022 Composite momentum scores\n\u2022 Current market regime classification\n\u2022 Trailing stop alerts\n\u2022 Daily email digest\n\nWant to see today's signals?\n\nLink in bio.",
      hashtags: '#trading #signals #stockmarket #algotrading #investing #daytrading #momentum #rigacap',
    },
  },
];

function LaunchQueueSection({ fetchWithAuth, onQueued }) {
  const [expanded, setExpanded] = useState(true);
  const [queueing, setQueueing] = useState(false);
  const [queued, setQueued] = useState(false);

  const noop = () => {};

  const queueAllPosts = async () => {
    if (!window.confirm('Queue all 10 launch posts (5 Twitter + 5 Instagram) as drafts?')) return;
    setQueueing(true);
    let success = 0;
    for (const lp of LAUNCH_POSTS) {
      for (const platform of ['twitter', 'instagram']) {
        const content = platform === 'twitter' ? lp.twitter : lp.instagram;
        try {
          const res = await fetchWithAuth(`${API_URL}/api/admin/social/posts/compose`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              platform,
              text_content: content.text,
              hashtags: content.hashtags,
              post_type: 'manual',
              status: 'draft',
            }),
          });
          if (res.ok) success++;
        } catch (err) {
          console.error(`Failed to queue ${lp.label} (${platform}):`, err);
        }
      }
    }
    setQueueing(false);
    setQueued(true);
    alert(`Queued ${success} of 10 launch posts as drafts.`);
    if (onQueued) onQueued();
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-6 py-4 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Rocket size={18} className="text-amber-500" />
          <span className="text-sm font-semibold text-gray-800">Launch Posts Preview</span>
          <span className="text-xs text-gray-400">5 concepts &times; 2 platforms = 10 posts</span>
        </div>
        {expanded ? <ChevronUp size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
      </button>

      {expanded && (
        <div className="border-t border-gray-200 p-6 space-y-8">
          {/* Queue button */}
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-500">Preview how launch posts will look. When ready, queue them all as drafts to review and approve individually.</p>
            <button
              onClick={queueAllPosts}
              disabled={queueing || queued}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                queued
                  ? 'bg-green-100 text-green-700 cursor-default'
                  : 'bg-amber-500 hover:bg-amber-600 text-white disabled:opacity-50'
              }`}
            >
              {queueing ? (
                <><RefreshCw size={14} className="animate-spin" /> Queueing...</>
              ) : queued ? (
                <><Check size={14} /> Queued!</>
              ) : (
                <><Plus size={14} /> Queue All as Drafts</>
              )}
            </button>
          </div>

          {LAUNCH_POSTS.map((lp, idx) => {
            const twitterPost = {
              id: lp.id + '-tw',
              post_type: 'manual',
              platform: 'twitter',
              status: 'draft',
              text_content: lp.twitter.text,
              hashtags: lp.twitter.hashtags,
              created_at: new Date().toISOString(),
            };
            const instagramPost = {
              id: lp.id + '-ig',
              post_type: 'manual',
              platform: 'instagram',
              status: 'draft',
              text_content: lp.instagram.text,
              hashtags: lp.instagram.hashtags,
              created_at: new Date().toISOString(),
            };

            return (
              <div key={lp.id}>
                <div className="flex items-center gap-2 mb-3">
                  <span className="flex items-center justify-center w-6 h-6 rounded-full bg-amber-100 text-amber-700 text-xs font-bold">{idx + 1}</span>
                  <h4 className="text-sm font-semibold text-gray-700">{lp.label}</h4>
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <TwitterCard
                    post={twitterPost}
                    preview={null}
                    actionLoading={null}
                    onApprove={noop}
                    onReject={noop}
                    onRegenerate={noop}
                    onDelete={noop}
                    onPublish={noop}
                    onEdit={noop}
                    publishingLive={false}
                  />
                  <InstagramCard
                    post={instagramPost}
                    preview={null}
                    actionLoading={null}
                    onApprove={noop}
                    onReject={noop}
                    onRegenerate={noop}
                    onDelete={noop}
                    onPublish={noop}
                    onEdit={noop}
                    onGenerateChart={noop}
                    publishingLive={false}
                  />
                </div>
                {idx < LAUNCH_POSTS.length - 1 && <hr className="mt-6 border-gray-100" />}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ComposeModal({ onClose, onSubmit }) {
  const [composePlatform, setComposePlatform] = useState('twitter');
  const [text, setText] = useState('');
  const [hashtags, setHashtags] = useState('');
  const [saving, setSaving] = useState(false);

  const fullText = hashtags ? `${text}\n\n${hashtags}` : text;
  const charCount = fullText.length;
  const overLimit = composePlatform === 'twitter' && charCount > 280;

  const handleSubmit = async (saveStatus) => {
    if (!text.trim()) return;
    setSaving(true);
    await onSubmit({
      platform: composePlatform,
      text_content: text,
      hashtags: hashtags || null,
      post_type: 'manual',
      status: saveStatus,
    });
    setSaving(false);
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-gray-900">New Post</h2>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600"><X size={20} /></button>
          </div>
        </div>

        <div className="p-6 space-y-5">
          {/* Platform Toggle */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Platform</label>
            <div className="flex gap-2">
              {['twitter', 'instagram'].map(p => (
                <button
                  key={p}
                  onClick={() => setComposePlatform(p)}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                    composePlatform === p
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {p === 'twitter' ? 'Twitter/X' : 'Instagram'}
                </button>
              ))}
            </div>
          </div>

          {/* Text Area */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Post Text</label>
            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              rows={5}
              className="w-full border border-gray-300 rounded-lg p-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              placeholder={composePlatform === 'twitter' ? "What's happening?" : "Write a caption..."}
            />
          </div>

          {/* Hashtags */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Hashtags</label>
            <input
              type="text"
              value={hashtags}
              onChange={e => setHashtags(e.target.value)}
              className="w-full border border-gray-300 rounded-lg p-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="#trading #momentum #stocks"
            />
          </div>

          {/* Preview */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Preview</label>
            {composePlatform === 'twitter' ? (
              <div className="border border-gray-200 rounded-xl p-4">
                <div className="flex gap-3">
                  <RigaCapLogo size={40} className="shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1">
                      <span className="font-bold text-sm text-gray-900">RigaCap</span>
                      <span className="text-sm text-gray-500">@rigacap</span>
                    </div>
                    <p className="text-sm text-gray-900 mt-1 whitespace-pre-wrap break-words">{text || 'Your post text here...'}</p>
                    {hashtags && <p className="text-sm text-blue-500 mt-1 break-words">{hashtags}</p>}
                  </div>
                </div>
              </div>
            ) : (
              <div className="border border-gray-200 rounded-xl overflow-hidden">
                <div className="flex items-center gap-2.5 px-3 py-2.5 border-b border-gray-100">
                  <RigaCapLogo size={32} className="shrink-0" />
                  <span className="font-semibold text-sm text-gray-900">rigacap</span>
                </div>
                <div className="w-full aspect-square bg-gray-50 flex items-center justify-center">
                  <Image size={40} className="text-gray-300" />
                </div>
                <div className="px-3 py-2.5">
                  <p className="text-sm text-gray-900">
                    <span className="font-semibold">rigacap</span>{' '}
                    <span className="whitespace-pre-wrap break-words">{text || 'Your caption here...'}</span>
                  </p>
                  {hashtags && <p className="text-sm text-blue-500 mt-1 break-words">{hashtags}</p>}
                </div>
              </div>
            )}
          </div>

          {/* Char count for Twitter */}
          {composePlatform === 'twitter' && (
            <div className={`text-xs font-medium ${overLimit ? 'text-red-600' : 'text-gray-400'}`}>
              {charCount}/280
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-200 flex items-center justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-800"
          >
            Cancel
          </button>
          <button
            onClick={() => handleSubmit('draft')}
            disabled={!text.trim() || saving}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
          >
            Save as Draft
          </button>
          <button
            onClick={() => handleSubmit('approved')}
            disabled={!text.trim() || saving || overLimit}
            className="px-4 py-2 text-sm font-medium text-white bg-green-600 hover:bg-green-700 rounded-lg transition-colors disabled:opacity-50"
          >
            Save & Approve
          </button>
        </div>
      </div>
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

function ActionButtons({ post, actionLoading, onApprove, onReject, onRegenerate, onDelete, onPublish, publishingLive, extraButtons }) {
  const isLoading = !!actionLoading;
  const canModify = post.status === 'draft' || post.status === 'rejected';

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {post.status === 'approved' && publishingLive && (
        <button
          onClick={() => onPublish(post.id)}
          disabled={isLoading}
          className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50"
          title="Publish to platform"
        >
          {actionLoading === 'publish' ? <RefreshCw size={13} className="animate-spin" /> : <Send size={13} />}
          Publish
        </button>
      )}
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
      {post.status !== 'posted' && (
        <button
          onClick={() => onRegenerate(post.id)}
          disabled={isLoading}
          className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
          title="Regenerate"
        >
          {actionLoading === 'regenerate' ? <RefreshCw size={13} className="animate-spin" /> : <RefreshCw size={13} />}
          Regen
        </button>
      )}
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

function InlineEditableText({ text, hashtags, postId, canEdit, onEdit }) {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(text);
  const [editHashtags, setEditHashtags] = useState(hashtags);
  const [saving, setSaving] = useState(false);

  if (!editing) {
    return (
      <div
        className={`group relative ${canEdit ? 'cursor-pointer' : ''}`}
        onClick={() => { if (canEdit) { setEditText(text); setEditHashtags(hashtags); setEditing(true); } }}
      >
        <p className="text-sm text-gray-900 whitespace-pre-wrap break-words">{text}</p>
        {hashtags && <p className="text-sm text-blue-500 mt-1 break-words">{hashtags}</p>}
        {canEdit && (
          <div className="absolute top-0 right-0 opacity-0 group-hover:opacity-100 transition-opacity">
            <Edit3 size={12} className="text-gray-400" />
          </div>
        )}
      </div>
    );
  }

  const handleSave = async () => {
    setSaving(true);
    const ok = await onEdit(postId, editText, editHashtags);
    setSaving(false);
    if (ok) setEditing(false);
  };

  return (
    <div className="space-y-2">
      <textarea
        value={editText}
        onChange={e => setEditText(e.target.value)}
        rows={3}
        className="w-full border border-blue-300 rounded-lg p-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
        autoFocus
      />
      <input
        type="text"
        value={editHashtags}
        onChange={e => setEditHashtags(e.target.value)}
        className="w-full border border-blue-300 rounded-lg p-2 text-sm text-blue-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        placeholder="Hashtags"
      />
      <div className="flex gap-2">
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg disabled:opacity-50"
        >
          <Save size={12} /> Save
        </button>
        <button
          onClick={() => setEditing(false)}
          className="px-2.5 py-1 text-xs font-medium text-gray-600 hover:text-gray-800"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

function TwitterCard({ post, preview, actionLoading, onApprove, onReject, onRegenerate, onDelete, onPublish, onEdit, publishingLive }) {
  const { mainText, tags } = splitTextAndHashtags(post.text_content, post.hashtags);
  const fullText = tags ? `${mainText}\n\n${tags}` : mainText;
  const charCount = preview?.char_count ?? fullText.length;
  const overLimit = preview?.over_limit ?? charCount > 280;
  const canEdit = post.status === 'draft' || post.status === 'approved';

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      <div className="p-4 space-y-3">
        <PostBadges post={post} />

        {/* Mock Tweet */}
        <div className="border border-gray-200 rounded-xl p-4">
          <div className="flex gap-3">
            <RigaCapLogo size={40} className="shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1">
                <span className="font-bold text-sm text-gray-900">RigaCap</span>
                <span className="text-sm text-gray-500">@rigacap</span>
              </div>
              <div className="mt-1">
                <InlineEditableText
                  text={mainText}
                  hashtags={tags}
                  postId={post.id}
                  canEdit={canEdit}
                  onEdit={onEdit}
                />
              </div>
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
            onPublish={onPublish}
            publishingLive={publishingLive}
          />
        </div>
      </div>
    </div>
  );
}

function InstagramCard({ post, preview, actionLoading, onApprove, onReject, onRegenerate, onDelete, onPublish, onEdit, onGenerateChart, publishingLive }) {
  const { mainText, tags } = splitTextAndHashtags(post.text_content, post.hashtags);
  const imageUrl = preview?.image_url || null;
  const hasImage = !!post.image_s3_key || !!imageUrl;
  const chartLoading = actionLoading === 'generate-chart';
  const canGenerateChart = post.post_type === 'trade_result' || post.post_type === 'missed_opportunity';
  const canEdit = post.status === 'draft' || post.status === 'approved';

  const generateChartButton = canGenerateChart && !hasImage && (
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
            <RigaCapLogo size={32} className="shrink-0" />
            <span className="font-semibold text-sm text-gray-900">rigacap</span>
          </div>

          {/* Image Area — only show if there's an image or the post type supports chart generation */}
          {imageUrl ? (
            <img
              src={imageUrl}
              alt="Chart card"
              className="w-full aspect-square object-cover bg-gray-100"
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          ) : (canGenerateChart || post.image_s3_key) && (
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
            <div className="text-sm text-gray-900">
              <span className="font-semibold">rigacap</span>{' '}
              <InlineEditableText
                text={mainText}
                hashtags={tags}
                postId={post.id}
                canEdit={canEdit}
                onEdit={onEdit}
              />
            </div>
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
            onPublish={onPublish}
            publishingLive={publishingLive}
            extraButtons={generateChartButton}
          />
        </div>
      </div>
    </div>
  );
}
