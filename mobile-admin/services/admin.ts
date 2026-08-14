/**
 * Admin API — typed wrappers over /api/admin/* (and a couple public endpoints).
 *
 * Shapes mirror backend/app/api/admin.py response models. Endpoints that
 * aren't yet strongly typed return `any` and are read defensively in the UI.
 */

import api from './api';

// ── Growth / users ────────────────────────────────────────────────
export interface AdminStats {
  total_users: number;
  active_trials: number;
  paid_subscribers: number;
  expired_trials: number;
  disabled_users: number;
  new_users_today: number;
  new_users_week: number;
  mrr: number;
  comped_subscribers?: number;
}

export interface UserSummary {
  id: string;
  email: string;
  name: string | null;
  role: string;
  is_active: boolean;
  created_at: string;
  last_login: string | null;
  subscription_status: string | null;
  trial_days_remaining: number | null;
  is_founding: boolean;
}

export interface UserList {
  users: UserSummary[];
  total: number;
  page: number;
  per_page: number;
}

export async function getStats(): Promise<AdminStats> {
  const { data } = await api.get('/api/admin/stats');
  return data;
}

export async function getUsers(page = 1, perPage = 50): Promise<UserList> {
  const { data } = await api.get('/api/admin/users', {
    params: { page, per_page: perPage },
  });
  return data;
}

// ── Pipeline / service health ─────────────────────────────────────
export interface ServiceStatus {
  overall_status: string;
  services: Record<string, any>;
  metrics: Record<string, any>;
}

export async function getServiceStatus(): Promise<ServiceStatus> {
  const { data } = await api.get('/api/admin/service-status');
  return data;
}

// ── Live intraday quotes (daily change) ───────────────────────────
export interface LiveQuote {
  price: number;
  change: number; // $ change vs prev close
  change_pct: number; // % change vs prev close
  prev_close?: number;
}

// Same source the subscriber app uses (/api/quotes/live). Returns a map keyed
// by symbol. Empty object on no symbols.
export async function getLiveQuotes(symbols: string[]): Promise<Record<string, LiveQuote>> {
  if (!symbols.length) return {};
  const { data } = await api.get('/api/quotes/live', {
    params: { symbols: symbols.join(',') },
  });
  return (data?.quotes as Record<string, LiveQuote>) || {};
}

// ── Live model portfolio ──────────────────────────────────────────
export async function getModelPortfolio(): Promise<any> {
  // Without portfolio_type the API returns { live, walkforward } nested — request
  // the live book so we get a flat object with positions/value/cash directly.
  const { data } = await api.get('/api/admin/model-portfolio', {
    params: { portfolio_type: 'live' },
  });
  return data;
}

export async function getCurrentRegime(): Promise<any> {
  const { data } = await api.get('/api/admin/market-regime/current');
  return data;
}

// ── Founding-seat counter (public endpoint) ───────────────────────
export interface FoundingStatus {
  seats_taken?: number;
  seats_total?: number;
  seats_remaining?: number;
  price?: number;
  is_open?: boolean;
  [k: string]: any;
}

export async function getFoundingStatus(): Promise<FoundingStatus> {
  const { data } = await api.get('/api/billing/founding-status');
  return data;
}

// ── Ads summary (milestone 2 — backend endpoint not built yet) ─────
// Returns null if the endpoint doesn't exist yet (404), so the Ads tab can
// render a "not configured" state without crashing.
export interface AdsSummary {
  spend?: number;
  clicks?: number;
  impressions?: number;
  conversions?: number;
  cpc?: number;
  date_range?: string;
  campaigns?: Array<Record<string, any>>;
  [k: string]: any;
}

export async function getAdsSummary(): Promise<AdsSummary | null> {
  try {
    const { data } = await api.get('/api/admin/ads/summary');
    return data;
  } catch (err: any) {
    if (err?.response?.status === 404) return null;
    throw err;
  }
}

// ── Tier books (Preserver + Maximizer) + Cascade Guard ────────────
// GET /api/admin/tier-books — the served books, mobilized. Core is ignored by
// the app (Erik follows Preserver + Maximizer). `fills` carry the per-name detail
// (open buys with unrealized=true = current holdings, with current_price + pnl_pct
// + days_held); `books[tier].holdings` are {symbol,shares,eod_price} for repricing.
export interface TierHolding {
  symbol: string;
  shares: number;
  eod_price?: number;
}
export interface TierFillRow {
  fill_date: string | null;
  symbol: string;
  side: string;
  shares?: number;
  price?: number; // entry (buy) or exit (sell)
  current_price?: number | null;
  pnl_pct?: number | null;
  gross?: number;
  source?: string;
  reason?: string;
  days_held?: number | null;
  realized_pnl?: number | null;
  unrealized?: boolean;
}
export interface TierBook {
  equity?: number | null;
  as_of?: string | null;
  regime?: string | null;
  held?: number | null;
  note?: string;
  holdings?: TierHolding[];
}
export interface CascadeGuard {
  enabled?: boolean;
  paused?: boolean;
  pause_until?: string | null;
  pause_source?: string | null;
  last_triggered_at?: string | null;
  threshold_stops?: number;
  pause_days?: number;
  last_stopped_symbols?: string[];
}
export interface TierBooks {
  books: { core?: TierBook; preserver?: TierBook; maximizer?: TierBook };
  fills: { core?: TierFillRow[]; preserver?: TierFillRow[]; maximizer?: TierFillRow[] };
  cascade_guard?: CascadeGuard | null;
}
export async function getTierBooks(limit = 60): Promise<TierBooks> {
  const { data } = await api.get('/api/admin/tier-books', { params: { limit } });
  return data;
}

// ── Cookieless traffic + /should-i-sell conversion funnel ─────────
// GET /api/admin/pageviews/summary — first-party, consent-free traffic.
export interface FunnelStep {
  step: string;
  label: string;
  count: number;
}
export interface TrafficSummary {
  days: number;
  total: number;
  paid_clicks: number;
  mobile: number;
  by_path: Array<{ path: string; count: number }>;
  by_source: Array<{ source: string; count: number }>;
  by_day: Array<{ date: string; count: number }>;
  by_event?: Array<{ event: string; count: number }>;
  sis_funnel?: FunnelStep[]; // /should-i-sell funnel; first step (Landed) = denominator
}
export async function getTrafficSummary(days = 7): Promise<TrafficSummary> {
  const { data } = await api.get('/api/admin/pageviews/summary', { params: { days } });
  return data;
}

// ── Social queue (read-only follow-along) ─────────────────────────
// GET /api/admin/social/posts?status=draft|scheduled — approve/kill happen via email.
export interface SocialPost {
  id: number;
  post_type: string;
  platform: string;
  status: string;
  text_content?: string | null;
  scheduled_for?: string | null;
  created_at?: string | null;
  reply_to_username?: string | null;
  ai_generated?: boolean;
  tier?: string | null;
  [k: string]: any;
}
export async function getSocialPosts(status: string, limit = 50): Promise<SocialPost[]> {
  const { data } = await api.get('/api/admin/social/posts', { params: { status, limit } });
  return (data?.posts as SocialPost[]) || [];
}
