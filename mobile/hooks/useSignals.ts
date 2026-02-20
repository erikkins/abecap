/**
 * Hook for fetching dashboard / signal data from the API.
 */

import { useCallback, useEffect, useState } from 'react';
import api from '@/services/api';

export interface Signal {
  symbol: string;
  price: number;
  pct_above_dwap: number;
  is_strong: boolean;
  momentum_rank: number;
  ensemble_score: number;
  dwap_crossover_date: string | null;
  ensemble_entry_date: string | null;
  days_since_crossover: number | null;
  days_since_entry: number | null;
  is_fresh: boolean;
}

export interface Position {
  id: number;
  symbol: string;
  shares: number;
  entry_price: number;
  entry_date: string;
  current_price: number;
  highest_price: number;
  pnl_pct: number;
  sell_guidance: string;
}

export interface Trade {
  id: number;
  symbol: string;
  entry_date: string;
  entry_price: number;
  exit_date: string;
  exit_price: number;
  shares: number;
  pnl: number;
  pnl_pct: number;
  exit_reason: string;
}

export interface MissedOpportunity {
  symbol: string;
  entry_date: string;
  entry_price: number;
  sell_date: string;
  sell_price: number;
  would_be_return: number;
  days_held: number;
  exit_reason: string;
}

export interface RegimeForecast {
  current_regime: string;
  current_regime_name: string;
  outlook: string;
  outlook_detail: string;
  recommended_action: string;
  risk_change: string;
  probabilities: Record<string, number>;
  transition_probabilities: Record<string, number>;
}

export interface DashboardData {
  buy_signals: Signal[];
  market_stats: {
    spy_price: number;
    spy_change_pct: number;
    vix_level: number;
  };
  regime_forecast: RegimeForecast;
  model_portfolio?: {
    total_value: number;
    total_return_pct: number;
    positions: Array<{
      symbol: string;
      entry_price: number;
      current_price: number;
      pnl_pct: number;
    }>;
  };
  positions_with_guidance?: Position[];
  missed_opportunities?: MissedOpportunity[];
}

export function useDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const resp = await api.get('/api/signals/dashboard');
      setData(resp.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load dashboard');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, isLoading, error, refresh: fetch };
}

export async function trackSignal(symbol: string, price: number) {
  return api.post('/api/portfolio/positions', { symbol, price });
}

export async function sellPosition(positionId: number, exitPrice: number) {
  return api.delete(`/api/portfolio/positions/${positionId}?exit_price=${exitPrice}`);
}

export function useTradeHistory() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const resp = await api.get('/api/portfolio/trades?limit=50');
      setTrades(resp.data.trades || []);
      setTotal(resp.data.total || 0);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load trade history');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { trades, total, isLoading, error, refresh: fetch };
}
