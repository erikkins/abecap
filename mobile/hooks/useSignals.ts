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
  symbol: string;
  shares: number;
  entry_price: number;
  entry_date: string;
  current_price: number;
  highest_price: number;
  pnl_pct: number;
  sell_guidance: string;
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
  outlook: string;
  recommended_action: string;
  risk_change: string;
  probabilities: Record<string, number>;
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
