/**
 * Hook for fetching price chart data from the API.
 */

import { useCallback, useEffect, useState } from 'react';
import api from '@/services/api';

export interface ChartPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  dwap: number | null;
  ma_50: number | null;
  ma_200: number | null;
}

export function useChartData(symbol: string, days: number = 180) {
  const [data, setData] = useState<ChartPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const resp = await api.get(`/api/stock/${symbol}/history`, {
        params: { days },
      });
      setData(resp.data.data || []);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load chart data');
    } finally {
      setIsLoading(false);
    }
  }, [symbol, days]);

  useEffect(() => {
    if (symbol) fetch();
  }, [fetch, symbol]);

  return { data, isLoading, error, refresh: fetch };
}
