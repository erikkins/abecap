/**
 * Hook for fetching company info (name, sector, industry, description).
 */

import { useEffect, useState } from 'react';
import api from '@/services/api';

export interface StockInfo {
  name: string;
  sector: string;
  industry: string;
  description: string;
  market_cap: string;
}

export function useStockInfo(symbol: string) {
  const [info, setInfo] = useState<StockInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!symbol) return;
    let cancelled = false;
    setIsLoading(true);

    api
      .get(`/api/signals/info/${symbol}`)
      .then(({ data }) => {
        if (!cancelled) setInfo(data);
      })
      .catch(() => {
        if (!cancelled) setInfo(null);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  return { info, isLoading };
}
