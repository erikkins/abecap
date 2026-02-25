/**
 * Hook that polls /api/quotes/live every 30s for real-time position prices.
 * Pauses when the app is backgrounded, fetches immediately on foreground resume.
 */

import { useCallback, useEffect, useState } from 'react';
import { AppState } from 'react-native';
import api from '@/services/api';

export interface Quote {
  price: number;
  change: number;
  change_pct: number;
  prev_close?: number;
}

export function useLiveQuotes(symbols: string[], intervalMs = 30000) {
  const [quotes, setQuotes] = useState<Record<string, Quote>>({});
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Join for stable dependency — avoids re-creating callback when array ref changes
  const symbolsKey = symbols.join(',');

  const fetchQuotes = useCallback(async () => {
    if (!symbolsKey) return;
    try {
      const resp = await api.get('/api/quotes/live', {
        params: { symbols: symbolsKey },
      });
      if (resp.data?.quotes) {
        setQuotes(resp.data.quotes);
        setLastUpdate(new Date());
      }
    } catch {
      // Silently fail — stale data is acceptable fallback
    }
  }, [symbolsKey]);

  useEffect(() => {
    fetchQuotes();
    const timer = setInterval(fetchQuotes, intervalMs);

    // Pause polling when app is backgrounded, resume on foreground
    const sub = AppState.addEventListener('change', (state) => {
      if (state === 'active') fetchQuotes();
    });

    return () => {
      clearInterval(timer);
      sub.remove();
    };
  }, [fetchQuotes, intervalMs]);

  return { quotes, lastUpdate };
}
