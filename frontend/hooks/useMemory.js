// frontend/hooks/useMemory.js
'use client';

import { useState, useEffect, useCallback } from 'react';
import { apiGet } from '@/lib/api';

// IMPORTANT: This hook assumes the memory endpoints follow the same
// pattern as the rest of the API (/api/v1/memory/history,
// /api/v1/memory/summaries), based on the Phase 6 handoff doc.
// These were NOT verified against a running backend in this session -
// if they 404, check the real route names in
// backend/app/api/v1/endpoints/memory.py and update the paths below.
export function useMemory() {
  const [history, setHistory] = useState([]);
  const [summaries, setSummaries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchMemory = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [historyData, summaryData] = await Promise.all([
        apiGet('/memory/history'),
        apiGet('/memory/summaries'),
      ]);
      setHistory(historyData.messages || historyData.history || []);
      setSummaries(summaryData.summaries || []);
    } catch (err) {
      setError(err.message || 'Failed to load memory');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMemory();
  }, [fetchMemory]);

  return { history, summaries, loading, error, refresh: fetchMemory };
}