'use client';

import { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';

const ChatContext = createContext(null);
const STORAGE_KEY = 'ragworkspace:activeChat';
const CHAT_TIMEOUT_MS = 15 * 60 * 1000;
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export function ChatProvider({ token, children }) {
  const [messages, setMessages] = useState(() => {
    if (typeof window === 'undefined') return [];
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved).messages || [] : [];
    } catch {
      return [];
    }
  });
  const [loading, setLoading] = useState(false);
  const abortRef = useRef(null);

  // Persist on every change — survives refresh, tab close, etc.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ messages, savedAt: Date.now() }));
    } catch {}
  }, [messages]);

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || loading) return;
    setMessages(p => [...p, { role: 'user', content: text }]);
    setLoading(true);
    abortRef.current = new AbortController();
    const tid = setTimeout(() => abortRef.current?.abort(), CHAT_TIMEOUT_MS);
    try {
      const res = await fetch(`${API_URL}/api/v1/agents/chat`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, top_k: 5 }),
        signal: abortRef.current.signal,
      });
      clearTimeout(tid);
      if (!res.ok) {
        const e = await res.json().catch(() => null);
        throw new Error(e?.detail || `Error ${res.status}`);
      }
      const data = await res.json();
      setMessages(p => [...p, {
        role: 'assistant',
        content: data.answer,
        confidence: data.confidence,
        sources: data.sources || [],
        searchTimeMs: data.search_time_ms,
      }]);
    } catch (err) {
      clearTimeout(tid);
      setMessages(p => [...p, {
        role: 'assistant',
        content: err.name === 'AbortError' ? 'Request timed out after 15 minutes.' : `Error: ${err.message}`,
      }]);
    } finally {
      setLoading(false);
    }
  }, [token, loading]);

  const cancelMessage = useCallback(() => abortRef.current?.abort(), []);

  // ONLY this clears the chat. Nothing else should call it.
  const newChat = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setLoading(false);
    try { localStorage.removeItem(STORAGE_KEY); } catch {}
  }, []);

  return (
    <ChatContext.Provider value={{ messages, loading, sendMessage, cancelMessage, newChat }}>
      {children}
    </ChatContext.Provider>
  );
}

export const useChat = () => {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error('useChat must be used inside ChatProvider');
  return ctx;
};