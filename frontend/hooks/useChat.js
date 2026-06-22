// frontend/hooks/useChat.js
'use client';

import { useState } from 'react';
import { apiPost } from '@/lib/api';

// Manages the chat conversation state: the message list, sending a
// new message, and the loading state while the backend's agent
// pipeline is thinking (this can take a while - see note below).
//
// NOTE: the backend's agent pipeline (Planner -> Retriever -> Tool ->
// Answer -> Critic) can take a long time on local hardware - we've
// seen it take anywhere from a few seconds up to ~100+ seconds. There
// is no streaming yet, so the UI just shows a "thinking" state for
// however long the single request takes. Don't add a short timeout
// here or real responses will get cut off.
export function useChat() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const sendMessage = async (text) => {
    if (!text.trim()) return;

    const userMessage = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMessage]);
    setError('');
    setLoading(true);

    try {
      const data = await apiPost('/agents/chat', { message: text });

      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
        confidence: data.confidence,
        searchTimeMs: data.search_time_ms,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(err.message || 'Failed to get a response');
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, something went wrong answering that.', isError: true },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const clearMessages = () => setMessages([]);

  return { messages, loading, error, sendMessage, clearMessages };
}