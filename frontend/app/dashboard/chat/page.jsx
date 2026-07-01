'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useAuth } from '@/app/context/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const CHAT_TIMEOUT_MS = 15 * 60 * 1000; // 15 minutes — matches Ollama on CPU

// NOTE: Your backend's /api/v1/agents/chat is currently stateless — one
// message in, one answer out, nothing saved server-side. There is no
// session concept yet, so this page is a single ongoing conversation
// (cleared on refresh) rather than a multi-conversation history rail.
// If/when session persistence is added on the backend, the history rail
// can come back — this was deliberately simplified for now.

const INPUT_MIN_HEIGHT = 44; // px — matches old single-line input height
const INPUT_MAX_HEIGHT = 160; // px — caps growth so the composer doesn't eat the thread

// Mirrors the auto-resize behavior from the v0 chat pattern, trimmed down
// (no min/max props plumbing) since this page only ever needs one size.
function useAutoResizeTextarea() {
  const textareaRef = useRef(null);

  const adjustHeight = useCallback((reset) => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    if (reset) {
      textarea.style.height = `${INPUT_MIN_HEIGHT}px`;
      return;
    }

    textarea.style.height = `${INPUT_MIN_HEIGHT}px`;
    const newHeight = Math.max(
      INPUT_MIN_HEIGHT,
      Math.min(textarea.scrollHeight, INPUT_MAX_HEIGHT)
    );
    textarea.style.height = `${newHeight}px`;
  }, []);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = `${INPUT_MIN_HEIGHT}px`;
    }
  }, []);

  useEffect(() => {
    const handleResize = () => adjustHeight();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [adjustHeight]);

  return { textareaRef, adjustHeight };
}

export default function ChatPage() {
  const { token } = useAuth();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [sourcesOpen, setSourcesOpen] = useState(true);

  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);
  const timerRef = useRef(null);
  const { textareaRef, adjustHeight } = useAutoResizeTextarea();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (loading) {
      setElapsedSeconds(0);
      timerRef.current = setInterval(() => setElapsedSeconds((s) => s + 1), 1000);
    } else {
      clearInterval(timerRef.current);
      setElapsedSeconds(0);
    }
    return () => clearInterval(timerRef.current);
  }, [loading]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    adjustHeight(true);
    setLoading(true);

    abortControllerRef.current = new AbortController();
    const timeoutId = setTimeout(() => abortControllerRef.current?.abort(), CHAT_TIMEOUT_MS);

    try {
      // Matches your real backend's ChatRequest shape exactly:
      // { message: str, top_k: int = 5 } — see app/api/v1/endpoints/agents.py
      const response = await fetch(`${API_URL}/api/v1/agents/chat`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: currentInput, top_k: 5 }),
        signal: abortControllerRef.current.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errBody = await response.json().catch(() => null);
        throw new Error(errBody?.detail || `Request failed (${response.status})`);
      }

      // Matches your real ChatResponse shape:
      // { answer, sources: list[dict], confidence, search_time_ms }
      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          confidence: data.confidence,
          sources: data.sources || [],
          searchTimeMs: data.search_time_ms,
        },
      ]);
    } catch (error) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: 'Request timed out after 15 minutes. Try a shorter question or wait.' },
        ]);
      } else {
        setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${error.message}` }]);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = (e) => {
    e.preventDefault();
    sendMessage();
  };

  // Enter sends, Shift+Enter inserts a newline — same convention as the
  // v0 composer, applied to your existing send flow.
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleCancel = () => {
    abortControllerRef.current?.abort();
    setLoading(false);
  };

  const clearChat = () => {
    if (messages.length === 0) return;
    if (!window.confirm('Clear this conversation? This cannot be undone since nothing is saved.')) return;
    setMessages([]);
  };

  const lastAssistantMessage = [...messages].reverse().find((m) => m.role === 'assistant');
  const currentSources = lastAssistantMessage?.sources || [];

  return (
    <div className="flex h-[calc(100vh-57px)]">
      {/* ---------- Conversation ---------- */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-between border-b border-white/10 px-4 py-3 lg:px-8">
          <div>
            <h1 className="text-[15px] font-medium text-white" style={{ fontFamily: 'Inter, sans-serif' }}>
              Chat
            </h1>
            <p className="text-[12px]" style={{ color: '#71717A' }}>
              Ask questions about your documents
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={clearChat}
              className="rounded-[8px] border border-white/10 px-3 py-1.5 text-[12px] text-[#A1A1AA] transition-colors hover:bg-white/5 hover:text-white"
            >
              Clear chat
            </button>
            <button
              onClick={() => setSourcesOpen((v) => !v)}
              className="rounded-[8px] border border-white/10 p-2 text-[#A1A1AA] transition-colors hover:text-white xl:hidden"
              aria-label="Toggle sources"
            >
              <SourcesIcon />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-6 lg:px-8">
          {messages.length === 0 && (
            <div className="mt-12 text-center" style={{ color: '#71717A' }}>
              <p className="text-[15px]">Ask something about your documents.</p>
            </div>
          )}
          <div className="mx-auto max-w-2xl">
            {messages.map((msg, idx) => (
              <div key={idx} className={`mb-4 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-md rounded-[11px] px-4 py-2 ${
                    msg.role === 'user' ? 'text-[#022C22]' : 'border border-white/10 bg-white/6 text-[#F4F4F5]'
                  }`}
                  style={msg.role === 'user' ? { background: '#34D399' } : undefined}
                >
                  <p className="whitespace-pre-wrap text-[14px]">{msg.content}</p>
                  {msg.confidence !== undefined && (
                    <p className="mt-1 text-[11px] opacity-70" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
                      Confidence: {(msg.confidence * 100).toFixed(0)}%
                      {msg.searchTimeMs !== undefined && ` • ${msg.searchTimeMs.toFixed(0)}ms`}
                    </p>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="mb-4 flex items-center gap-3">
                <div className="flex items-center gap-2 rounded-[11px] border border-white/10 bg-white/6 px-4 py-2 text-[#F4F4F5]">
                  <span className="animate-pulse" style={{ color: '#34D399' }}>●</span>
                  <span className="animate-pulse delay-100" style={{ color: '#34D399' }}>●</span>
                  <span className="animate-pulse delay-200" style={{ color: '#34D399' }}>●</span>
                  <span className="ml-2 text-[13px]" style={{ color: '#A1A1AA' }}>
                    Thinking… {elapsedSeconds}s
                  </span>
                </div>
                <button onClick={handleCancel} className="text-[13px] underline hover:text-white" style={{ color: '#F87171' }}>
                  Cancel
                </button>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ---------- Composer ---------- */}
        {/* Was a single-line <input>; now an auto-resizing <textarea> so
            multi-line questions don't get clipped. Same border/bg/focus
            tokens as before — just a taller, growable field. */}
        <form onSubmit={handleSendMessage} className="border-t border-white/10 p-4">
          <div className="mx-auto flex max-w-2xl items-end gap-2">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                adjustHeight();
              }}
              onKeyDown={handleKeyDown}
              placeholder="Ask me anything about your documents…"
              disabled={loading}
              className="flex-1 resize-none rounded-[11px] border border-white/10 bg-[#0F0F11]/60 px-4 py-2.5 text-[14px] text-white placeholder-[#52525B] outline-none transition-all duration-200 focus:border-[#34D399] focus:shadow-[0_0_0_3px_rgba(52,211,153,0.15)]"
              style={{ height: `${INPUT_MIN_HEIGHT}px`, overflow: 'hidden' }}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="shrink-0 rounded-[11px] px-6 py-2.5 text-[14px] font-medium text-[#022C22] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_8px_24px_rgba(52,211,153,0.35)] disabled:translate-y-0 disabled:opacity-50 disabled:hover:shadow-none"
              style={{ background: '#34D399' }}
            >
              {loading ? `${elapsedSeconds}s…` : 'Send'}
            </button>
          </div>
          <p className="mx-auto mt-1.5 max-w-2xl text-[11px]" style={{ color: '#52525B' }}>
            Shift + Enter for a new line
          </p>
        </form>
      </div>

      {/* ---------- Sources panel ---------- */}
      <aside
        className={`hidden shrink-0 flex-col border-l border-white/10 bg-[#18181B]/20 backdrop-blur-xl xl:flex ${
          sourcesOpen ? 'w-72' : 'w-0 overflow-hidden border-l-0'
        } transition-all duration-200`}
      >
        <div className="flex items-center justify-between border-b border-white/10 p-4">
          <h3
            className="text-[11px] font-semibold uppercase tracking-[0.08em]"
            style={{ fontFamily: 'JetBrains Mono, monospace', color: '#71717A' }}
          >
            Sources
          </h3>
          <button onClick={() => setSourcesOpen(false)} className="text-[#71717A] hover:text-white" aria-label="Collapse sources">
            <CollapseIcon />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {currentSources.length === 0 ? (
            <p className="text-[13px]" style={{ color: '#71717A' }}>
              {lastAssistantMessage
                ? 'No sources were cited for this answer.'
                : 'Sources used to answer your question will appear here.'}
            </p>
          ) : (
            <div className="space-y-2">
              {currentSources.map((src, idx) => (
                <div key={idx} className="rounded-[11px] border border-white/10 bg-white/4 p-3">
                  <p className="truncate text-[13px] font-medium text-white">
                    {src.filename || src.doc_id || src.source || `Source ${idx + 1}`}
                  </p>
                  {(src.page !== undefined || src.score !== undefined || src.relevance_score !== undefined) && (
                    <p className="mt-0.5 text-[12px]" style={{ color: '#71717A' }}>
                      {src.page !== undefined ? `Page ${src.page} • ` : ''}
                      {(src.score ?? src.relevance_score) !== undefined
                        ? `${Math.round((src.score ?? src.relevance_score) * 100)}% match`
                        : ''}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </aside>

      {!sourcesOpen && (
        <button
          onClick={() => setSourcesOpen(true)}
          className="hidden w-8 shrink-0 items-center justify-center border-l border-white/10 bg-[#18181B]/20 text-[#71717A] backdrop-blur-xl hover:text-white xl:flex"
          aria-label="Expand sources"
        >
          <ExpandIcon />
        </button>
      )}
    </div>
  );
}

/* --- icons --- */
function SourcesIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M14 2v6h6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
function CollapseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9 18l6-6-6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
function ExpandIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M15 18l-6-6 6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}