'use client';

// Chat view: message thread, sources-cited cards, typing indicator, composer.
// Chat state and API logic now live in ChatContext — this file is a thin
// consumer. It only owns UI-local state: the composer's textarea value
// and its auto-resize behavior.

import { useState, useRef, useEffect, useCallback } from 'react';
import { useChat } from '@/app/context/ChatContext';

const INPUT_MIN_HEIGHT = 44;
const INPUT_MAX_HEIGHT = 160;

function useAutoResizeTextarea() {
  const ref = useRef(null);
  const adjust = useCallback((reset) => {
    const el = ref.current;
    if (!el) return;
    el.style.height = `${INPUT_MIN_HEIGHT}px`;
    if (!reset) {
      el.style.height = `${Math.min(el.scrollHeight, INPUT_MAX_HEIGHT)}px`;
    }
  }, []);
  useEffect(() => {
    if (ref.current) ref.current.style.height = `${INPUT_MIN_HEIGHT}px`;
  }, []);
  return { textareaRef: ref, adjustHeight: adjust };
}

export default function ChatView({ documentCount = 0 }) {
  const { messages, loading, elapsed, sendMessage, cancelMessage } = useChat();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const { textareaRef, adjustHeight } = useAutoResizeTextarea();

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const handleSend = () => {
    if (!input.trim() || loading) return;
    sendMessage(input);
    setInput('');
    adjustHeight(true);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%', position: 'relative' }}>
      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '32px 24px 160px', maxWidth: 860, margin: '0 auto', width: '100%' }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', marginTop: 80, color: '#b3b3b3' }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>✦</div>
            <p style={{ fontSize: 16, color: '#b3b3b3' }}>Ask me anything about your documents.</p>
            <p style={{ fontSize: 13, color: '#4d4d4d', marginTop: 6 }}>
              Searching {documentCount} file{documentCount !== 1 ? 's' : ''} in Project Alpha
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <MessageBubble key={idx} msg={msg} />
        ))}

        {loading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 24, opacity: 0.7 }}>
            <div style={{ width: 24, height: 24, borderRadius: '50%', background: '#252525', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#b3b3b3" strokeWidth="2"><path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" strokeLinecap="round"/></svg>
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              {[0, 0.2, 0.4].map((delay, i) => (
                <div key={i} style={{ width: 6, height: 6, borderRadius: '50%', background: '#b3b3b3', animation: `blink 1.4s ${delay}s infinite` }} />
              ))}
            </div>
            <button onClick={cancelMessage} style={{ fontSize: 12, color: '#f3727f', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}>
              Cancel ({elapsed}s)
            </button>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Composer */}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, background: 'linear-gradient(to top, #121212 60%, transparent)', padding: '20px 24px 24px' }}>
        <div style={{ maxWidth: 860, margin: '0 auto' }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, background: '#1f1f1f', borderRadius: 9999, padding: '8px 8px 8px 16px', boxShadow: 'rgb(18,18,18) 0px 1px 0px, rgb(77,77,77) 0px 0px 0px 1px inset' }}>
            <button style={{ width: 38, height: 38, borderRadius: '50%', background: 'transparent', border: 'none', color: '#b3b3b3', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4m0 4h.01" strokeLinecap="round"/></svg>
            </button>
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={e => { setInput(e.target.value); adjustHeight(); }}
              onKeyDown={handleKeyDown}
              placeholder="Ask about your knowledge base…"
              disabled={loading}
              style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', resize: 'none', fontSize: 15, color: '#fff', lineHeight: 1.5, height: INPUT_MIN_HEIGHT, overflow: 'hidden', paddingTop: 8 }}
            />
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
              <button style={{ width: 34, height: 34, borderRadius: '50%', background: 'transparent', border: 'none', color: '#b3b3b3', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8"/></svg>
              </button>
              <button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                style={{ width: 40, height: 40, borderRadius: '50%', background: '#1ed760', border: 'none', cursor: loading || !input.trim() ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: loading || !input.trim() ? 0.5 : 1, transition: 'transform 0.1s', flexShrink: 0 }}
                onMouseEnter={e => { if (!loading && input.trim()) e.currentTarget.style.transform = 'scale(1.08)'; }}
                onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="#003913"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
              </button>
            </div>
          </div>
          <p style={{ textAlign: 'center', fontSize: 11, color: '#4d4d4d', marginTop: 8 }}>
            Shift + Enter for new line • Searching {documentCount} files in <span style={{ color: '#1ed760', cursor: 'pointer' }}>Project Alpha</span>
          </p>
        </div>
      </div>

      <style>{`@keyframes blink { 0%,100% { opacity: 0.2; } 50% { opacity: 1; } }`}</style>
    </div>
  );
}

function MessageBubble({ msg }) {
  return (
    <div style={{ marginBottom: 28, display: 'flex', flexDirection: 'column', alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
      {msg.role === 'assistant' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, paddingLeft: 4 }}>
          <div style={{ width: 24, height: 24, borderRadius: '50%', background: '#1ed760', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="#003913"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
          </div>
          <span style={{ fontSize: 14, fontWeight: 700, color: '#fff' }}>AI Assistant</span>
        </div>
      )}
      <div style={{
        maxWidth: '82%',
        background: msg.role === 'user' ? '#1f1f1f' : '#181818',
        border: msg.role === 'assistant' ? '1px solid rgba(255,255,255,0.06)' : 'none',
        borderRadius: msg.role === 'user' ? '18px 18px 4px 18px' : '4px 18px 18px 18px',
        padding: '14px 18px',
        fontSize: 15, lineHeight: 1.6, color: '#e5e2e1',
        boxShadow: msg.role === 'assistant' ? 'rgba(0,0,0,0.3) 0px 8px 24px' : 'none',
      }}>
        <p style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{msg.content}</p>

        {msg.sources?.length > 0 && (
          <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            <p style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#b3b3b3', marginBottom: 8 }}>Sources Cited</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px,1fr))', gap: 8 }}>
              {msg.sources.map((src, si) => (
                <div key={si} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', background: '#121212', borderRadius: 8, border: '1px solid rgba(255,255,255,0.06)', cursor: 'pointer' }}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1ed760" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>
                  <div style={{ flex: 1, overflow: 'hidden' }}>
                    <p style={{ fontSize: 12, color: '#e5e2e1', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', margin: 0 }}>{src.filename || `Source ${si + 1}`}</p>
                    {src.page && <p style={{ fontSize: 10, color: '#b3b3b3', margin: 0 }}>Page {src.page}{src.score !== undefined ? ` • ${Math.round(src.score * 100)}% match` : ''}</p>}
                  </div>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#b3b3b3" strokeWidth="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15,3 21,3 21,9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                </div>
              ))}
            </div>
          </div>
        )}

        {msg.confidence !== undefined && (
          <p style={{ fontSize: 11, color: '#4d4d4d', marginTop: 8, marginBottom: 0, fontFamily: 'monospace' }}>
            Confidence: {(msg.confidence * 100).toFixed(0)}%{msg.searchTimeMs !== undefined ? ` • ${msg.searchTimeMs.toFixed(0)}ms` : ''}
          </p>
        )}
      </div>

      {msg.role === 'assistant' && (
        <div style={{ display: 'flex', gap: 4, marginTop: 6, paddingLeft: 4 }}>
          {['thumb_up', 'thumb_down', 'copy'].map(label => (
            <button key={label} style={{ padding: '3px 6px', background: 'transparent', border: 'none', cursor: 'pointer', fontSize: 14, color: '#b3b3b3', borderRadius: 6 }}
              onMouseEnter={e => e.currentTarget.style.color = '#1ed760'}
              onMouseLeave={e => e.currentTarget.style.color = '#b3b3b3'}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                {label === 'thumb_up' && <><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></>}
                {label === 'thumb_down' && <><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zM17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></>}
                {label === 'copy' && <><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></>}
              </svg>
            </button>
          ))}
        </div>
      )}

      {msg.role === 'user' && (
        <span style={{ fontSize: 11, color: '#4d4d4d', marginTop: 4, paddingRight: 4 }}>
          {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      )}
    </div>
  );
}