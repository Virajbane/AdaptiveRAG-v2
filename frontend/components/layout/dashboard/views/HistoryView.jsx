'use client';

// History view: conversation memory + session summaries, with a
// session-id switcher and clear-session action.
// Self-contained — if memory/history breaks, only this file changes.

import { useState, useEffect } from 'react';
import { S } from '../styles';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function HistoryView({ token }) {
  const [sessionId, setSessionId] = useState('default_session');
  const [messages, setMessages] = useState([]);
  const [summaries, setSummaries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadMemory = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_URL}/api/v1/memory/load`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (!res.ok) throw new Error('Failed to load memory');
      const data = await res.json();
      setMessages(data.history || []);
      setSummaries(data.summaries || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (token) loadMemory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const clearSession = async () => {
    if (!window.confirm('Clear this conversation?')) return;
    try {
      await fetch(`${API_URL}/api/v1/memory/session/${sessionId}`, { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } });
      setMessages([]);
      setSummaries([]);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ padding: '32px 32px', overflowY: 'auto', height: 'calc(100vh - 64px)' }}>
      <div style={{ maxWidth: 860, margin: '0 auto' }}>
        <h1 style={S.pageTitle}>Conversation Memory</h1>
        <p style={S.pageSubtitle}>View your conversation history and session summaries.</p>

        {/* Session controls */}
        <div style={{ ...S.card, padding: 20, marginBottom: 24 }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <label style={{ fontSize: 13, color: '#b3b3b3', flexShrink: 0 }}>Session ID</label>
            <input
              value={sessionId}
              onChange={e => setSessionId(e.target.value)}
              style={{ flex: 1, background: '#121212', border: 'none', borderRadius: 9999, padding: '8px 14px', fontSize: 13, color: '#fff', outline: 'none', boxShadow: 'rgb(18,18,18) 0px 1px 0px, rgb(77,77,77) 0px 0px 0px 1px inset' }}
            />
            <button onClick={loadMemory} style={{ padding: '8px 18px', borderRadius: 9999, background: '#252525', border: '1px solid rgba(255,255,255,0.1)', color: '#e5e2e1', fontSize: 13, cursor: 'pointer', fontWeight: 600 }}>Load</button>
            <button onClick={clearSession} style={{ padding: '8px 18px', borderRadius: 9999, background: 'transparent', border: '1px solid rgba(243,114,127,0.35)', color: '#f3727f', fontSize: 13, cursor: 'pointer', fontWeight: 600 }}>Clear</button>
          </div>
        </div>

        {error && <p style={{ color: '#f3727f', fontSize: 14, marginBottom: 16 }}>{error}</p>}
        {loading && <p style={{ color: '#b3b3b3', fontSize: 14 }}>Loading memory…</p>}

        {!loading && messages.length === 0 && (
          <div style={{ ...S.card, padding: 48, textAlign: 'center', marginBottom: 20 }}>
            <p style={{ color: '#b3b3b3', fontSize: 14 }}>No conversation history yet</p>
          </div>
        )}

        {messages.length > 0 && (
          <>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#b3b3b3', marginBottom: 10 }}>
              Conversation History ({messages.length} messages)
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 28 }}>
              {messages.map((msg, idx) => (
                <div key={idx} style={{ ...S.card, padding: '14px 18px', borderLeft: `3px solid ${msg.role === 'user' ? '#539df5' : '#1ed760'}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <span style={{ fontSize: 12, fontWeight: 700, color: msg.role === 'user' ? '#539df5' : '#1ed760', textTransform: 'capitalize' }}>{msg.role}</span>
                    {msg.timestamp && <span style={{ fontSize: 11, color: '#4d4d4d' }}>{new Date(msg.timestamp).toLocaleString()}</span>}
                  </div>
                  <p style={{ fontSize: 14, color: '#e5e2e1', lineHeight: 1.6, margin: 0 }}>{msg.content}</p>
                </div>
              ))}
            </div>
          </>
        )}

        {summaries.length > 0 && (
          <>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#b3b3b3', marginBottom: 10 }}>Session Summaries</div>
            {summaries.map((s, i) => (
              <div key={i} style={{ ...S.card, padding: '18px 20px', marginBottom: 10 }}>
                <p style={{ fontSize: 14, fontWeight: 600, color: '#e5e2e1', marginBottom: 10, marginTop: 0 }}>{s.summary}</p>
                {s.topics && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {s.topics.map((t, ti) => (
                      <span key={ti} style={{ padding: '3px 10px', borderRadius: 9999, background: 'rgba(30,215,96,0.12)', color: '#1ed760', fontSize: 12, fontWeight: 600 }}>{t}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}