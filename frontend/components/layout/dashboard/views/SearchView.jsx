'use client';

// Search view: query bar + result cards.
// Update the fetch URL below if your backend's search route differs
// from /api/v1/search — this file is self-contained, no other file
// needs to change.

import { useState } from 'react';
import { S } from '../styles';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function SearchView({ token }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    setError('');
    try {
      const res = await fetch(`${API_URL}/api/v1/search?q=${encodeURIComponent(query)}&top_k=10`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error('Search failed');
      const data = await res.json();
      setResults(data.results || []);
    } catch (err) {
      setError(err.message);
      setResults([]);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div style={{ padding: '32px 32px', overflowY: 'auto', height: 'calc(100vh - 64px)' }}>
      <div style={{ maxWidth: 860, margin: '0 auto' }}>
        <h1 style={S.pageTitle}>Search Documents</h1>
        <p style={S.pageSubtitle}>Find specific content across your knowledge base.</p>

        {/* Search bar */}
        <div style={{ ...S.card, padding: 20, marginBottom: 24, display: 'flex', gap: 10, alignItems: 'center' }}>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 10, background: '#121212', borderRadius: 9999, padding: '10px 16px', boxShadow: 'rgb(18,18,18) 0px 1px 0px, rgb(77,77,77) 0px 0px 0px 1px inset' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#b3b3b3" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35" strokeLinecap="round"/></svg>
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
              placeholder="Search your documents…"
              style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', fontSize: 15, color: '#fff' }}
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={searching || !query.trim()}
            style={{ padding: '10px 24px', borderRadius: 9999, background: '#1ed760', border: 'none', color: '#003913', fontSize: 14, fontWeight: 700, letterSpacing: '0.05em', cursor: 'pointer', opacity: searching || !query.trim() ? 0.5 : 1 }}
          >
            {searching ? 'Searching…' : 'Search'}
          </button>
        </div>

        {error && <p style={{ color: '#f3727f', fontSize: 14, marginBottom: 16 }}>{error}</p>}

        {/* Results */}
        {results.length === 0 && !searching && (
          <div style={{ ...S.card, padding: 48, textAlign: 'center' }}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#4d4d4d" strokeWidth="1.5" style={{ margin: '0 auto 12px', display: 'block' }}><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35" strokeLinecap="round"/></svg>
            <p style={{ color: '#b3b3b3', fontSize: 14 }}>Enter a query to search your documents</p>
          </div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {results.map((r, i) => (
            <div key={i} style={{ ...S.card, padding: '16px 20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: '#1ed760' }}>{r.filename || r.source || `Result ${i + 1}`}</span>
                {r.score !== undefined && <span style={{ fontSize: 11, color: '#b3b3b3' }}>{Math.round(r.score * 100)}% match</span>}
              </div>
              <p style={{ fontSize: 14, color: '#cbcbcb', lineHeight: 1.6, margin: 0 }}>{r.content || r.text}</p>
              {r.page && <p style={{ fontSize: 11, color: '#4d4d4d', marginTop: 6, marginBottom: 0 }}>Page {r.page}</p>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}