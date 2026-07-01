'use client';

// Top header bar: workspace title, view tabs (Chat / Knowledge Base / Search),
// workspace search pill, icon buttons, and the avatar dropdown menu.

import { useState, useRef, useEffect } from 'react';
import { S } from './styles';
import { FilterIcon, HelpIcon, SettingsIcon } from './icons';

const TABS = [
  { key: 'chat',    label: 'Chat' },
  { key: 'library', label: 'Knowledge Base' },
  { key: 'search',  label: 'Search' },
];

export default function Topbar({ view, setView, displayName, initial, onSettings, onLogout }) {
  const [avatarOpen, setAvatarOpen] = useState(false);
  const avatarRef = useRef(null);

  useEffect(() => {
    const h = (e) => { if (avatarRef.current && !avatarRef.current.contains(e.target)) setAvatarOpen(false); };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, []);

  return (
    <header style={S.topbar}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 32 }}>
        <span style={S.topbarTitle}>RAG Workspace</span>
        <nav style={{ display: 'flex', gap: 4 }}>
          {TABS.map(({ key, label }) => {
            const active = view === key;
            return (
              <button
                key={key}
                onClick={() => setView(key)}
                style={{ ...S.topTab, ...(active ? S.topTabActive : {}) }}
                onMouseEnter={e => { if (!active) e.currentTarget.style.color = '#fff'; }}
                onMouseLeave={e => { if (!active) e.currentTarget.style.color = '#b3b3b3'; }}
              >
                {label}
              </button>
            );
          })}
        </nav>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {/* Search pill */}
        <div style={S.searchPill}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#b3b3b3" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35" strokeLinecap="round"/></svg>
          <input placeholder="Search workspace…" style={S.searchPillInput} />
        </div>

        {/* Icon buttons */}
        {[FilterIcon, HelpIcon].map((Icon, i) => (
          <button
            key={i}
            style={S.iconBtn}
            onMouseEnter={e => { e.currentTarget.style.background = '#252525'; e.currentTarget.style.color = '#fff'; }}
            onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#b3b3b3'; }}
          >
            <Icon />
          </button>
        ))}
        <button
          style={S.iconBtn}
          onClick={onSettings}
          onMouseEnter={e => { e.currentTarget.style.background = '#252525'; e.currentTarget.style.color = '#fff'; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#b3b3b3'; }}
        >
          <SettingsIcon />
        </button>

        {/* Avatar */}
        <div style={{ position: 'relative' }} ref={avatarRef}>
          <button onClick={() => setAvatarOpen(v => !v)} style={S.avatarBtn}>{initial}</button>
          {avatarOpen && (
            <div style={S.avatarMenu}>
              <div style={S.avatarMenuHeader}>
                <div style={{ fontSize: 11, color: '#b3b3b3' }}>Signed in as</div>
                <div style={{ fontSize: 13, color: '#fff', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{displayName}</div>
              </div>
              <div style={{ borderTop: '1px solid rgba(255,255,255,0.07)', margin: '4px 0' }} />
              <button
                style={S.avatarMenuItem}
                onClick={() => { setAvatarOpen(false); onSettings?.(); }}
                onMouseEnter={e => e.currentTarget.style.background = '#252525'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                Settings
              </button>
              <button
                style={{ ...S.avatarMenuItem, color: '#f3727f' }}
                onClick={() => { setAvatarOpen(false); onLogout?.(); }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(243,114,127,0.1)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                Sign out
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}