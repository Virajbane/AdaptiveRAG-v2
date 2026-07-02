'use client';

// Left sidebar: brand, "New Query" button, main nav (Home/Library/History),
// workspace docs shortcuts, and footer (Upgrade/Account).
// Props: view (current view key), setView, documents (for the docs list).

import { S } from './styles';
import { HomeIcon, LibraryIcon, HistoryIcon, DocFileIcon, UpgradeIcon, AccountIcon } from './icons';

const NAV_MAIN = [
  { key: 'chat',    label: 'Home',    icon: HomeIcon },
  { key: 'library', label: 'Library', icon: LibraryIcon },
  { key: 'history', label: 'History', icon: HistoryIcon },
];

export default function Sidebar({ view, setView, documents = [], onNewQuery }) {
  return (
    <aside style={S.sidebar}>
      {/* Brand */}
      <div style={S.brand}>
        <div style={S.brandIcon}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="#003913" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <div>
          <div style={S.brandName}>Adaptive-RAG</div>
          <div style={S.brandSub}>Multi Agent AI</div>
        </div>
      </div>

      {/* New Query */}
      <button
        style={S.newQueryBtn}
        onClick={onNewQuery}
        onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.02)'}
        onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
      >
        New Query
      </button>

      {/* Main nav */}
      <nav style={{ flex: 1, overflowY: 'auto' }}>
        <div style={S.navLabel}>Main</div>
        {NAV_MAIN.map(({ key, label, icon: Icon }) => {
          const active = view === key;
          return (
            <button
              key={key}
              onClick={() => setView(key)}
              style={{ ...S.navItem, ...(active ? S.navItemActive : {}) }}
              onMouseEnter={e => { if (!active) e.currentTarget.style.background = '#252525'; }}
              onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent'; }}
            >
              <Icon active={active} />
              <span style={{ fontWeight: active ? 700 : 400 }}>{label}</span>
            </button>
          );
        })}

        {/* Workspace docs list */}
        {documents.length > 0 && (
          <>
            <div style={{ ...S.navLabel, marginTop: 20 }}>Workspace Docs</div>
            {documents.slice(0, 6).map(doc => (
              <button
                key={doc._id}
                onClick={() => setView('library')}
                style={{ ...S.navItem, gap: 10 }}
                onMouseEnter={e => e.currentTarget.style.background = '#252525'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                <DocFileIcon />
                <span style={{ fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#b3b3b3' }}>
                  {doc.filename}
                </span>
              </button>
            ))}
          </>
        )}
      </nav>

      {/* Footer */}
      <div style={S.sidebarFooter}>
        {[
          { label: 'Upgrade', icon: <UpgradeIcon /> },
          { label: 'Account', icon: <AccountIcon /> },
        ].map(({ label, icon }) => (
          <button
            key={label}
            style={S.navItem}
            onMouseEnter={e => e.currentTarget.style.background = '#252525'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            {icon}
            <span style={{ fontWeight: 400 }}>{label}</span>
          </button>
        ))}
      </div>
    </aside>
  );
}