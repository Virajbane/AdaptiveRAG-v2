'use client';

// Root Dashboard — wires the sidebar, topbar, and the 4 views together.
// Chat state now lives in ChatContext (see app/context/ChatContext.jsx),
// NOT inside ChatView — this is what makes chat survive tab switches
// and page refreshes. Only the "NEW QUERY" button resets it.

import { useState } from 'react';
import { useAuth } from '@/app/context/AuthContext';
import { ChatProvider, useChat } from '@/app/context/ChatContext';
import { SettingsPanel } from '@/components/settings/SettingsPanel';

import Sidebar from './Sidebar';
import Topbar from './Topbar';
import ChatView from './views/ChatView';
import LibraryView from './views/LibraryView';
import SearchView from './views/SearchView';
import HistoryView from './views/HistoryView';

export default function Dashboard() {
  const { user, logout, token } = useAuth?.() ?? {};

  return (
    <ChatProvider token={token}>
      <DashboardInner user={user} logout={logout} token={token} />
    </ChatProvider>
  );
}

function DashboardInner({ user, logout, token }) {
  const [view, setView] = useState('chat'); // 'chat' | 'library' | 'search' | 'history'
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [documents, setDocuments] = useState([]); // shared with Sidebar + ChatView's file count

  const { newChat } = useChat();

  const displayName = user?.name || user?.email || 'Account';
  const initial = (user?.name || user?.email || 'U').charAt(0).toUpperCase();

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', background: '#121212', color: '#e5e2e1', fontFamily: "'Hanken Grotesk', sans-serif" }}>
      <Sidebar
        view={view}
        setView={setView}
        documents={documents}
        onNewQuery={() => {
          newChat();       // clear chat state + localStorage
          setView('chat');  // make sure we land on the chat tab
        }}
      />

      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#121212' }}>
        <Topbar
          view={view}
          setView={setView}
          displayName={displayName}
          initial={initial}
          onSettings={() => setSettingsOpen(true)}
          onLogout={logout}
        />

        {/*
          IMPORTANT: views are hidden with CSS `display`, not conditionally
          rendered with `&&`. Conditional rendering unmounts the component,
          which used to wipe chat state and drop in-flight responses.
          Keeping all views mounted (just hidden) fixes that — and now that
          chat state lives in ChatContext anyway, this is mostly a safety net
          for any local UI state (scroll position, input draft, etc).
        */}
        <div style={{ display: view === 'chat' ? 'flex' : 'none', flex: 1, minHeight: 0 }}>
          <ChatView documentCount={documents.length} />
        </div>
        <div style={{ display: view === 'library' ? 'flex' : 'none', flex: 1, minHeight: 0, flexDirection: 'column' }}>
          <LibraryView token={token} onDocumentsChange={setDocuments} />
        </div>
        <div style={{ display: view === 'search' ? 'flex' : 'none', flex: 1, minHeight: 0, flexDirection: 'column' }}>
          <SearchView token={token} />
        </div>
        <div style={{ display: view === 'history' ? 'flex' : 'none', flex: 1, minHeight: 0, flexDirection: 'column' }}>
          <HistoryView token={token} />
        </div>
      </main>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;600;700&display=swap');
        * { box-sizing: border-box; }
        body { overflow: hidden; }
        ::-webkit-scrollbar { width: 12px; }
        ::-webkit-scrollbar-track { background: #121212; }
        ::-webkit-scrollbar-thumb { background: #4d4d4d; border: 3px solid #121212; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #7c7c7c; }
      `}</style>

      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}