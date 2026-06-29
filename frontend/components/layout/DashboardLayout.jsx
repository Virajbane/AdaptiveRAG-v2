'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAuth } from '@/app/context/AuthContext';
import { SettingsPanel } from '@/components/settings/SettingsPanel';

export function DashboardLayout({ children }) {
  const pathname = usePathname();
  const { user, logout } = useAuth?.() ?? {};
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [avatarMenuOpen, setAvatarMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const avatarMenuRef = useRef(null);

  const navItems = [
  { href: '/dashboard/chat',      label: 'Chat',      icon: ChatIcon },
  { href: '/dashboard/search',    label: 'Search',    icon: SearchIcon },  // ← add
  { href: '/dashboard/documents', label: 'Documents', icon: DocIcon },
  { href: '/dashboard/memory',    label: 'Memory',    icon: MemoryIcon }
];

  useEffect(() => {
    const onClick = (e) => {
      if (avatarMenuRef.current && !avatarMenuRef.current.contains(e.target)) {
        setAvatarMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, []);

  const displayName = user?.name || user?.email || 'Account';
  const initial = (user?.name || user?.email || 'U').charAt(0).toUpperCase();

  return (
    <div className="min-h-screen text-[#F4F4F5]">
      {/* Topbar */}
      <header className="sticky top-0 z-20 border-b border-white/10 bg-[#18181B]/40 backdrop-blur-xl">
        <div className="flex items-center justify-between px-4 py-3 lg:px-6">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setMobileNavOpen((v) => !v)}
              className="rounded-[11px] border border-white/10 p-2 text-[#A1A1AA] transition-colors hover:text-white lg:hidden"
              aria-label="Toggle navigation"
            >
              <MenuIcon />
            </button>
            <span
              className="text-[12px] font-semibold tracking-widest"
              style={{ fontFamily: 'JetBrains Mono, monospace', color: '#34D399' }}
            >
              RAG 2.0
            </span>
          </div>

          {/* Avatar + dropdown */}
          <div className="relative" ref={avatarMenuRef}>
            <button
              type="button"
              onClick={() => setAvatarMenuOpen((v) => !v)}
              className="flex items-center gap-2 rounded-[11px] px-2 py-1.5 transition-colors hover:bg-white/5"
            >
              <span className="hidden text-[13px] sm:inline" style={{ color: '#A1A1AA' }}>
                {displayName}
              </span>
              <div
                className="flex h-8 w-8 items-center justify-center rounded-full text-[12px] font-semibold"
                style={{ background: 'rgba(52,211,153,0.15)', color: '#34D399' }}
              >
                {initial}
              </div>
            </button>

            {avatarMenuOpen && (
              <div className="absolute right-0 top-full mt-2 w-48 rounded-[11px] border border-white/10 bg-[#18181B]/95 p-1.5 shadow-xl backdrop-blur-xl">
                <div className="px-3 py-2 text-[13px]" style={{ color: '#A1A1AA' }}>
                  Signed in as
                  <div className="truncate text-white">{displayName}</div>
                </div>
                <div className="my-1 border-t border-white/10" />
                <button
                  onClick={() => { setAvatarMenuOpen(false); setSettingsOpen(true); }}
                  className="flex w-full items-center gap-2 rounded-[8px] px-3 py-2 text-left text-[14px] text-[#F4F4F5] transition-colors hover:bg-white/5"
                >
                  <SettingsIcon active={false} />
                  Settings
                </button>
                <button
                  onClick={() => { setAvatarMenuOpen(false); logout?.(); }}
                  className="flex w-full items-center gap-2 rounded-[8px] px-3 py-2 text-left text-[14px] transition-colors hover:bg-red-500/10"
                  style={{ color: '#FCA5A5' }}
                >
                  <SignOutIcon />
                  Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar (desktop) */}
        <aside className="sticky top-14.25 hidden h-[calc(100vh-57px)] w-56 shrink-0 border-r border-white/10 bg-[#18181B]/30 backdrop-blur-xl lg:flex lg:flex-col lg:p-4">
          <nav className="flex flex-col gap-1">
            {navItems.map((item) => {
              const active = pathname?.startsWith(item.href);
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-3 rounded-[11px] px-3 py-2.5 text-[14px] font-medium transition-colors ${
                    active
                      ? 'bg-[#34D399]/10 text-[#34D399]'
                      : 'text-[#A1A1AA] hover:bg-white/5 hover:text-white'
                  }`}
                >
                  <Icon active={active} />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </aside>

        {/* Sidebar (mobile drawer) */}
        {mobileNavOpen && (
          <div className="fixed inset-0 z-30 lg:hidden">
            <div className="absolute inset-0 bg-black/60" onClick={() => setMobileNavOpen(false)} />
            <aside className="absolute left-0 top-0 h-full w-64 border-r border-white/10 bg-[#18181B]/95 p-4 backdrop-blur-xl">
              <nav className="mt-14 flex flex-col gap-1">
                {navItems.map((item) => {
                  const active = pathname?.startsWith(item.href);
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={() => setMobileNavOpen(false)}
                      className={`flex items-center gap-3 rounded-[11px] px-3 py-2.5 text-[14px] font-medium transition-colors ${
                        active
                          ? 'bg-[#34D399]/10 text-[#34D399]'
                          : 'text-[#A1A1AA] hover:bg-white/5 hover:text-white'
                      }`}
                    >
                      <Icon active={active} />
                      {item.label}
                    </Link>
                  );
                })}
              </nav>
            </aside>
          </div>
        )}

        {/* Page content */}
        <main className="min-h-[calc(100vh-57px)] flex-1">{children}</main>
      </div>

      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}

/* --- Icons --- */
function MenuIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 6h18M3 12h18M3 18h18" strokeLinecap="round" />
    </svg>
  );
}

function iconColor(active) {
  return active ? '#34D399' : '#71717A';
}

function ChatIcon({ active }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={iconColor(active)} strokeWidth="2">
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function DocIcon({ active }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={iconColor(active)} strokeWidth="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M14 2v6h6M9 13h6M9 17h6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function MemoryIcon({ active }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={iconColor(active)} strokeWidth="2">
      <ellipse cx="12" cy="5" rx="9" ry="3" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SettingsIcon({ active }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={iconColor(active)} strokeWidth="2">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SearchIcon({ active }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={iconColor(active)} strokeWidth="2">
      <circle cx="11" cy="11" r="8" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M21 21l-4.35-4.35" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SignOutIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FCA5A5" strokeWidth="2">
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M16 17l5-5-5-5M21 12H9" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}