'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/app/context/AuthContext';

/**
 * Settings as a slide-over, opened from the avatar dropdown in DashboardLayout —
 * not a sidebar nav item. Matches the convention that account settings are
 * low-traffic and shouldn't compete for space with Chat / Knowledge Base.
 */
export function SettingsPanel({ open, onClose }) {
  const { token, logout } = useAuth();
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [settings, setSettings] = useState({
    theme: 'dark',
    notifications: true,
    maxResults: 5,
  });

  useEffect(() => {
    if (!open || !token) return;

    const loadUserProfile = async () => {
      setLoading(true);
      try {
        const res = await fetch('http://localhost:8000/api/v1/auth/me', {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error('Failed to load profile');
        const data = await res.json();
        setUser(data);

        const savedSettings = localStorage.getItem('app_settings');
        if (savedSettings) setSettings(JSON.parse(savedSettings));
        setError('');
      } catch (err) {
        console.error('Error loading profile:', err);
        setError('Failed to load profile');
      } finally {
        setLoading(false);
      }
    };

    loadUserProfile();
  }, [open, token]);

  const saveSettings = () => {
    try {
      localStorage.setItem('app_settings', JSON.stringify(settings));
      setSuccess('Settings saved');
      setTimeout(() => setSuccess(''), 2500);
    } catch (err) {
      setError('Failed to save settings');
    }
  };

  const labelClass = 'mb-2 block text-[11px] font-semibold uppercase tracking-[0.08em]';
  const labelStyle = { fontFamily: 'JetBrains Mono, monospace', color: '#71717A' };
  const fieldClass =
    'w-full rounded-[11px] border border-white/10 bg-[#0F0F11]/60 px-4 py-2.5 text-[14px] text-white outline-none transition-all duration-200 focus:border-[#34D399] focus:shadow-[0_0_0_3px_rgba(52,211,153,0.15)]';
  const disabledFieldClass =
    'w-full cursor-not-allowed rounded-[11px] border border-white/10 bg-white/[0.03] px-4 py-2.5 text-[14px] text-[#A1A1AA]';

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-40">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />

      {/* Panel */}
      <div className="absolute right-0 top-0 h-full w-full max-w-md overflow-y-auto border-l border-white/10 bg-[#18181B]/95 p-6 backdrop-blur-xl">
        <div className="mb-6 flex items-center justify-between">
          <h2
            className="text-xl font-medium tracking-tight text-white"
            style={{ fontFamily: 'Inter, sans-serif' }}
          >
            Settings
          </h2>
          <button
            onClick={onClose}
            className="rounded-[8px] p-1.5 text-[#A1A1AA] transition-colors hover:bg-white/5 hover:text-white"
            aria-label="Close settings"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {error && (
          <div
            className="mb-4 rounded-[11px] border px-4 py-3 text-[13px]"
            style={{ borderColor: 'rgba(248,113,113,0.3)', background: 'rgba(248,113,113,0.08)', color: '#FCA5A5' }}
          >
            {error}
          </div>
        )}
        {success && (
          <div
            className="mb-4 rounded-[11px] border px-4 py-3 text-[13px]"
            style={{ borderColor: 'rgba(52,211,153,0.3)', background: 'rgba(52,211,153,0.08)', color: '#34D399' }}
          >
            {success}
          </div>
        )}

        {loading && (
          <div className="py-12 text-center">
            <div className="inline-block h-7 w-7 animate-spin rounded-full border-b-2" style={{ borderColor: '#34D399' }} />
          </div>
        )}

        {!loading && user && (
          <div className="space-y-6">
            <div>
              <h3
                className="mb-3 text-[11px] font-semibold uppercase tracking-[0.08em]"
                style={labelStyle}
              >
                Profile
              </h3>
              <div className="space-y-3 rounded-[11px] border border-white/10 bg-white/4 p-4">
                <div>
                  <label className={labelClass} style={labelStyle}>Email</label>
                  <input type="email" value={user.email || ''} disabled className={disabledFieldClass} />
                </div>
                <div>
                  <label className={labelClass} style={labelStyle}>Account Created</label>
                  <input
                    type="text"
                    value={user.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'}
                    disabled
                    className={disabledFieldClass}
                  />
                </div>
              </div>
            </div>

            <div>
              <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.08em]" style={labelStyle}>
                Preferences
              </h3>
              <div className="space-y-4 rounded-[11px] border border-white/10 bg-white/4 p-4">
                <div>
                  <label className={labelClass} style={labelStyle}>Theme</label>
                  <select
                    value={settings.theme}
                    onChange={(e) => setSettings({ ...settings, theme: e.target.value })}
                    className={fieldClass}
                  >
                    <option value="light" className="bg-[#18181B]">Light</option>
                    <option value="dark" className="bg-[#18181B]">Dark</option>
                    <option value="auto" className="bg-[#18181B]">Auto</option>
                  </select>
                </div>

                <div>
                  <label className={labelClass} style={labelStyle}>Max Search Results</label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={settings.maxResults}
                    onChange={(e) => setSettings({ ...settings, maxResults: parseInt(e.target.value, 10) })}
                    className={fieldClass}
                  />
                </div>

                <label className="flex items-center gap-3 text-[14px]" style={{ color: '#A1A1AA' }}>
                  <input
                    type="checkbox"
                    checked={settings.notifications}
                    onChange={(e) => setSettings({ ...settings, notifications: e.target.checked })}
                    className="h-4 w-4 rounded"
                    style={{ accentColor: '#34D399' }}
                  />
                  Enable notifications
                </label>

                <button
                  onClick={saveSettings}
                  className="w-full rounded-[11px] py-2.5 text-[14px] font-medium text-[#022C22] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_8px_24px_rgba(52,211,153,0.35)]"
                  style={{ background: '#34D399' }}
                >
                  Save Preferences
                </button>
              </div>
            </div>

            <div
              className="rounded-[11px] border p-4"
              style={{ borderColor: 'rgba(248,113,113,0.25)', background: 'rgba(248,113,113,0.05)' }}
            >
              <p className="mb-3 text-[13px]" style={{ color: '#FCA5A5' }}>
                Sign out of your account on this device.
              </p>
              <button
                onClick={() => {
                  onClose();
                  logout();
                }}
                className="w-full rounded-[11px] border py-2 text-[14px] font-medium transition-colors hover:bg-red-500/10"
                style={{ borderColor: 'rgba(248,113,113,0.4)', color: '#FCA5A5' }}
              >
                Sign Out
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}