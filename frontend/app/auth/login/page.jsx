// frontend/app/auth/login/page.jsx
'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { validateEmail } from '@/config/security';
import KineticNetwork from '@/components/KineticNetwork';

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [focusedField, setFocusedField] = useState(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (searchParams.get('session_expired') === 'true') {
      setError('Your session expired. Please sign in again.');
    }
  }, [searchParams]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');

    const emailError = validateEmail(email);
    if (emailError) {
      setError(emailError);
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/v1/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error('Too many login attempts. Please try again later.');
        }
        const data = await response.json();
        throw new Error(data.detail || 'Login failed');
      }

      const data = await response.json();

      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('user_id', data.user_id);
      localStorage.setItem('user_name', data.name);

      const redirect = searchParams.get('redirect') || '/dashboard/chat';
      router.push(redirect);
    } catch (err) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-[#18181B] text-[#F4F4F5]">
      {/* ambient corner glow, derived from background token, kept as accent not flat fill */}
      <div
        className="pointer-events-none absolute -left-40 -top-40 h-120 w-120 rounded-full opacity-20 blur-[120px]"
        style={{ background: '#2EE496' }}
      />
      <div
        className="pointer-events-none absolute -bottom-40 -right-40 h-105 w-105 rounded-full opacity-10 blur-[120px]"
        style={{ background: '#047857' }}
      />

      <div className="relative z-10 mx-auto flex min-h-screen max-w-350 flex-col lg:flex-row">
        {/* Left: kinetic network panel */}
        <div className="relative hidden flex-1 overflow-hidden border-r border-[#27272A] lg:flex lg:flex-col lg:justify-between lg:p-20">
          <KineticNetwork active={!!focusedField} />

          <div
            className={`relative z-10 transition-all duration-700 ease-out ${
              mounted ? 'translate-y-0 opacity-100' : 'translate-y-3 opacity-0'
            }`}
          >
            <span
              className="inline-flex items-center gap-2 rounded-full border border-[#27272A] px-3 py-1 text-[12px] font-semibold tracking-[0.08em]"
              style={{ fontFamily: 'JetBrains Mono, monospace', color: '#34D399' }}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-[#34D399] shadow-[0_0_8px_#34D399]" />
              PIPELINE_STATUS: ONLINE
            </span>
          </div>

          <div
            className={`relative z-10 max-w-md transition-all delay-150 duration-700 ease-out ${
              mounted ? 'translate-y-0 opacity-100' : 'translate-y-3 opacity-0'
            }`}
          >
            <h1
              className="font-medium leading-[1.04] tracking-tight text-white"
              style={{ fontFamily: 'Inter, sans-serif', fontSize: '48px' }}
            >
              Index the knowledge.
              <br />
              Retrieve with precision.
              <br />
              <span style={{ color: '#34D399' }}>Answer with confidence.</span>
            </h1>
            <p
              className="mt-4 text-[15px] leading-relaxed"
              style={{ color: '#A1A1AA', fontFamily: 'Inter, sans-serif' }}
            >
              An agentic RAG system that searches your documents, the web, and
              memory together, then verifies every answer before it reaches you.
            </p>

            <div
              className="mt-8 grid grid-cols-2 gap-3 text-[12px]"
              style={{ fontFamily: 'JetBrains Mono, monospace' }}
            >
              {[
                { label: 'Planner Agent', active: true },
                { label: 'Retriever Agent', active: false },
                { label: 'Critic Agent', active: false },
                { label: 'Answer Agent', active: false },
              ].map((agent) => (
                <div
                  key={agent.label}
                  className="flex items-center gap-2 rounded-[11px] border border-[#27272A] px-3 py-2 text-[#A1A1AA]"
                >
                  <span
                    className="h-1.5 w-1.5 shrink-0 rounded-full"
                    style={{ background: agent.active ? '#34D399' : '#3F3F46' }}
                  />
                  {agent.label}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: auth card */}
        <div className="flex w-full flex-1 items-center justify-center px-6 py-16 lg:max-w-130 lg:px-20">
          <div
            className={`w-full max-w-sm transition-all duration-700 ease-out ${
              mounted ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
            }`}
          >
            <div className="mb-8">
              <span
                className="text-[12px] font-semibold tracking-widest"
                style={{ fontFamily: 'JetBrains Mono, monospace', color: '#34D399' }}
              >
                RAG 2.0 / ACCESS
              </span>
              <h2
                className="mt-3 text-[28px] font-medium tracking-tight text-white"
                style={{ fontFamily: 'Inter, sans-serif' }}
              >
                Sign in to your workspace
              </h2>
              <p className="mt-1 text-[14px]" style={{ color: '#A1A1AA' }}>
                Authenticate to resume your sessions, documents, and memory.
              </p>
            </div>

            <form className="space-y-5" onSubmit={handleLogin}>
              {error && (
                <div
                  className="rounded-[11px] border px-4 py-3 text-[13px]"
                  style={{
                    borderColor: 'rgba(248,113,113,0.3)',
                    background: 'rgba(248,113,113,0.08)',
                    color: '#FCA5A5',
                  }}
                >
                  {error}
                </div>
              )}

              <div>
                <label
                  htmlFor="email"
                  className="mb-2 block text-[11px] font-semibold uppercase tracking-[0.08em]"
                  style={{ fontFamily: 'JetBrains Mono, monospace', color: '#71717A' }}
                >
                  Email address
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  required
                  autoComplete="email"
                  placeholder="you@company.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onFocus={() => setFocusedField('email')}
                  onBlur={() => setFocusedField(null)}
                  className="w-full rounded-[11px] border bg-[#0F0F11] px-4 py-3 text-[15px] text-white placeholder-[#52525B] outline-none transition-all duration-200"
                  style={{
                    borderColor: focusedField === 'email' ? '#34D399' : '#27272A',
                    boxShadow:
                      focusedField === 'email' ? '0 0 0 3px rgba(52,211,153,0.15)' : 'none',
                  }}
                />
              </div>

              <div>
                <label
                  htmlFor="password"
                  className="mb-2 block text-[11px] font-semibold uppercase tracking-[0.08em]"
                  style={{ fontFamily: 'JetBrains Mono, monospace', color: '#71717A' }}
                >
                  Password
                </label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  required
                  autoComplete="current-password"
                  placeholder="••••••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onFocus={() => setFocusedField('password')}
                  onBlur={() => setFocusedField(null)}
                  className="w-full rounded-[11px] border bg-[#0F0F11] px-4 py-3 text-[15px] text-white placeholder-[#52525B] outline-none transition-all duration-200"
                  style={{
                    borderColor: focusedField === 'password' ? '#34D399' : '#27272A',
                    boxShadow:
                      focusedField === 'password' ? '0 0 0 3px rgba(52,211,153,0.15)' : 'none',
                  }}
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="group relative flex w-full items-center justify-center overflow-hidden rounded-[11px] py-3 text-[15px] font-medium text-[#022C22] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_8px_24px_rgba(52,211,153,0.35)] disabled:translate-y-0 disabled:opacity-60 disabled:hover:shadow-none"
                style={{ background: '#34D399', fontFamily: 'Inter, sans-serif' }}
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#022C22]" />
                    Verifying credentials…
                  </span>
                ) : (
                  'Sign in'
                )}
              </button>

              <div className="pt-2 text-center text-[14px]" style={{ color: '#71717A' }}>
                Don&apos;t have an account?{' '}
                <Link
                  href="/auth/register"
                  className="font-medium transition-colors hover:text-[#34D399]"
                  style={{ color: '#34D399' }}
                >
                  Register
                </Link>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}