// frontend/app/auth/register/page.jsx
'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import KineticNetwork from '@/components/KineticNetwork';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function RegisterPage() {
  const router = useRouter();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [focusedField, setFocusedField] = useState(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Checks the password against the same rules the backend enforces.
  // Returns a plain object: { valid: true/false, message: '...' }
  const validatePassword = (pwd) => {
    if (pwd.length < 8)
      return {
        valid: false,
        message: 'Password must be at least 8 characters',
      };
    if (!/[A-Z]/.test(pwd))
      return {
        valid: false,
        message: 'Password must contain uppercase letter',
      };
    if (!/[a-z]/.test(pwd))
      return {
        valid: false,
        message: 'Password must contain lowercase letter',
      };
    if (!/[0-9]/.test(pwd))
      return { valid: false, message: 'Password must contain digit' };
    if (!/[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(pwd)) {
      return {
        valid: false,
        message: 'Password must contain special character',
      };
    }
    return { valid: true, message: '' };
  };

  // Live password requirement checklist, derived from the same rules above —
  // gives the person a visible readout instead of a single error after submit.
  const passwordChecks = [
    { label: '8+ characters', met: password.length >= 8 },
    { label: 'Uppercase letter', met: /[A-Z]/.test(password) },
    { label: 'Lowercase letter', met: /[a-z]/.test(password) },
    { label: 'Digit', met: /[0-9]/.test(password) },
    { label: 'Special character', met: /[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(password) },
  ];

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');

    // Validate on the frontend first, so the user gets instant feedback
    // before we even talk to the backend
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    const validation = validatePassword(password);
    if (!validation.valid) {
      setError(validation.message);
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/api/v1/auth/register`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password, name }),
        },
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Registration failed');
      }

      // Registration successful, redirect to login
      router.push('/auth/login?registered=true');
    } catch (err) {
      setError(err.message || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-[#18181B] text-[#F4F4F5]">
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
              PIPELINE_STATUS: PROVISIONING
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
              Provision a workspace.
              <br />
              Bring your knowledge.
              <br />
              <span style={{ color: '#34D399' }}>Answer with confidence.</span>
            </h1>
            <p
              className="mt-4 text-[15px] leading-relaxed"
              style={{ color: '#A1A1AA', fontFamily: 'Inter, sans-serif' }}
            >
              Create an account to upload documents, run hybrid search, and
              get cited, verified answers from your own data.
            </p>

            <div
              className="mt-8 grid grid-cols-2 gap-3 text-[12px]"
              style={{ fontFamily: 'JetBrains Mono, monospace' }}
            >
              {[
                { label: 'Planner Agent', active: false },
                { label: 'Retriever Agent', active: true },
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

        {/* Right: register card */}
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
                RAG 2.0 / REGISTER
              </span>
              <h2
                className="mt-3 text-[28px] font-medium tracking-tight text-white"
                style={{ fontFamily: 'Inter, sans-serif' }}
              >
                Create your account
              </h2>
              <p className="mt-1 text-[14px]" style={{ color: '#A1A1AA' }}>
                Set up your workspace for documents, chat, and memory.
              </p>
            </div>

            <form className="space-y-5" onSubmit={handleRegister}>
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
                  className="mb-2 block text-[11px] font-semibold uppercase tracking-[0.08em]"
                  style={{ fontFamily: 'JetBrains Mono, monospace', color: '#71717A' }}
                >
                  Full name
                </label>
                <input
                  type="text"
                  required
                  placeholder="Jordan Avery"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onFocus={() => setFocusedField('name')}
                  onBlur={() => setFocusedField(null)}
                  className="w-full rounded-[11px] border bg-[#0F0F11] px-4 py-3 text-[15px] text-white placeholder-[#52525B] outline-none transition-all duration-200"
                  style={{
                    borderColor: focusedField === 'name' ? '#34D399' : '#27272A',
                    boxShadow:
                      focusedField === 'name' ? '0 0 0 3px rgba(52,211,153,0.15)' : 'none',
                  }}
                />
              </div>

              <div>
                <label
                  className="mb-2 block text-[11px] font-semibold uppercase tracking-[0.08em]"
                  style={{ fontFamily: 'JetBrains Mono, monospace', color: '#71717A' }}
                >
                  Email address
                </label>
                <input
                  type="email"
                  required
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
                  className="mb-2 block text-[11px] font-semibold uppercase tracking-[0.08em]"
                  style={{ fontFamily: 'JetBrains Mono, monospace', color: '#71717A' }}
                >
                  Password
                </label>
                <input
                  type="password"
                  required
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
                {/* live requirement readout, mono labels per design system */}
                <div
                  className="mt-2 grid grid-cols-2 gap-1.5 text-[11px]"
                  style={{ fontFamily: 'JetBrains Mono, monospace' }}
                >
                  {passwordChecks.map((check) => (
                    <div
                      key={check.label}
                      className="flex items-center gap-1.5 transition-colors duration-200"
                      style={{ color: check.met ? '#34D399' : '#52525B' }}
                    >
                      <span
                        className="h-1 w-1 rounded-full transition-colors duration-200"
                        style={{ background: check.met ? '#34D399' : '#3F3F46' }}
                      />
                      {check.label}
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <label
                  className="mb-2 block text-[11px] font-semibold uppercase tracking-[0.08em]"
                  style={{ fontFamily: 'JetBrains Mono, monospace', color: '#71717A' }}
                >
                  Confirm password
                </label>
                <input
                  type="password"
                  required
                  placeholder="••••••••••••"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  onFocus={() => setFocusedField('confirm')}
                  onBlur={() => setFocusedField(null)}
                  className="w-full rounded-[11px] border bg-[#0F0F11] px-4 py-3 text-[15px] text-white placeholder-[#52525B] outline-none transition-all duration-200"
                  style={{
                    borderColor:
                      focusedField === 'confirm'
                        ? '#34D399'
                        : confirmPassword && confirmPassword !== password
                          ? 'rgba(248,113,113,0.5)'
                          : '#27272A',
                    boxShadow:
                      focusedField === 'confirm' ? '0 0 0 3px rgba(52,211,153,0.15)' : 'none',
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
                    Setting up workspace…
                  </span>
                ) : (
                  'Create account'
                )}
              </button>

              <div className="pt-2 text-center text-[14px]" style={{ color: '#71717A' }}>
                Already have an account?{' '}
                <Link
                  href="/auth/login"
                  className="font-medium transition-colors hover:text-[#34D399]"
                  style={{ color: '#34D399' }}
                >
                  Sign in
                </Link>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}