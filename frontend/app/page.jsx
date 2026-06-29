// app/page.tsx
import Link from 'next/link';

export default function Home() {
  return (
    <main className="flex min-h-screen items-center justify-center text-[#F4F4F5]">
      <div className="text-center px-6">
        <span
          className="inline-flex items-center gap-2 rounded-full border border-[#27272A] bg-[#18181B]/60 px-3 py-1 text-[12px] font-semibold tracking-widest"
          style={{ fontFamily: 'JetBrains Mono, monospace', color: '#34D399' }}
        >
          <span className="h-1.5 w-1.5 rounded-full bg-[#34D399] shadow-[0_0_8px_#34D399]" />
          PIPELINE_STATUS: ONLINE
        </span>

        <h1
          className="mt-6 text-5xl font-medium tracking-tight text-white"
          style={{ fontFamily: 'Inter, sans-serif' }}
        >
          RAG 2.0 System
        </h1>
        <p
          className="mt-3 text-lg"
          style={{ color: '#A1A1AA', fontFamily: 'Inter, sans-serif' }}
        >
          Enterprise-grade Adaptive Retrieval Augmented Generation
        </p>

        <div className="mt-10 flex justify-center gap-4">
          <Link
            href="/auth/login"
            className="rounded-[11px] px-6 py-2.5 text-[15px] font-medium text-[#022C22] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_8px_24px_rgba(52,211,153,0.35)]"
            style={{ background: '#34D399', fontFamily: 'Inter, sans-serif' }}
          >
            Login
          </Link>

          <Link
            href="/auth/register"
            className="rounded-[11px] border border-[#27272A] px-6 py-2.5 text-[15px] font-medium text-white transition-all duration-200 hover:-translate-y-0.5 hover:border-[#34D399]"
            style={{ fontFamily: 'Inter, sans-serif' }}
          >
            Register
          </Link>
        </div>
      </div>
    </main>
  );
}