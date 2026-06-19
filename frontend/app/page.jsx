// app/page.tsx
export default function Home() {
  return (
    <main className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">RAG 2.0 System</h1>
        <p className="text-xl text-gray-600">
          Enterprise-grade Adaptive Retrieval Augmented Generation
        </p>
        <div className="mt-8 space-x-4">
          <a
            href="/auth/login"
            className="px-6 py-2 bg-blue-600 text-white rounded"
          >
            Login
          </a>

          <a
            href="/auth/register"
            className="px-6 py-2 bg-gray-600 text-white rounded"
          >
            Register
          </a>
        </div>
      </div>
    </main>
  )
}