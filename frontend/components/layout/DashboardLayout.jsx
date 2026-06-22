// frontend/components/layout/DashboardLayout.jsx
'use client';

import { Navbar } from '@/components/common/Navbar';
import { Sidebar } from '@/components/common/Sidebar';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';

// Wraps every dashboard page with the same navbar + sidebar shell.
// The actual page content (chat, documents, search, memory) is
// passed in as `children` and rendered in the main scrollable area.
export function DashboardLayout({ children }) {
  return (
    <div className="h-screen flex flex-col">
      <Navbar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto bg-white">
          <ErrorBoundary>{children}</ErrorBoundary>
        </main>
      </div>
    </div>
  );
}