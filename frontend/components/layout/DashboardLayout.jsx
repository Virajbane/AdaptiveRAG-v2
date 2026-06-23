// frontend/components/layout/DashboardLayout.jsx
'use client';

import { Navbar } from '@/components/common/Navbar';
import { AppSidebar } from '@/components/common/Sidebar';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';
import {
  SidebarProvider,
  SidebarInset,
  SidebarTrigger,
} from '@/components/ui/sidebar';

// Wraps every dashboard page with the same navbar + sidebar shell.
// The actual page content (chat, documents, search, memory) is
// passed in as `children` and rendered in the main scrollable area.
export function DashboardLayout({ children }) {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset className="flex h-screen flex-col">
        <div className="flex items-center border-b border-gray-200">
          <SidebarTrigger className="ml-2" />
          <div className="flex-1">
            <Navbar />
          </div>
        </div>
        <main className="flex-1 overflow-y-auto bg-white">
          <ErrorBoundary>{children}</ErrorBoundary>
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}