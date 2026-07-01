import { ProtectedRoute } from '@/components/security/ProtectedRoute';
import Dashboard from '@/components/layout/dashboard/Dashboard';

// All views (Chat, Library, Search, History) now live inside the single
// Dashboard component as local-state view switches — no per-route pages.
export default function DashboardRootLayout() {
  return (
    <ProtectedRoute>
      <Dashboard />
    </ProtectedRoute>
  );
}