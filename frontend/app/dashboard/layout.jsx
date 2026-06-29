import { ProtectedRoute } from '@/components/security/ProtectedRoute';
import { DashboardLayout } from '@/components/layout/DashboardLayout';

// Note: AuthProvider is already provided once in the root app/layout.jsx.
// Wrapping it again here would create a second, disconnected auth context —
// removed so the whole app shares a single source of truth for the token/user.
export default function DashboardRootLayout({ children }) {
  return (
    <ProtectedRoute>
      <DashboardLayout>{children}</DashboardLayout>
    </ProtectedRoute>
  );
}