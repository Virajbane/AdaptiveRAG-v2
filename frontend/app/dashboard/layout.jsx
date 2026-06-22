// frontend/app/dashboard/layout.jsx
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import { AuthProvider } from '@/app/context/AuthContext';
import { DashboardLayout } from '@/components/layout/DashboardLayout';

// Wraps every page under /dashboard with:
// 1. AuthProvider - makes the login token available everywhere inside
// 2. ProtectedRoute - redirects to login if not authenticated
// 3. DashboardLayout - the navbar + sidebar shell around the page content
export default function DashboardRootLayout({ children }) {
  return (
    <AuthProvider>
      <ProtectedRoute>
        <DashboardLayout>{children}</DashboardLayout>
      </ProtectedRoute>
    </AuthProvider>
  );
}