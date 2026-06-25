import { AuthProvider } from '@/app/context/AuthContext';
import { ProtectedRoute } from '@/components/security/ProtectedRoute';
import { DashboardLayout } from '@/components/layout/DashboardLayout';

export default function DashboardRootLayout({ children }) {
  return (
    <AuthProvider>
      <ProtectedRoute>
        <DashboardLayout>{children}</DashboardLayout>
      </ProtectedRoute>
    </AuthProvider>
  );
}