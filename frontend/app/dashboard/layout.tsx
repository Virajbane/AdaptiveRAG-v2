import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import { AuthProvider } from '@/app/context/AuthContext';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AuthProvider>
      <ProtectedRoute>{children}</ProtectedRoute>
    </AuthProvider>
  );
}