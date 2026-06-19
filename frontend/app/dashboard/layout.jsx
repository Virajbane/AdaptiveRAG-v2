// frontend/app/dashboard/layout.jsx
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import { AuthProvider } from '@/app/context/AuthContext';

// This is a special Next.js file: layout.jsx wraps every page inside
// the "dashboard" folder automatically (documents, chat, settings, etc).
//
// AuthProvider makes the login token available to all pages inside.
// ProtectedRoute then checks that token and redirects to login if missing.
export default function DashboardLayout({ children }) {
  return (
    <AuthProvider>
      <ProtectedRoute>{children}</ProtectedRoute>
    </AuthProvider>
  );
}