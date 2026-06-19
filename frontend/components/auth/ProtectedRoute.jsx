// frontend/components/auth/ProtectedRoute.jsx
'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/app/context/AuthContext';

// This component wraps any page that should only be visible to logged-in users.
// If the user is NOT authenticated, it redirects them to the login page.
// While we wait to find out if they're logged in, it shows "Loading...".
export function ProtectedRoute({ children }) {
  const router = useRouter();
  const { isAuthenticated } = useAuth();

  // Runs whenever isAuthenticated changes (e.g. right after AuthContext
  // finishes checking localStorage for a saved token)
  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/auth/login');
    }
  }, [isAuthenticated, router]);

  // While not authenticated (or still checking), don't render the protected page
  if (!isAuthenticated) {
    return <div>Loading...</div>;
  }

  // User is authenticated - render whatever page/content was passed in
  return <>{children}</>;
}