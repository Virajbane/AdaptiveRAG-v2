// frontend/components/common/Navbar.jsx
'use client';

import { useAuth } from '@/app/context/AuthContext';
import { useRouter } from 'next/navigation';

export function Navbar() {
  const { userName, logout } = useAuth();
  const router = useRouter();

  const handleLogout = () => {
    logout();
    router.push('/auth/login');
  };

  return (
    <header className="h-14 border-b border-gray-200 bg-white flex items-center justify-between px-6">
      <span className="font-semibold text-gray-900">RAG 2.0 System</span>

      <div className="flex items-center gap-4">
        {userName && (
          <span className="text-sm text-gray-600">
            Signed in as <span className="font-medium text-gray-900">{userName}</span>
          </span>
        )}
        <button
          onClick={handleLogout}
          className="text-sm font-medium text-gray-600 hover:text-gray-900"
        >
          Log out
        </button>
      </div>
    </header>
  );
}