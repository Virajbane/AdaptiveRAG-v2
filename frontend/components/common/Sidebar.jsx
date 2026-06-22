// frontend/components/common/Sidebar.jsx
'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

// Each entry maps a nav label to the route it links to.
// Add new dashboard pages here as they're built.
const NAV_ITEMS = [
  { label: 'Chat', href: '/dashboard/chat' },
  { label: 'Documents', href: '/dashboard/documents' },
  { label: 'Search', href: '/dashboard/search' },
  { label: 'Memory', href: '/dashboard/memory' },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <nav className="w-56 border-r border-gray-200 bg-gray-50 p-4 flex flex-col gap-1">
      {NAV_ITEMS.map((item) => {
        const isActive = pathname?.startsWith(item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
              isActive
                ? 'bg-gray-900 text-white'
                : 'text-gray-700 hover:bg-gray-200'
            }`}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}