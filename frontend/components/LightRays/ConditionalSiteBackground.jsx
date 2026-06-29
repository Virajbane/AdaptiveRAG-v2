'use client';

import { usePathname } from 'next/navigation';
import SiteBackground from './SiteBackground';

// Routes that bring their own background (KineticNetwork) and should
// never also render the global LightRays background underneath.
const EXCLUDED_PREFIXES = ['/auth/login', '/auth/register'];

export default function ConditionalSiteBackground() {
  const pathname = usePathname();

  const isExcluded = EXCLUDED_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`)
  );

  if (isExcluded) return null;

  return <SiteBackground />;
}