'use client';

import LightRays from './LightRays';

/**
 * Fixed, full-viewport ambient background for the app shell.
 * Sits behind all page content (z-index handled via the wrapper below,
 * LightRays itself stays pointer-events: none so it never blocks clicks).
 *
 * Deliberately excluded from /auth/login and /auth/register, which keep
 * their own KineticNetwork background instead — see app/layout.jsx.
 */
export default function SiteBackground() {
  return (
    <div
      aria-hidden="true"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 0,
        background: '#18181B',
        overflow: 'hidden',
      }}
    >
      <LightRays
        raysOrigin="top-center"
        raysColor="#34D399"
        raysSpeed={1.1}
        lightSpread={0.7}
        rayLength={1.4}
        followMouse={true}
        mouseInfluence={0.12}
        noiseAmount={0.06}
        distortion={0.03}
        saturation={1.0}
        fadeDistance={1.0}
      />
    </div>
  );
}