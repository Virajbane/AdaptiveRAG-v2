// frontend/components/KineticNetwork.jsx
'use client';

import { useEffect, useRef } from 'react';

/**
 * Ambient animated node network — the visual signature of the AegisNode
 * auth surface. Nodes drift slowly and connect to nearby neighbors with
 * thin lines; lines brighten briefly when a connection "pulses", echoing
 * a network handshake. Pure canvas, no dependency weight.
 */
export default function KineticNetwork({ active = false }) {
  const canvasRef = useRef(null);
  const stateRef = useRef({ nodes: [], pulses: [] });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf;
    let width = 0;
    let height = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const resize = () => {
      const rect = canvas.parentElement.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const NODE_COUNT = 34;
    const initNodes = () => {
      stateRef.current.nodes = Array.from({ length: NODE_COUNT }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.12,
        vy: (Math.random() - 0.5) * 0.12,
        r: Math.random() * 1.6 + 1,
        pulse: Math.random() * Math.PI * 2,
      }));
    };

    resize();
    initNodes();

    const onResize = () => {
      resize();
      initNodes();
    };
    window.addEventListener('resize', onResize);

    const MAX_DIST = 150;

    const tick = (t) => {
      ctx.clearRect(0, 0, width, height);
      const { nodes } = stateRef.current;

      // node movement
      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < 0 || n.x > width) n.vx *= -1;
        if (n.y < 0 || n.y > height) n.vy *= -1;
      }

      // connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < MAX_DIST) {
            const baseAlpha = (1 - dist / MAX_DIST) * (active ? 0.22 : 0.13);
            ctx.strokeStyle = `rgba(52, 211, 153, ${baseAlpha})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      // nodes
      for (const n of nodes) {
        const glow = 0.5 + Math.sin(t / 900 + n.pulse) * 0.5;
        ctx.beginPath();
        ctx.fillStyle = `rgba(52, 211, 153, ${0.35 + glow * 0.4})`;
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fill();
      }

      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
    };
  }, [active]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 h-full w-full"
      aria-hidden="true"
    />
  );
}