// Shared style tokens for the dashboard — colors, spacing, shadows.
// Pulled from DESIGN.md ("Aural Immersion" dark theme).
// Edit this file to retheme the whole dashboard in one place.

export const S = {
  root: {
    display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden',
    background: '#121212', color: '#e5e2e1',
    fontFamily: "'Hanken Grotesk', sans-serif",
  },
  sidebar: {
    width: 260, flexShrink: 0,
    background: '#0e0e0e',
    borderRight: '1px solid rgba(255,255,255,0.05)',
    display: 'flex', flexDirection: 'column',
    padding: '16px 12px', gap: 16, overflowY: 'auto',
  },
  brand: { display: 'flex', alignItems: 'center', gap: 10, padding: '4px 8px' },
  brandIcon: { width: 40, height: 40, borderRadius: 10, background: '#1ed760', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  brandName: { fontSize: 16, fontWeight: 700, color: '#fff', lineHeight: 1.2 },
  brandSub: { fontSize: 12, color: '#b3b3b3' },
  newQueryBtn: {
    width: '100%', padding: '11px 16px',
    borderRadius: 9999, border: 'none',
    background: '#fff', color: '#121212',
    fontSize: 14, fontWeight: 700, letterSpacing: '1.6px', textTransform: 'uppercase',
    cursor: 'pointer', transition: 'transform 0.15s',
  },
  navLabel: { fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', color: '#b3b3b3', textTransform: 'uppercase', padding: '0 8px', marginBottom: 4 },
  navItem: {
    display: 'flex', alignItems: 'center', gap: 12,
    width: '100%', padding: '9px 8px', borderRadius: 8, border: 'none',
    background: 'transparent', color: '#b3b3b3', cursor: 'pointer',
    fontSize: 14, textAlign: 'left', transition: 'background 0.1s, color 0.1s',
  },
  navItemActive: { background: '#252525', color: '#fff' },
  sidebarFooter: { borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: 12, display: 'flex', flexDirection: 'column', gap: 2 },
  main: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: '#121212' },
  topbar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '0 24px', height: 64, flexShrink: 0,
    background: 'rgba(19,19,19,0.7)', backdropFilter: 'blur(10px)',
    borderBottom: '1px solid rgba(255,255,255,0.05)',
    position: 'sticky', top: 0, zIndex: 20,
  },
  topbarTitle: { fontSize: 18, fontWeight: 700, color: '#fff' },
  topTab: {
    padding: '5px 14px', borderRadius: 9999, border: 'none',
    background: 'transparent', color: '#b3b3b3',
    fontSize: 14, fontWeight: 400, cursor: 'pointer', transition: 'all 0.15s',
  },
  topTabActive: { background: '#252525', color: '#fff', fontWeight: 700 },
  searchPill: {
    display: 'flex', alignItems: 'center', gap: 8,
    background: '#2a2a2a', borderRadius: 9999, padding: '7px 14px',
    boxShadow: 'rgb(18,18,18) 0px 1px 0px, rgb(77,77,77) 0px 0px 0px 1px inset',
  },
  searchPillInput: { background: 'transparent', border: 'none', outline: 'none', fontSize: 13, color: '#fff', width: 150 },
  iconBtn: {
    width: 36, height: 36, borderRadius: '50%', border: 'none',
    background: 'transparent', color: '#b3b3b3', cursor: 'pointer',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    transition: 'background 0.15s, color 0.15s',
  },
  avatarBtn: {
    width: 32, height: 32, borderRadius: '50%',
    background: '#353534', border: 'none', cursor: 'pointer',
    fontSize: 13, fontWeight: 700, color: '#e5e2e1',
  },
  avatarMenu: {
    position: 'absolute', right: 0, top: 'calc(100% + 8px)', zIndex: 50,
    width: 200, borderRadius: 12,
    background: '#1c1b1b', border: '1px solid rgba(255,255,255,0.08)',
    boxShadow: 'rgba(0,0,0,0.5) 0px 8px 24px', padding: 6,
  },
  avatarMenuHeader: { padding: '8px 12px 10px', borderBottom: '1px solid rgba(255,255,255,0.06)', marginBottom: 4 },
  avatarMenuItem: {
    display: 'block', width: '100%', textAlign: 'left',
    padding: '8px 12px', borderRadius: 8, border: 'none',
    background: 'transparent', cursor: 'pointer',
    fontSize: 14, color: '#e5e2e1', transition: 'background 0.1s',
  },
  card: {
    background: '#181818', borderRadius: 12,
    border: '1px solid rgba(255,255,255,0.06)',
  },
  pageTitle: { fontSize: 24, fontWeight: 700, color: '#fff', marginBottom: 6 },
  pageSubtitle: { fontSize: 14, color: '#b3b3b3', marginBottom: 28 },
};