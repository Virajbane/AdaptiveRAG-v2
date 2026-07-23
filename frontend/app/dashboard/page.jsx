// app/dashboard/page.jsx
//
// Root index for /dashboard. Renders nothing itself -- app/dashboard/layout.jsx
// always renders <Dashboard/> directly regardless of {children} -- but this
// file must exist for Next.js to treat /dashboard as a valid route at all.
export default function DashboardIndexPage() {
  return null;
}