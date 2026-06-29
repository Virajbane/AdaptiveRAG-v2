// app/layout.jsx
import { Metadata } from 'next'
import { AuthProvider } from "@/app/context/AuthContext";
import ConditionalSiteBackground from "@/components/LightRays/ConditionalSiteBackground";
import './globals.css'

export const metadata = {
  title: 'RAG 2.0 System',
  description: 'Enterprise-grade Adaptive RAG',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ConditionalSiteBackground />
        <AuthProvider>
          <div style={{ position: 'relative', zIndex: 1 }}>{children}</div>
        </AuthProvider>
      </body>
    </html>
  );
}