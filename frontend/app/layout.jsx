// app/layout.jsx
import { Metadata } from 'next'
import { AuthProvider } from "@/app/context/AuthContext";
import './globals.css'
export const metadata = {
  title: 'RAG 2.0 System',
  description: 'Enterprise-grade Adaptive RAG',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body><AuthProvider>{children}</AuthProvider></body>
    </html>
  );
}