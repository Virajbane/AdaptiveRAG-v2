// app/layout.jsx
import "./globals.css";
export const metadata = {
  title: 'RAG 2.0 System',
  description: 'Enterprise-grade Adaptive RAG',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}