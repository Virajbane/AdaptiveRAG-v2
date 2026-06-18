import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'RAG 2.0 System',
  description: 'Enterprise-grade Adaptive RAG',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  )
}