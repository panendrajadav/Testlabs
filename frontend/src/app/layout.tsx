import type { ReactNode } from 'react'
import './globals.css'
import { Metadata } from 'next'
import Providers from './providers'

export const metadata: Metadata = {
  title: 'TestLabs AutoML',
  description: 'Automated ML pipeline with intelligent agents',
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="bg-black text-white" suppressHydrationWarning>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
