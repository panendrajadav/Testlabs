import type { ReactNode } from 'react'
import './globals.css'
import { Metadata } from 'next'
import Providers from './providers'

export const metadata: Metadata = {
  title: 'TestLabs AutoML',
  description: 'Automated ML pipeline with intelligent agents',
  icons: {
    icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%238b5cf6'/><path d='M8 16 L14 10 L20 16 L14 22Z' fill='%2306b6d4'/><circle cx='22' cy='16' r='4' fill='%23fff' opacity='.9'/></svg>",
  },
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
