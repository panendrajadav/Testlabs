'use client'

import { type ReactNode, useState, useEffect } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { usePathname } from 'next/navigation'
import Sidebar from '@/components/layout/Sidebar'
import TopNav from '@/components/layout/TopNav'

export default function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () => new QueryClient({
      defaultOptions: {
        queries: {
          staleTime: 1000 * 60 * 5,
          gcTime: 1000 * 60 * 15,
          refetchOnWindowFocus: false,
          refetchOnMount: false,
        },
      },
    })
  )
  const pathname = usePathname()
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])

  const isFullScreen = pathname === '/' || pathname === '/login'

  // Render children only — no layout shell until client has mounted.
  // This ensures SSR and first client render always agree, preventing
  // hydration mismatches caused by localStorage reads in Sidebar.
  if (!mounted) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    )
  }

  return (
    <QueryClientProvider client={queryClient}>
      {isFullScreen ? (
        children
      ) : (
        <div className="flex h-screen bg-black overflow-hidden">
          <Sidebar />
          <div className="flex-1 flex flex-col min-w-0">
            <TopNav />
            <main className="flex-1 overflow-y-auto">{children}</main>
          </div>
        </div>
      )}
    </QueryClientProvider>
  )
}
