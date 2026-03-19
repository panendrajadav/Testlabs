'use client'

import { type ReactNode, useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { usePathname } from 'next/navigation'
import Sidebar from '@/components/layout/Sidebar'
import TopNav from '@/components/layout/TopNav'

export default function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () => new QueryClient({
      defaultOptions: { queries: { staleTime: 1000 * 60 * 5, gcTime: 1000 * 60 * 10 } },
    })
  )
  const pathname = usePathname()
  const isFullScreen = pathname === '/'

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
