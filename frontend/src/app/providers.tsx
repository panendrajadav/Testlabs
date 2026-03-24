'use client'

import { type ReactNode, useState, useEffect } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { usePathname } from 'next/navigation'
import Sidebar from '@/components/layout/Sidebar'
import TopNav from '@/components/layout/TopNav'

function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname()
  const [isFullScreen, setIsFullScreen] = useState(true)

  useEffect(() => {
    setIsFullScreen(pathname === '/' || pathname === '/login')
  }, [pathname])

  if (isFullScreen) return <>{children}</>

  return (
    <div className="flex h-screen bg-black overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <TopNav />
        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  )
}

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

  return (
    <QueryClientProvider client={queryClient}>
      <AppShell>{children}</AppShell>
    </QueryClientProvider>
  )
}
