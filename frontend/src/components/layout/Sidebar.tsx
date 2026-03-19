'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  BarChart3,
  Upload,
  Zap,
  TrendingUp,
  MessageSquare,
  Settings,
  LogOut,
  Home,
} from 'lucide-react'
import { slideInFromLeftVariants } from '@/animations/variants'

const sidebarItems = [
  { id: 'dashboard', label: 'Dashboard', icon: Home, href: '/' },
  { id: 'upload', label: 'Upload Dataset', icon: Upload, href: '/upload' },
  { id: 'pipeline', label: 'Pipeline', icon: Zap, href: '/pipeline' },
  { id: 'models', label: 'Models', icon: TrendingUp, href: '/models' },
  { id: 'chat', label: 'Dataset Chat', icon: MessageSquare, href: '/chat' },
]

export default function Sidebar() {
  const pathname = usePathname()

  return (
    <motion.aside
      initial="hidden"
      animate="visible"
      variants={slideInFromLeftVariants}
      className="w-64 bg-gradient-to-b from-slate-950 via-slate-900 to-black border-r border-purple-500/20 glass flex flex-col"
    >
      {/* Logo */}
      <div className="p-6 flex items-center gap-3">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
          className="w-8 h-8 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-lg"
        />
        <div className="flex flex-col">
          <h1 className="text-lg font-bold text-white">AutoML</h1>
          <p className="text-xs text-purple-400">Dashboard</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-2">
        {sidebarItems.map((item, index) => {
          const Icon = item.icon
          const isActive = pathname === item.href
          return (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Link href={item.href}>
                <motion.div
                  whileHover={{ x: 5 }}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                    isActive
                      ? 'bg-purple-600/30 text-purple-300 border border-purple-500/50 glow'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon size={20} />
                  <span className="font-medium">{item.label}</span>
                  {isActive && (
                    <motion.div
                      layoutId="activeIndicator"
                      className="ml-auto w-1 h-1 bg-purple-400 rounded-full"
                    />
                  )}
                </motion.div>
              </Link>
            </motion.div>
          )
        })}
      </nav>

      {/* Bottom Section */}
      <div className="p-4 space-y-2 border-t border-purple-500/20">
        <motion.button
          whileHover={{ x: 5 }}
          suppressHydrationWarning
          className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all"
        >
          <Settings size={20} />
          <span>Settings</span>
        </motion.button>
        <motion.button
          whileHover={{ x: 5 }}
          suppressHydrationWarning
          className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-gray-400 hover:text-red-400 hover:bg-red-500/5 transition-all"
        >
          <LogOut size={20} />
          <span>Logout</span>
        </motion.button>
      </div>
    </motion.aside>
  )
}
