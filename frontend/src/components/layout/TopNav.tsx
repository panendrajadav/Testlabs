'use client'

import { Bell, Search, Zap } from 'lucide-react'
import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'

export default function TopNav() {
  const [currentDataset, setCurrentDataset] = useState<{ filename: string; rows: number; columns: number } | null>(null)

  const readDataset = () => {
    const stored = localStorage.getItem('currentDataset')
    if (stored) setCurrentDataset(JSON.parse(stored))
    else setCurrentDataset(null)
  }

  useEffect(() => {
    readDataset()

    // cross-tab updates
    const handleStorage = () => readDataset()
    window.addEventListener('storage', handleStorage)

    // same-tab updates via custom event
    window.addEventListener('datasetUpdated', handleStorage)

    return () => {
      window.removeEventListener('storage', handleStorage)
      window.removeEventListener('datasetUpdated', handleStorage)
    }
  }, [])

  return (
    <header className="bg-gradient-to-r from-slate-950 via-slate-900 to-black border-b border-purple-500/20 glass px-8 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          {currentDataset ? (
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-cyan-500 rounded-lg flex items-center justify-center">
                <Zap size={16} className="text-white" />
              </div>
              <div className="flex flex-col">
                <p className="text-sm font-semibold text-white">{currentDataset.filename}</p>
                <p className="text-xs text-gray-400">
                  {currentDataset.rows} rows • {currentDataset.columns} columns
                </p>
              </div>
            </div>
          ) : (
            <p className="text-gray-400 text-sm">No dataset selected</p>
          )}
        </div>

        <motion.div
          className="hidden md:flex items-center gap-2 bg-white/5 px-4 py-2 rounded-lg border border-purple-500/20 flex-1 mx-8"
          whileHover={{ borderColor: 'rgba(139, 92, 246, 0.5)' }}
        >
          <Search size={16} className="text-gray-400" />
          <input
            type="text"
            placeholder="Search..."
            className="bg-transparent text-sm text-white placeholder-gray-400 outline-none w-full"
          />
        </motion.div>

        <div className="flex items-center gap-4">
          <motion.button whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.95 }} suppressHydrationWarning className="relative text-gray-400 hover:text-white">
            <Bell size={20} />
            <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full" />
          </motion.button>
          <motion.div
            whileHover={{ scale: 1.1 }}
            className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-semibold text-sm cursor-pointer"
          >
            U
          </motion.div>
        </div>
      </div>
    </header>
  )
}
