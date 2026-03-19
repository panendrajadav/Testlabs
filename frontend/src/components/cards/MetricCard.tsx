'use client'

import { motion } from 'framer-motion'
import { ReactNode } from 'react'

interface MetricCardProps {
  label: string
  value: string | number
  icon?: ReactNode
  color?: 'purple' | 'cyan' | 'green' | 'pink' | 'orange' | 'blue'
  trend?: number
  animate?: boolean
  index?: number
}

const colorClasses = {
  purple: 'from-purple-900/30 to-purple-800/20 border-purple-500/30',
  cyan: 'from-cyan-900/30 to-cyan-800/20 border-cyan-500/30',
  green: 'from-green-900/30 to-green-800/20 border-green-500/30',
  pink: 'from-pink-900/30 to-pink-800/20 border-pink-500/30',
  orange: 'from-orange-900/30 to-orange-800/20 border-orange-500/30',
  blue: 'from-blue-900/30 to-blue-800/20 border-blue-500/30',
}

const textClasses = {
  purple: 'text-purple-400',
  cyan: 'text-cyan-400',
  green: 'text-green-400',
  pink: 'text-pink-400',
  orange: 'text-orange-400',
  blue: 'text-blue-400',
}

export function MetricCard({
  label,
  value,
  icon,
  color = 'purple',
  trend,
  animate = true,
  index = 0,
}: MetricCardProps) {
  return (
    <motion.div
      initial={animate ? { opacity: 0, scale: 0.8 } : {}}
      animate={animate ? { opacity: 1, scale: 1 } : {}}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      whileHover={animate ? { y: -5 } : {}}
      className={`bg-gradient-to-br ${colorClasses[color]} rounded-lg border glass p-6`}
    >
      <div className="flex items-center justify-between mb-2">
        <p className="text-gray-400 text-sm">{label}</p>
        {icon && <div className={textClasses[color]}>{icon}</div>}
      </div>

      <motion.p
        initial={animate ? { opacity: 0, scale: 0.5 } : {}}
        animate={animate ? { opacity: 1, scale: 1 } : {}}
        transition={{ duration: 0.5, delay: index * 0.1 }}
        className={`text-3xl font-bold ${textClasses[color]}`}
      >
        {value}
      </motion.p>

      {trend !== undefined && (
        <p
          className={`text-xs mt-2 ${
            trend > 0 ? 'text-green-400' : 'text-red-400'
          }`}
        >
          {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}%
        </p>
      )}
    </motion.div>
  )
}

interface StatsPanelProps {
  stats: Array<{
    label: string
    value: string | number
    icon?: ReactNode
    color?: 'purple' | 'cyan' | 'green' | 'pink' | 'orange' | 'blue'
    trend?: number
  }>
}

export function StatsPanel({ stats }: StatsPanelProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, i) => (
        <MetricCard
          key={i}
          {...stat}
          index={i}
          animate
        />
      ))}
    </div>
  )
}
