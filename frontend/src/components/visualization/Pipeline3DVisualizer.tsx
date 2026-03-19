'use client'

import { useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, Clock, AlertCircle, Zap, Brain, Database, Cpu, GitBranch, Sliders, BarChart3 } from 'lucide-react'

interface PipelineNode {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
}

interface Pipeline3DVisualizerProps {
  stages: PipelineNode[]
}

const STAGE_ICONS = [Database, Brain, GitBranch, Cpu, Sliders, BarChart3]

const STATUS_COLORS = {
  completed: { bg: 'from-emerald-500/30 to-teal-600/20', border: '#10b981', glow: 'rgba(16,185,129,0.6)', text: 'text-emerald-400', particle: '#10b981' },
  running:   { bg: 'from-violet-500/40 to-cyan-600/30',  border: '#8b5cf6', glow: 'rgba(139,92,246,0.8)',  text: 'text-violet-300', particle: '#06b6d4' },
  failed:    { bg: 'from-red-500/30 to-rose-600/20',     border: '#ef4444', glow: 'rgba(239,68,68,0.6)',   text: 'text-red-400',    particle: '#ef4444' },
  pending:   { bg: 'from-slate-700/30 to-slate-800/20',  border: '#475569', glow: 'rgba(71,85,105,0.3)',   text: 'text-slate-400',  particle: '#475569' },
}

// Animated particle stream between two nodes
function ParticleStream({ active, completed }: { active: boolean; completed: boolean }) {
  const count = 5
  return (
    <div className="relative w-16 h-1 mx-1 flex items-center">
      {/* Base line */}
      <div className={`absolute inset-0 rounded-full transition-all duration-700 ${
        completed ? 'bg-gradient-to-r from-emerald-500/60 to-teal-500/60' :
        active    ? 'bg-gradient-to-r from-violet-500/40 to-cyan-500/40' :
                    'bg-slate-700/40'
      }`} />

      {/* Flowing particles */}
      {(active || completed) && Array.from({ length: count }).map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1.5 h-1.5 rounded-full"
          style={{ background: completed ? '#10b981' : '#8b5cf6', top: '-2px' }}
          animate={{ x: [0, 56, 56], opacity: [0, 1, 0] }}
          transition={{
            duration: completed ? 1.2 : 0.9,
            delay: i * (completed ? 0.24 : 0.18),
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}

      {/* Arrow tip */}
      <div className={`absolute right-0 w-0 h-0 border-t-4 border-b-4 border-l-6 border-transparent transition-colors duration-500 ${
        completed ? 'border-l-emerald-400' : active ? 'border-l-violet-400' : 'border-l-slate-600'
      }`}
        style={{ borderLeftWidth: 8, borderLeftColor: completed ? '#10b981' : active ? '#8b5cf6' : '#475569' }}
      />
    </div>
  )
}

// Orbital ring for running node
function OrbitalRing({ color }: { color: string }) {
  return (
    <>
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="absolute inset-0 rounded-2xl border"
          style={{ borderColor: color, opacity: 0.15 + i * 0.1 }}
          animate={{ scale: [1, 1.08 + i * 0.06, 1], opacity: [0.15 + i * 0.1, 0.4, 0.15 + i * 0.1] }}
          transition={{ duration: 1.6 + i * 0.4, repeat: Infinity, delay: i * 0.3, ease: 'easeInOut' }}
        />
      ))}
      {/* Spinning orbit dot */}
      <motion.div
        className="absolute w-2 h-2 rounded-full"
        style={{ background: color, top: -4, left: '50%', marginLeft: -4, boxShadow: `0 0 8px ${color}` }}
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        transformTemplate={({ rotate }) => `rotate(${rotate}) translateX(72px)`}
      />
    </>
  )
}

// Hex grid background canvas
function HexBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resize = () => { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight }
    resize()
    window.addEventListener('resize', resize)

    let t = 0
    let raf: number

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const size = 28
      const cols = Math.ceil(canvas.width / (size * 1.75)) + 2
      const rows = Math.ceil(canvas.height / (size * 1.5)) + 2

      for (let row = -1; row < rows; row++) {
        for (let col = -1; col < cols; col++) {
          const x = col * size * 1.75 + (row % 2 === 0 ? 0 : size * 0.875)
          const y = row * size * 1.5
          const dist = Math.hypot(x - canvas.width / 2, y - canvas.height / 2)
          const wave = Math.sin(dist * 0.015 - t * 0.04) * 0.5 + 0.5
          const alpha = wave * 0.07 + 0.02

          ctx.beginPath()
          for (let i = 0; i < 6; i++) {
            const angle = (Math.PI / 3) * i - Math.PI / 6
            const px = x + size * 0.85 * Math.cos(angle)
            const py = y + size * 0.85 * Math.sin(angle)
            i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)
          }
          ctx.closePath()
          ctx.strokeStyle = `rgba(139,92,246,${alpha})`
          ctx.lineWidth = 0.8
          ctx.stroke()
        }
      }
      t++
      raf = requestAnimationFrame(draw)
    }

    draw()
    return () => { window.removeEventListener('resize', resize); cancelAnimationFrame(raf) }
  }, [])

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
}

export default function Pipeline3DVisualizer({ stages }: Pipeline3DVisualizerProps) {
  const anyRunning = stages.some((s) => s.status === 'running')

  return (
    <div className="relative w-full min-h-56 flex items-center justify-center overflow-hidden rounded-xl py-10 px-4">
      {/* Hex grid background */}
      <HexBackground />

      {/* Ambient glow blobs */}
      <motion.div
        className="absolute w-64 h-64 rounded-full pointer-events-none"
        style={{ background: 'radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%)', left: '10%', top: '20%' }}
        animate={{ scale: [1, 1.3, 1], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute w-48 h-48 rounded-full pointer-events-none"
        style={{ background: 'radial-gradient(circle, rgba(6,182,212,0.1) 0%, transparent 70%)', right: '10%', bottom: '10%' }}
        animate={{ scale: [1.2, 1, 1.2], opacity: [0.5, 0.9, 0.5] }}
        transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
      />

      {/* Nodes row */}
      <div className="relative z-10 flex items-center gap-0">
        {stages.map((stage, index) => {
          const colors = STATUS_COLORS[stage.status]
          const Icon = STAGE_ICONS[index] ?? Cpu
          const prevCompleted = index === 0 || stages[index - 1].status === 'completed'
          const streamActive = stage.status === 'running' || (stage.status === 'completed' && index < stages.length - 1)

          return (
            <div key={stage.id} className="flex items-center">
              {/* Node */}
              <motion.div
                custom={index}
                initial={{ opacity: 0, y: 40, scale: 0.6 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ delay: index * 0.12, duration: 0.6, ease: [0.34, 1.56, 0.64, 1] }}
                className="relative"
              >
                {/* Outer glow */}
                <motion.div
                  className="absolute -inset-3 rounded-3xl pointer-events-none"
                  style={{ background: `radial-gradient(circle, ${colors.glow} 0%, transparent 70%)` }}
                  animate={stage.status === 'running'
                    ? { opacity: [0.4, 1, 0.4], scale: [0.9, 1.1, 0.9] }
                    : stage.status === 'completed'
                    ? { opacity: [0.3, 0.6, 0.3] }
                    : { opacity: 0.2 }
                  }
                  transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
                />

                {/* Card */}
                <motion.div
                  whileHover={{ scale: 1.08, rotateY: 8, rotateX: -4 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                  className={`relative w-28 h-28 rounded-2xl bg-gradient-to-br ${colors.bg} border overflow-hidden cursor-pointer`}
                  style={{
                    borderColor: colors.border,
                    boxShadow: `0 0 20px ${colors.glow}, 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08)`,
                    transformStyle: 'preserve-3d',
                  }}
                >
                  {/* Shimmer sweep on running */}
                  {stage.status === 'running' && (
                    <motion.div
                      className="absolute inset-0 pointer-events-none"
                      style={{ background: 'linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.08) 50%, transparent 60%)' }}
                      animate={{ x: ['-100%', '200%'] }}
                      transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut', repeatDelay: 0.4 }}
                    />
                  )}

                  {/* Completed checkmark flash */}
                  {stage.status === 'completed' && (
                    <motion.div
                      className="absolute inset-0 bg-emerald-400/10 pointer-events-none"
                      initial={{ opacity: 1 }}
                      animate={{ opacity: 0 }}
                      transition={{ duration: 1.2, delay: index * 0.1 }}
                    />
                  )}

                  {/* Orbital rings for running */}
                  {stage.status === 'running' && <OrbitalRing color={colors.border} />}

                  {/* Content */}
                  <div className="relative h-full flex flex-col items-center justify-center gap-1.5 px-2">
                    {/* Icon */}
                    <motion.div
                      animate={stage.status === 'running'
                        ? { rotate: [0, 10, -10, 0], scale: [1, 1.15, 1] }
                        : stage.status === 'completed'
                        ? { scale: [1, 1.2, 1] }
                        : {}
                      }
                      transition={{ duration: stage.status === 'running' ? 1.2 : 0.5, repeat: stage.status === 'running' ? Infinity : 0 }}
                    >
                      {stage.status === 'completed' ? (
                        <CheckCircle size={22} className="text-emerald-400" />
                      ) : stage.status === 'failed' ? (
                        <AlertCircle size={22} className="text-red-400" />
                      ) : stage.status === 'running' ? (
                        <motion.div animate={{ rotate: 360 }} transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}>
                          <Zap size={22} className="text-violet-300" />
                        </motion.div>
                      ) : (
                        <Icon size={22} className="text-slate-400" />
                      )}
                    </motion.div>

                    {/* Label */}
                    <p className={`text-xs font-bold text-center leading-tight ${colors.text}`}>{stage.name}</p>

                    {/* Status dot row */}
                    <div className="flex gap-0.5 mt-0.5">
                      {[0, 1, 2].map((d) => (
                        <motion.div
                          key={d}
                          className="w-1 h-1 rounded-full"
                          style={{ background: colors.border }}
                          animate={stage.status === 'running'
                            ? { opacity: [0.3, 1, 0.3], scale: [0.8, 1.2, 0.8] }
                            : { opacity: stage.status === 'completed' ? 1 : 0.3 }
                          }
                          transition={{ duration: 0.8, repeat: stage.status === 'running' ? Infinity : 0, delay: d * 0.2 }}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Bottom progress bar for running */}
                  {stage.status === 'running' && (
                    <motion.div
                      className="absolute bottom-0 left-0 h-0.5 bg-gradient-to-r from-violet-500 to-cyan-400"
                      animate={{ width: ['0%', '100%', '0%'] }}
                      transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                    />
                  )}
                  {stage.status === 'completed' && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-emerald-500 to-teal-400" />
                  )}
                </motion.div>

                {/* Step number badge */}
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: index * 0.12 + 0.3, type: 'spring' }}
                  className="absolute -top-2 -right-2 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold z-10"
                  style={{
                    background: colors.border,
                    boxShadow: `0 0 8px ${colors.glow}`,
                    color: '#000',
                  }}
                >
                  {index + 1}
                </motion.div>
              </motion.div>

              {/* Connector stream */}
              {index < stages.length - 1 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: index * 0.12 + 0.5 }}
                >
                  <ParticleStream
                    active={stage.status === 'running' || stages[index + 1].status === 'running'}
                    completed={stage.status === 'completed'}
                  />
                </motion.div>
              )}
            </div>
          )
        })}
      </div>

      {/* Loop indicator when pipeline is cycling */}
      {anyRunning && (
        <motion.div
          className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-2 px-3 py-1 rounded-full border border-violet-500/30 bg-violet-500/10"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
        >
          <motion.div
            className="w-1.5 h-1.5 rounded-full bg-violet-400"
            animate={{ scale: [1, 1.6, 1], opacity: [0.6, 1, 0.6] }}
            transition={{ duration: 1, repeat: Infinity }}
          />
          <span className="text-xs text-violet-300 font-medium">Pipeline running</span>
          <motion.div className="flex gap-0.5">
            {[0, 1, 2].map((i) => (
              <motion.span key={i} className="text-violet-400 text-xs"
                animate={{ opacity: [0, 1, 0] }}
                transition={{ duration: 1, repeat: Infinity, delay: i * 0.25 }}
              >.</motion.span>
            ))}
          </motion.div>
        </motion.div>
      )}
    </div>
  )
}
