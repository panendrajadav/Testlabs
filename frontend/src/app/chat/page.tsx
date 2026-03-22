'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence, useMotionValue, useTransform } from 'framer-motion'
import dynamic from 'next/dynamic'
import { useChat } from '@/hooks/useApi'
import { useStoredDataset } from '@/hooks/useStoredDataset'
import { Send, Sparkles, BarChart2, Brain, Zap, Database } from 'lucide-react'
import { PulseLoader } from '@/components/loaders/LoadingAnimations'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  chart?: Record<string, unknown> | null
}

const SUGGESTED = [
  { icon: Database,  text: 'What is this dataset about?' },
  { icon: Brain,     text: 'What features are most important?' },
  { icon: BarChart2, text: 'Show distribution of target column' },
  { icon: Sparkles,  text: 'Show correlation heatmap' },
  { icon: Zap,       text: 'Are there any missing values?' },
]

// ── Floating 3-D orb ──────────────────────────────────────────────────────────
function Orb({ x, y, size, color, delay }: { x: string; y: string; size: number; color: string; delay: number }) {
  return (
    <motion.div
      data-orb
      className="absolute rounded-full pointer-events-none"
      style={{
        left: x, top: y, width: size, height: size,
        background: `radial-gradient(circle at 35% 35%, ${color}55, ${color}11 60%, transparent)`,
        boxShadow: `0 0 ${size * 0.6}px ${color}33, inset 0 0 ${size * 0.3}px ${color}22`,
        filter: 'blur(1px)',
      }}
      animate={{ y: [0, -24, 0], x: [0, 12, 0], scale: [1, 1.08, 1] }}
      transition={{ duration: 6 + delay, repeat: Infinity, ease: 'easeInOut', delay }}
    />
  )
}

// ── Neural particle canvas ────────────────────────────────────────────────────
function NeuralField() {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current; if (!c) return
    const ctx = c.getContext('2d')!
    const resize = () => { c.width = c.offsetWidth; c.height = c.offsetHeight }
    resize(); window.addEventListener('resize', resize)
    const pts = Array.from({ length: 40 }, () => ({
      x: Math.random() * c.width, y: Math.random() * c.height,
      vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
    }))
    let raf: number
    const draw = () => {
      ctx.clearRect(0, 0, c.width, c.height)
      pts.forEach((p) => {
        p.x += p.vx; p.y += p.vy
        if (p.x < 0 || p.x > c.width) p.vx *= -1
        if (p.y < 0 || p.y > c.height) p.vy *= -1
        ctx.beginPath(); ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(139,92,246,0.35)'; ctx.fill()
      })
      for (let i = 0; i < pts.length; i++)
        for (let j = i + 1; j < pts.length; j++) {
          const d = Math.hypot(pts[j].x - pts[i].x, pts[j].y - pts[i].y)
          if (d < 120) {
            ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y)
            ctx.strokeStyle = `rgba(139,92,246,${0.12 * (1 - d / 120)})`; ctx.lineWidth = 0.6; ctx.stroke()
          }
        }
      raf = requestAnimationFrame(draw)
    }
    draw()
    return () => { window.removeEventListener('resize', resize); cancelAnimationFrame(raf) }
  }, [])
  return <canvas ref={ref} className="absolute inset-0 w-full h-full pointer-events-none" />
}

// ── AI avatar with pulsing rings ──────────────────────────────────────────────
function AIAvatar({ thinking }: { thinking: boolean }) {
  return (
    <div className="relative w-8 h-8 shrink-0 mt-1">
      {thinking && [0, 1].map((i) => (
        <motion.div key={i} className="absolute inset-0 rounded-full border border-violet-400/50"
          animate={{ scale: [1, 1.8 + i * 0.4], opacity: [0.6, 0] }}
          transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.4, ease: 'easeOut' }}
        />
      ))}
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center"
        style={{ boxShadow: thinking ? '0 0 16px rgba(139,92,246,0.8)' : '0 0 8px rgba(139,92,246,0.4)' }}>
        <Brain size={14} className="text-white" />
      </div>
    </div>
  )
}

// ── 3-D tilt card wrapper ─────────────────────────────────────────────────────
function TiltCard({ children, className }: { children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const rx = useMotionValue(0); const ry = useMotionValue(0)
  const rotX = useTransform(rx, [-1, 1], [4, -4])
  const rotY = useTransform(ry, [-1, 1], [-6, 6])

  const onMove = (e: React.MouseEvent) => {
    const el = ref.current; if (!el) return
    const { left, top, width, height } = el.getBoundingClientRect()
    rx.set(((e.clientY - top) / height - 0.5) * 2)
    ry.set(((e.clientX - left) / width - 0.5) * 2)
  }
  const onLeave = () => { rx.set(0); ry.set(0) }

  return (
    <motion.div ref={ref} onMouseMove={onMove} onMouseLeave={onLeave}
      style={{ rotateX: rotX, rotateY: rotY, transformStyle: 'preserve-3d', perspective: 800 }}
      className={className}>
      {children}
    </motion.div>
  )
}

// Decode HTML entities that occasionally leak from the LLM response
function decodeEntities(str: string): string {
  if (typeof document === 'undefined') return str
  const txt = document.createElement('textarea')
  txt.innerHTML = str
  return txt.value
}

// Safe Plotly wrapper — catches render errors so a bad chart never crashes the chat
function SafePlot({ chart }: { chart: Record<string, unknown> }) {
  const [error, setError] = useState<string | null>(null)
  if (error) {
    return (
      <div className="px-4 py-6 text-center text-sm text-gray-500">
        Chart could not be rendered. The data may be in an unexpected format.
      </div>
    )
  }
  const data   = (chart.data   as any[]) ?? []
  const layout = (chart.layout as Record<string, unknown>) ?? {}
  if (!data.length) return null
  return (
    <Plot
      data={data}
      layout={{
        ...layout,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  'rgba(0,0,0,0)',
        font: { color: '#cbd5e1', size: 11, family: 'Inter, sans-serif' },
        margin: { t: 32, r: 16, b: 48, l: 52 },
        autosize: true,
        xaxis: { ...(layout.xaxis as any), gridcolor: 'rgba(139,92,246,0.1)', zerolinecolor: 'rgba(139,92,246,0.2)' },
        yaxis: { ...(layout.yaxis as any), gridcolor: 'rgba(139,92,246,0.1)', zerolinecolor: 'rgba(139,92,246,0.2)' },
      }}
      config={{ displayModeBar: true, responsive: true, displaylogo: false }}
      style={{ width: '100%', minHeight: 340 }}
      useResizeHandler
      onError={() => setError('render error')}
    />
  )
}

export default function ChatPage() {
  const { dataset: datasetInfo, datasetId, mounted } = useStoredDataset()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isThinking, setIsThinking] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatMutation = useChat()

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, isThinking])

  const send = async (question: string) => {
    if (!question.trim() || !datasetId || isThinking) return
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'user', content: question }])
    setInput('')
    setIsThinking(true)
    try {
      const res = await chatMutation.mutateAsync({ datasetId, question })
      setMessages((prev) => [...prev, {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: decodeEntities(res.answer ?? ''),
        chart: res.chart ?? null,
      }])
    } catch {
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'assistant', content: 'Something went wrong. Please try again.', chart: null }])
    } finally {
      setIsThinking(false)
    }
  }

  if (!mounted) return null

  if (!datasetId) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-black">
        <div className="text-center">
          <Brain size={40} className="text-violet-400 mx-auto mb-4" />
          <p className="text-gray-400">Please upload a dataset first</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative min-h-screen bg-black flex flex-col overflow-hidden">

      {/* ── 3D background layer ── */}
      <div className="absolute inset-0 pointer-events-none">
        <NeuralField />
        <Orb x="5%"  y="10%" size={180} color="#8b5cf6" delay={0} />
        <Orb x="75%" y="5%"  size={140} color="#06b6d4" delay={1.5} />
        <Orb x="60%" y="65%" size={200} color="#7c3aed" delay={2.5} />
        <Orb x="15%" y="70%" size={120} color="#0891b2" delay={0.8} />
        {/* Grid lines */}
        <div className="absolute inset-0"
          style={{ backgroundImage: 'linear-gradient(rgba(139,92,246,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,0.04) 1px, transparent 1px)', backgroundSize: '48px 48px' }} />
      </div>

      {/* ── Header ── */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 shrink-0 px-6 py-4 border-b border-violet-500/20"
        style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(20px)' }}
      >
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Animated logo */}
            <div className="relative w-10 h-10">
              <motion.div className="absolute inset-0 rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500"
                animate={{ rotate: [0, 360] }} transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
                style={{ clipPath: 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)' }}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <Brain size={18} className="text-white" />
              </div>
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">Dataset Assistant</h1>
              {datasetInfo && <p className="text-xs text-violet-300">{datasetInfo.filename}</p>}
            </div>
          </div>

          {/* Live indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-violet-500/30 bg-violet-500/10">
            <motion.div className="w-2 h-2 rounded-full bg-emerald-400"
              animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 1.5, repeat: Infinity }} />
            <span className="text-xs text-gray-300">AI Online</span>
          </div>
        </div>
      </motion.div>

      {/* ── Messages ── */}
      <div className="relative z-10 flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          <AnimatePresence>

            {/* Empty state */}
            {messages.length === 0 && (
              <motion.div key="empty" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center pt-12 pb-6">

                {/* 3D rotating icon */}
                <div className="relative w-24 h-24 mb-6">
                  {[0, 1, 2].map((i) => (
                    <motion.div key={i} className="absolute inset-0 rounded-full border border-violet-500/30"
                      animate={{ scale: [1, 1.4 + i * 0.3], opacity: [0.5, 0] }}
                      transition={{ duration: 2, repeat: Infinity, delay: i * 0.5, ease: 'easeOut' }}
                    />
                  ))}
                  <motion.div className="absolute inset-0 rounded-full bg-gradient-to-br from-violet-600/30 to-cyan-600/20 border border-violet-500/40 flex items-center justify-center"
                    animate={{ rotateY: [0, 360] }} transition={{ duration: 6, repeat: Infinity, ease: 'linear' }}
                    style={{ transformStyle: 'preserve-3d' }}>
                    <Sparkles size={36} className="text-violet-300" />
                  </motion.div>
                </div>

                <h2 className="text-2xl font-bold text-white mb-2">Start a Conversation</h2>
                <p className="text-gray-400 text-center mb-8 max-w-md">
                  Ask me anything about your dataset. I can analyse, visualise, and explain your data.
                </p>

                {/* Suggested chips */}
                <div className="flex flex-wrap gap-2 justify-center max-w-2xl">
                  {SUGGESTED.map(({ icon: Icon, text }, i) => (
                    <motion.button key={text}
                      initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 + i * 0.08 }}
                      whileHover={{ scale: 1.04, y: -2 }} whileTap={{ scale: 0.97 }}
                      onClick={() => send(text)}
                      className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm text-gray-200 border border-violet-500/25 transition-all"
                      style={{ background: 'rgba(139,92,246,0.08)', backdropFilter: 'blur(12px)', boxShadow: '0 4px 16px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06)' }}
                    >
                      <Icon size={14} className="text-violet-400 shrink-0" />
                      {text}
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Messages */}
            {messages.map((msg, idx) => (
              <motion.div key={msg.id}
                initial={{ opacity: 0, y: 16, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.35, ease: [0.34, 1.2, 0.64, 1] }}
                className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
              >
                {/* Avatar */}
                {msg.role === 'assistant'
                  ? <AIAvatar thinking={false} />
                  : (
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-pink-500 to-violet-600 flex items-center justify-center shrink-0 mt-1"
                      style={{ boxShadow: '0 0 12px rgba(236,72,153,0.4)' }}>
                      <span className="text-xs font-bold text-white">U</span>
                    </div>
                  )
                }

                <div className={`flex flex-col gap-2 max-w-xl ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                  {/* Bubble */}
                  <TiltCard>
                    <div
                      className="px-5 py-3.5 rounded-2xl text-sm text-white whitespace-pre-wrap leading-relaxed"
                      style={msg.role === 'user' ? {
                        background: 'linear-gradient(135deg, rgba(124,58,237,0.85), rgba(6,182,212,0.6))',
                        boxShadow: '0 8px 32px rgba(124,58,237,0.35), 0 2px 8px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.15)',
                        border: '1px solid rgba(139,92,246,0.5)',
                        backdropFilter: 'blur(16px)',
                      } : {
                        background: 'rgba(15,15,30,0.75)',
                        boxShadow: '0 8px 32px rgba(0,0,0,0.5), 0 2px 8px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06)',
                        border: '1px solid rgba(139,92,246,0.2)',
                        backdropFilter: 'blur(20px)',
                      }}
                    >
                      {decodeEntities(msg.content)}
                    </div>
                  </TiltCard>

                  {/* Chart panel */}
                  {msg.chart && (
                    <motion.div
                      initial={{ opacity: 0, y: 12, scale: 0.96 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      transition={{ delay: 0.2, duration: 0.4 }}
                      className="w-full rounded-2xl overflow-hidden"
                      style={{
                        background: 'rgba(8,8,20,0.85)',
                        border: '1px solid rgba(139,92,246,0.25)',
                        boxShadow: '0 16px 48px rgba(0,0,0,0.6), 0 0 0 1px rgba(139,92,246,0.1), inset 0 1px 0 rgba(255,255,255,0.05)',
                        backdropFilter: 'blur(24px)',
                      }}
                    >
                      {/* Chart header bar */}
                      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-violet-500/15">
                        <div className="flex gap-1.5">
                          {['#ef4444','#f59e0b','#10b981'].map((c) => (
                            <div key={c} className="w-2.5 h-2.5 rounded-full" style={{ background: c }} />
                          ))}
                        </div>
                        <span className="text-xs text-gray-500 ml-1">Visualisation</span>
                        <motion.div className="ml-auto w-1.5 h-1.5 rounded-full bg-emerald-400"
                          animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 2, repeat: Infinity }} />
                      </div>
                      <SafePlot chart={msg.chart as Record<string, unknown>} />
                    </motion.div>
                  )}
                </div>
              </motion.div>
            ))}

            {/* Thinking */}
            {isThinking && (
              <motion.div key="thinking" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                className="flex gap-3 items-start">
                <AIAvatar thinking />
                <div className="px-5 py-4 rounded-2xl flex items-center gap-3"
                  style={{
                    background: 'rgba(15,15,30,0.75)',
                    border: '1px solid rgba(139,92,246,0.25)',
                    backdropFilter: 'blur(20px)',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                  }}>
                  <PulseLoader />
                  <span className="text-sm text-gray-400">Analysing your data</span>
                  <motion.div className="flex gap-0.5">
                    {[0,1,2].map((i) => (
                      <motion.span key={i} className="text-violet-400 text-base"
                        animate={{ opacity: [0,1,0], y: [0,-4,0] }}
                        transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.2 }}>.</motion.span>
                    ))}
                  </motion.div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* ── Input bar ── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 shrink-0 px-4 py-4 border-t border-violet-500/20"
        style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(24px)' }}
      >
        <div className="max-w-4xl mx-auto flex gap-3 items-center">
          <div className="flex-1 relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }}
              placeholder="Ask anything about your dataset..."
              className="w-full py-3.5 pl-5 pr-12 rounded-2xl text-sm text-white placeholder-gray-500 outline-none transition-all"
              style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(139,92,246,0.3)',
                backdropFilter: 'blur(16px)',
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05)',
              }}
              onFocus={(e) => { e.currentTarget.style.borderColor = 'rgba(139,92,246,0.7)'; e.currentTarget.style.boxShadow = '0 0 20px rgba(139,92,246,0.2), inset 0 1px 0 rgba(255,255,255,0.05)' }}
              onBlur={(e) => { e.currentTarget.style.borderColor = 'rgba(139,92,246,0.3)'; e.currentTarget.style.boxShadow = 'inset 0 1px 0 rgba(255,255,255,0.05)' }}
            />
          </div>

          <motion.button
            whileHover={{ scale: 1.08 }} whileTap={{ scale: 0.93 }}
            onClick={() => send(input)}
            disabled={!input.trim() || isThinking}
            className="w-12 h-12 rounded-2xl flex items-center justify-center text-white disabled:opacity-40 shrink-0 transition-all"
            style={{
              background: 'linear-gradient(135deg, #7c3aed, #0891b2)',
              boxShadow: '0 0 20px rgba(124,58,237,0.5), 0 4px 16px rgba(0,0,0,0.4)',
            }}
          >
            <Send size={18} />
          </motion.button>
        </div>
      </motion.div>
    </div>
  )
}
