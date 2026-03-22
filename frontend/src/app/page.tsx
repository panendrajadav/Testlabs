'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { ArrowRight, Zap, TrendingUp, Database } from 'lucide-react'
import NeuralNetworkBackground from '@/components/visualization/NeuralNetworkBackground'
import Pipeline3DVisualizer from '@/components/visualization/Pipeline3DVisualizer'
import { containerVariants, itemVariants } from '@/animations/variants'
import { PIPELINE_STAGES } from '@/utils/constants'
import { getAuth } from '@/hooks/useAuth'

export default function DashboardPage() {
  const router = useRouter()
  const [activeIdx, setActiveIdx] = useState(0)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    let idx = 0
    let prog = 0
    let animating = true

    const HOLD_MS = 200   // pause at 100% before moving on
    const TICK_MS = 40     // interval speed
    const STEP = 2         // progress increment per tick (~2s to fill)

    const tick = setInterval(() => {
      if (!animating) return
      prog += STEP
      if (prog <= 100) {
        setProgress(prog)
      } else {
        // Hold at 100% for HOLD_MS then advance
        animating = false
        setTimeout(() => {
          idx = (idx + 1) % PIPELINE_STAGES.length
          prog = 0
          setActiveIdx(idx)
          setProgress(0)
          animating = true
        }, HOLD_MS)
      }
    }, TICK_MS)

    return () => clearInterval(tick)
  }, [])

  const mockPipelineStages = PIPELINE_STAGES.map((stage, i) => ({
    id: stage.id,
    name: stage.label,
    status: (
      i < activeIdx ? 'completed' :
      i === activeIdx ? 'running' : 'pending'
    ) as 'completed' | 'running' | 'pending' | 'failed',
    progress: i === activeIdx ? progress : 0,
  }))

  return (
    <div className="relative min-h-screen bg-black overflow-hidden">
      {/* Neural Network Background */}
      <NeuralNetworkBackground />

      {/* Decorative floating orbs — repelled by cursor */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {[
          { left: '8%',  top: '15%', w: 220, color: '#8b5cf6', delay: 0 },
          { left: '78%', top: '8%',  w: 160, color: '#06b6d4', delay: 1.5 },
          { left: '65%', top: '60%', w: 240, color: '#7c3aed', delay: 2.5 },
          { left: '12%', top: '68%', w: 140, color: '#0891b2', delay: 0.8 },
          { left: '45%', top: '80%', w: 100, color: '#a855f7', delay: 3.2 },
          { left: '88%', top: '45%', w: 130, color: '#6366f1', delay: 1.9 },
        ].map((orb, i) => (
          <motion.div
            key={i}
            data-orb
            className="absolute rounded-full"
            style={{
              left: orb.left, top: orb.top,
              width: orb.w, height: orb.w,
              background: `radial-gradient(circle at 35% 35%, ${orb.color}44, ${orb.color}11 60%, transparent)`,
              boxShadow: `0 0 ${orb.w * 0.5}px ${orb.color}22`,
              filter: 'blur(2px)',
            }}
            animate={{ y: [0, -30, 0], x: [0, 15, 0], scale: [1, 1.1, 1] }}
            transition={{ duration: 7 + orb.delay, repeat: Infinity, ease: 'easeInOut', delay: orb.delay }}
          />
        ))}
      </div>

      {/* Content */}
      <div className="relative z-10">
        {/* Hero Section */}
        <motion.section
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="min-h-screen flex items-center justify-center px-4 py-20"
        >
          <div className="max-w-4xl mx-auto text-center">
            {/* Title */}
            <motion.h1
              variants={itemVariants}
              className="text-5xl md:text-7xl font-bold mb-6 leading-tight"
            >
              <span className="text-gradient">
                Automate the ML Lifecycle
              </span>
              <br />
              <span className="text-3xl md:text-5xl text-gray-300">
                with Intelligent Agents
              </span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              variants={itemVariants}
              className="text-lg md:text-xl text-gray-400 mb-12 leading-relaxed"
            >
              Upload a dataset and let AI agents build, train, and evaluate machine learning
              models automatically. Experience the future of AutoML with real-time monitoring and
              intelligent optimization.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              variants={itemVariants}
              className="flex flex-col sm:flex-row gap-4 justify-center mb-20"
            >
              <motion.button
                  whileHover={{ scale: 1.05, boxShadow: '0 0 50px rgba(139, 92, 246, 0.6)' }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => router.push(getAuth() ? '/upload' : '/login?redirect=/upload')}
                  className="px-8 py-4 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-lg font-semibold text-white flex items-center gap-2 glow text-center justify-center"
                >
                  <Zap size={20} />
                  Create Experiment
                  <ArrowRight size={20} />
                </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-8 py-4 bg-white/10 border border-purple-500/50 rounded-lg font-semibold text-white hover:bg-white/20 transition-all"
              >
                Learn More
              </motion.button>
            </motion.div>

            {/* Feature Cards */}
            <motion.div
              variants={containerVariants}
              className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-20"
            >
              {[
                {
                  icon: Database,
                  title: 'Smart EDA',
                  description: 'Automatic exploratory data analysis',
                },
                {
                  icon: TrendingUp,
                  title: 'Auto Training',
                  description: 'Intelligent model selection & tuning',
                },
                {
                  icon: Zap,
                  title: 'Fast Results',
                  description: 'Real-time pipeline monitoring',
                },
              ].map((feature, i) => {
                const Icon = feature.icon
                return (
                  <motion.div
                    key={i}
                    variants={itemVariants}
                    whileHover={{ y: -10 }}
                    className="bg-gradient-to-br from-slate-900/50 to-slate-800/50 p-6 rounded-xl border border-purple-500/20 glass"
                  >
                    <Icon className="w-10 h-10 text-purple-400 mb-3 mx-auto" />
                    <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                    <p className="text-sm text-gray-400">{feature.description}</p>
                  </motion.div>
                )
              })}
            </motion.div>
          </div>
        </motion.section>

        {/* Pipeline Visualization Section */}
        <motion.section
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true, margin: '-100px' }}
          className="py-20 px-4"
        >
          <div className="max-w-7xl mx-auto">
            <motion.h2
              variants={itemVariants}
              className="text-4xl font-bold text-center mb-6 text-gradient"
            >
              ML Pipeline Architecture
            </motion.h2>
            <p className="text-center text-gray-400 mb-12 max-w-2xl mx-auto">
              Our intelligent system processes your data through {PIPELINE_STAGES.length} automated stages,
              delivering optimized machine learning models automatically.
            </p>

            {/* 3D Pipeline Visualizer */}
            <div className="bg-gradient-to-b from-slate-900/30 to-transparent rounded-2xl border border-purple-500/20 glass p-8 overflow-hidden">
              <Pipeline3DVisualizer stages={mockPipelineStages} />
            </div>

            {/* Stage Details */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-12">
              {PIPELINE_STAGES.map((stage, i) => (
                <motion.div
                  key={stage.id}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  viewport={{ once: true }}
                  whileHover={{ y: -5 }}
                  className="bg-gradient-to-br from-slate-900/50 to-slate-800/50 p-5 rounded-lg border border-purple-500/20 glass"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-cyan-500 rounded flex items-center justify-center text-sm font-bold text-white flex-shrink-0">
                      {i + 1}
                    </div>
                    <div>
                      <h3 className="font-semibold text-white">{stage.label}</h3>
                      <p className="text-sm text-gray-400 mt-1">Automated {stage.name}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* CTA Section */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="py-20 px-4 border-t border-purple-500/20"
        >
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-4xl font-bold mb-6 text-white">Ready to Get Started?</h2>
            <p className="text-lg text-gray-400 mb-8">
              Upload your CSV dataset and watch as our AI agents build, train, and optimize your
              models in real-time.
            </p>
            <Link href="/upload">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-12 py-4 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-lg font-semibold text-white flex items-center gap-2 glow mx-auto"
              >
              Start Experiment <ArrowRight size={20} />
              </motion.button>
            </Link>
          </div>
        </motion.section>
      </div>
    </div>
  )
}
