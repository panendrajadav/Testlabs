'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { usePipelineStatus } from '@/hooks/useApi'
import { containerVariants, itemVariants } from '@/animations/variants'
import { PIPELINE_STAGES } from '@/utils/constants'
import { CheckCircle, Clock, AlertCircle, Zap, ArrowRight } from 'lucide-react'
import Pipeline3DVisualizer from '@/components/visualization/Pipeline3DVisualizer'
import { PipelineLoadingAnimation } from '@/components/loaders/LoadingAnimations'
import type { PipelineStatus } from '@/types'

export default function PipelinePage() {
  const router = useRouter()
  const [mounted, setMounted] = useState(false)
  const [datasetId, setDatasetId] = useState<string | null>(null)

  useEffect(() => {
    const stored = localStorage.getItem('currentDataset')
    if (stored) setDatasetId(JSON.parse(stored).dataset_id)
    setMounted(true)
  }, [])

  const { data: job, isLoading } = usePipelineStatus(datasetId)
  const status = job as PipelineStatus | undefined
  const isRunning = status?.status === 'running' || status?.status === 'queued'
  const isDone = status?.status === 'completed'
  const isFailed = status?.status === 'failed'

  const vizStages = PIPELINE_STAGES.map((s, thisIdx) => {
    const progressIdx = PIPELINE_STAGES.findIndex((p) => p.name === status?.progress)
    let stageStatus: 'pending' | 'running' | 'completed' | 'failed' = 'pending'
    if (isDone) stageStatus = 'completed'
    else if (isFailed && thisIdx === progressIdx) stageStatus = 'failed'
    else if (thisIdx < progressIdx) stageStatus = 'completed'
    else if (thisIdx === progressIdx) stageStatus = 'running'
    return { id: s.id, name: s.name, status: stageStatus, progress: stageStatus === 'running' ? 50 : 0 }
  })

  if (!mounted) return null

  if (!datasetId) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 mb-4">No dataset selected</p>
          <button
            onClick={() => router.push('/upload')}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-lg text-white font-medium"
          >
            Upload a Dataset
          </button>
        </div>
      </div>
    )
  }

  return (
    <motion.div initial="hidden" animate="visible" variants={containerVariants} className="min-h-screen bg-black p-8">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <motion.div variants={itemVariants} className="mb-8">
          <h1 className="text-4xl font-bold text-gradient mb-2">ML Pipeline</h1>
          <p className="text-gray-400">Real-time monitoring of your AutoML pipeline</p>
        </motion.div>

        {/* Status banner */}
        {status && (
          <motion.div variants={itemVariants}
            className={`border rounded-lg p-4 mb-8 glass flex items-center gap-3 ${
              isDone ? 'border-green-500/30 bg-green-500/10' :
              isFailed ? 'border-red-500/30 bg-red-500/10' :
              'border-purple-500/30 bg-purple-500/10'
            }`}>
            {isDone ? <CheckCircle className="text-green-400 shrink-0" size={20} /> :
             isFailed ? <AlertCircle className="text-red-400 shrink-0" size={20} /> :
             isRunning ? <motion.div animate={{ rotate: 360 }} transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}><Zap className="text-blue-400" size={20} /></motion.div> :
             <Clock className="text-gray-400 shrink-0" size={20} />}
            <span className="text-white font-medium capitalize">{status.status}</span>
            {status.progress && !isDone && (
              <span className="text-gray-400 text-sm">— {status.progress}</span>
            )}
            {status.error && <span className="text-red-400 text-sm ml-2">{status.error}</span>}

            {/* View Results button */}
            {isDone && (
              <motion.button
                whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                onClick={() => router.push('/models')}
                className="ml-auto flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-lg text-white text-sm font-medium glow"
              >
                View Results <ArrowRight size={16} />
              </motion.button>
            )}
          </motion.div>
        )}

        {/* Pipeline Visualizer */}
        <motion.div variants={itemVariants}
          className="bg-gradient-to-b from-slate-900/30 to-transparent rounded-2xl border border-purple-500/20 glass p-8 mb-8 min-h-64">
          {isLoading
            ? <PipelineLoadingAnimation />
            : <Pipeline3DVisualizer stages={vizStages} />
          }
        </motion.div>

        {/* Result summary cards */}
        {status?.result && (
          <motion.div variants={containerVariants} className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { label: 'Best Model', value: status.result.best_model ?? '—', color: 'text-purple-400' },
              { label: 'Best Score', value: status.result.best_score != null ? status.result.best_score.toFixed(4) : '—', color: 'text-cyan-400' },
              { label: 'Task Type', value: status.result.task_type ?? '—', color: 'text-green-400' },
            ].map((item) => (
              <motion.div key={item.label} variants={itemVariants}
                className="bg-slate-900/50 rounded-xl border border-purple-500/20 glass p-6">
                <p className="text-gray-400 text-sm mb-1">{item.label}</p>
                <p className={`font-bold text-xl capitalize ${item.color}`}>{item.value}</p>
              </motion.div>
            ))}
          </motion.div>
        )}

      </div>
    </motion.div>
  )
}
