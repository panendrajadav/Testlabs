'use client'

import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { usePipelineStatus, usePipelineWebSocket } from '@/hooks/useApi'
import { useStoredDataset, updateExperimentResult } from '@/hooks/useStoredDataset'
import { containerVariants, itemVariants } from '@/animations/variants'
import { PIPELINE_STAGES } from '@/utils/constants'
import { CheckCircle, Clock, AlertCircle, Zap, ArrowRight, Wifi, WifiOff } from 'lucide-react'
import Pipeline3DVisualizer from '@/components/visualization/Pipeline3DVisualizer'
import { PipelineLoadingAnimation } from '@/components/loaders/LoadingAnimations'
import { apiService } from '@/services/api'
import { fsSetArtifacts, type ArtifactPayload } from '@/services/firestoreService'
import type { PipelineStatus } from '@/types'

export default function PipelinePage() {
  const router = useRouter()
  const { datasetId, mounted } = useStoredDataset()

  const wsConnected = usePipelineWebSocket(datasetId)
  const { data: job, isLoading } = usePipelineStatus(datasetId)
  const status = job as PipelineStatus | undefined
  const isRunning = status?.status === 'running' || status?.status === 'queued'
  const isDone = status?.status === 'completed'
  const isFailed = status?.status === 'failed'

  // Patch experiment list when pipeline finishes
  const artifactsSynced = useRef(false)
  useEffect(() => {
    if (isDone && datasetId && status?.result) {
      updateExperimentResult(datasetId, {
        best_model: status.result.best_model,
        best_score: status.result.best_score,
        task_type:  status.result.task_type,
        status:     'completed',
      })

      // Push full artifacts to Firestore once per completion
      if (!artifactsSynced.current) {
        artifactsSynced.current = true
        const result = status.result
        const version = result.artifact_version
        if (version) {
          apiService.getArtifactSummary(datasetId, version)
            .then((summary) => {
              const payload: ArtifactPayload = {
                version,
                dataset_id:          datasetId,
                model_name:          result.best_model          ?? '',
                task_type:           result.task_type           ?? '',
                best_score:          result.best_score          ?? null,
                best_params:         result.best_params         ?? {},
                timestamp:           summary.metadata?.timestamp       ?? new Date().toISOString(),
                training_time_s:     summary.metadata?.training_time_s ?? 0,
                target_column:       result.target_column       ?? '',
                selected_features:   result.selected_features   ?? [],
                justification:       result.justification       ?? null,
                is_underfit:         result.is_underfit         ?? false,
                evaluation_results:  (result.evaluation_results ?? []) as unknown as Record<string, unknown>[],
                metadata:            summary.metadata                  ?? null,
                experiment_log:      summary.experiment                ?? null,
                training_log:        summary.training_log              ?? null,
                reproducibility:     summary.reproducibility           ?? null,
                inference_samples:   summary.inference_samples         ?? null,
                drift_hooks:         summary.drift_hooks               ?? null,
                api_export_code:     summary.api_export_code           ?? null,
                model_file_exists:   summary.model_file_exists         ?? false,
                model_size_kb:       summary.model_size_kb             ?? null,
                agent_logs:          result.agent_logs          ?? [],
                eda_summary:         result.eda_summary         ?? null,
                shap_values:         result.shap_values         ?? null,
                roc_data:            result.roc_data            ?? null,
                preprocessing_report: result.preprocessing_report ?? null,
              }
              return fsSetArtifacts(datasetId, version, payload)
            })
            .catch(console.error)
        }
      }
    } else if (isFailed && datasetId) {
      updateExperimentResult(datasetId, { status: 'failed' })
    }
  }, [isDone, isFailed, datasetId])

  const progressIdx = (isRunning || isFailed)
    ? PIPELINE_STAGES.findIndex((p) => p.name === status?.progress)
    : -1
  // "Starting", "Queued", or any unknown string → treat as stage 0 running
  const effectiveIdx = progressIdx >= 0 ? progressIdx : (isRunning ? 0 : -1)

  const vizStages = PIPELINE_STAGES.map((s, thisIdx) => {
    let stageStatus: 'pending' | 'running' | 'completed' | 'failed' = 'pending'
    if (isDone) stageStatus = 'completed'
    else if (isFailed && thisIdx === effectiveIdx) stageStatus = 'failed'
    else if (effectiveIdx >= 0 && thisIdx < effectiveIdx) stageStatus = 'completed'
    else if (effectiveIdx >= 0 && thisIdx === effectiveIdx) stageStatus = 'running'
    return { id: s.id, name: s.label, status: stageStatus, progress: stageStatus === 'running' ? 50 : 0 }
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
          <p className="text-gray-400 flex items-center gap-2">
            Real-time monitoring of your AutoML pipeline
            {wsConnected
              ? <span className="flex items-center gap-1 text-green-400 text-xs"><Wifi size={12} /> live</span>
              : <span className="flex items-center gap-1 text-gray-500 text-xs"><WifiOff size={12} /> polling</span>
            }
          </p>
        </motion.div>

        {/* Status banner */}
        <motion.div variants={itemVariants}
          className={`border rounded-lg p-4 mb-8 glass flex items-center gap-3 ${
            isDone    ? 'border-green-500/30 bg-green-500/10' :
            isFailed  ? 'border-red-500/30 bg-red-500/10' :
            isRunning ? 'border-purple-500/30 bg-purple-500/10' :
                        'border-slate-700/50 bg-slate-900/30'
          }`}>
          {isDone   ? <CheckCircle className="text-green-400 shrink-0" size={20} /> :
           isFailed ? <AlertCircle className="text-red-400 shrink-0" size={20} /> :
           isRunning ? <motion.div animate={{ rotate: 360 }} transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}><Zap className="text-blue-400" size={20} /></motion.div> :
           <Clock className="text-gray-500 shrink-0" size={20} />}

          <div className="flex flex-col gap-0.5 min-w-0">
            <span className="text-white font-medium capitalize">
              {isDone ? 'Pipeline Complete' :
               isFailed ? 'Pipeline Failed' :
               isRunning ? 'Running Pipeline' :
               'Waiting to start — click Run Pipeline below'}
            </span>
            {isRunning && status?.progress && (
              <span className="text-gray-400 text-xs">
                Step {effectiveIdx + 1}/{PIPELINE_STAGES.length} — {PIPELINE_STAGES.find(s => s.name === status.progress)?.label ?? status.progress}
              </span>
            )}
            {isFailed && status?.error && (
              <span className="text-red-400 text-xs truncate">{status.error}</span>
            )}
          </div>

          {isDone && (
            <motion.button
              whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
              onClick={() => router.push('/models')}
              className="ml-auto flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-lg text-white text-sm font-medium glow shrink-0"
            >
              View Results <ArrowRight size={16} />
            </motion.button>
          )}
        </motion.div>

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
