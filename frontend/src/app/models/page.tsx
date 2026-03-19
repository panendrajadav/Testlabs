'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { usePipelineStatus } from '@/hooks/useApi'
import { containerVariants, itemVariants } from '@/animations/variants'
import { ShimmerLoader } from '@/components/loaders/LoadingAnimations'
import { ModelMetricsChart, ROCCurveChart, FeatureImportanceChart } from '@/components/charts/ChartComponents'
import { AlertTriangle, CheckCircle, Info } from 'lucide-react'
import type { PipelineStatus } from '@/types'

const OVERFIT_WARN  = 0.05   // gap > 5%  → warning
const OVERFIT_ALERT = 0.15   // gap > 15% → alert

function OverfitBadge({ gap }: { gap: number }) {
  if (gap <= OVERFIT_WARN) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-green-500/20 text-green-400 border border-green-500/30">
        <CheckCircle size={12} /> Good generalisation
      </span>
    )
  }
  if (gap <= OVERFIT_ALERT) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
        <AlertTriangle size={12} /> Mild overfitting (gap {(gap * 100).toFixed(1)}%)
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-red-500/20 text-red-400 border border-red-500/30">
      <AlertTriangle size={12} /> Severe overfitting (gap {(gap * 100).toFixed(1)}%)
    </span>
  )
}

export default function ModelsPage() {
  const [mounted, setMounted] = useState(false)
  const [datasetId, setDatasetId] = useState<string | null>(null)

  useEffect(() => {
    const stored = localStorage.getItem('currentDataset')
    if (stored) setDatasetId(JSON.parse(stored).dataset_id)
    setMounted(true)
  }, [])

  const { data: job, isLoading } = usePipelineStatus(datasetId)
  const status = job as PipelineStatus | undefined
  const result = status?.result

  if (!mounted) return null

  if (!datasetId) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-400">Please upload a dataset first</p>
      </div>
    )
  }

  if (isLoading || status?.status === 'running' || status?.status === 'queued') {
    return (
      <div className="min-h-screen bg-black p-8">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="bg-slate-900/50 rounded-xl p-6"><ShimmerLoader count={3} /></div>
          ))}
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-400">No results yet. Run the pipeline first.</p>
      </div>
    )
  }

  const isClassification = result.task_type === 'classification'
  const evalResults = result.evaluation_results ?? []
  const bestResult = evalResults.find((r) => (r.model_name ?? r.model) === result.best_model)
  const metrics = bestResult?.metrics ?? {}
  const overfitGap = (metrics.overfit_gap as number) ?? 0

  // Primary metric cards
  const metricCards = isClassification
    ? [
        { label: 'CV Accuracy',    value: metrics.accuracy,      hint: 'Cross-validated' },
        { label: 'Test Accuracy',  value: metrics.test_accuracy,  hint: 'Held-out 20%' },
        { label: 'Train Accuracy', value: metrics.train_accuracy, hint: 'Training set' },
        { label: 'F1 Score',       value: metrics.f1_score,       hint: 'Weighted, test set' },
        { label: 'ROC-AUC',        value: metrics.roc_auc,        hint: 'Test set' },
        { label: 'CV Std Dev',     value: metrics.cv_std,         hint: 'Lower = more stable' },
      ]
    : [
        { label: 'CV R²',       value: metrics.r2_score, hint: 'Cross-validated' },
        { label: 'Test R²',     value: metrics.test_r2,  hint: 'Held-out 20%' },
        { label: 'Train R²',    value: metrics.train_r2, hint: 'Training set' },
        { label: 'RMSE',        value: metrics.rmse,     hint: 'Test set' },
        { label: 'MAE',         value: metrics.mae,      hint: 'Test set' },
        { label: 'CV Std Dev',  value: metrics.cv_std,   hint: 'Lower = more stable' },
      ]

  const rocPoints = (() => {
    const roc = result.roc_data
    if (!roc) return []
    if (roc.fpr && roc.tpr) return roc.fpr.map((fpr, i) => ({ fpr, tpr: roc.tpr![i] }))
    // multiclass: use first class curve
    if (roc.curves) {
      const firstKey = Object.keys(roc.curves)[0]
      if (firstKey) {
        const c = roc.curves[firstKey]
        return c.fpr.map((fpr, i) => ({ fpr, tpr: c.tpr[i] }))
      }
    }
    return []
  })()

  const shapData = result.shap_values
    ? result.shap_values.feature_names.map((f, i) => ({ feature: f, importance: result.shap_values!.mean_abs_shap[i] }))
    : []

  return (
    <motion.div initial="hidden" animate="visible" variants={containerVariants} className="min-h-screen bg-black p-8">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <motion.div variants={itemVariants} className="mb-8">
          <h1 className="text-4xl font-bold text-gradient mb-1">Model Results</h1>
          <div className="flex flex-wrap items-center gap-3 mt-2">
            <p className="text-gray-400">
              Best model: <span className="text-white font-semibold">{result.best_model}</span>
              {' '}— CV Score: <span className="text-cyan-400 font-semibold">{result.best_score != null ? result.best_score.toFixed(4) : 'N/A'}</span>
            </p>
            <OverfitBadge gap={overfitGap} />
          </div>
        </motion.div>

        {/* Overfitting warning banner */}
        {overfitGap > OVERFIT_WARN && (
          <motion.div variants={itemVariants}
            className={`rounded-xl border p-4 mb-8 glass flex gap-3 ${
              overfitGap > OVERFIT_ALERT
                ? 'border-red-500/40 bg-red-500/10'
                : 'border-yellow-500/40 bg-yellow-500/10'
            }`}>
            <AlertTriangle className={overfitGap > OVERFIT_ALERT ? 'text-red-400 shrink-0 mt-0.5' : 'text-yellow-400 shrink-0 mt-0.5'} size={18} />
            <div>
              <p className={`font-semibold text-sm ${overfitGap > OVERFIT_ALERT ? 'text-red-300' : 'text-yellow-300'}`}>
                {overfitGap > OVERFIT_ALERT ? 'Severe Overfitting Detected' : 'Mild Overfitting Detected'}
              </p>
              <p className="text-gray-400 text-sm mt-1">
                The model scores <span className="text-white font-medium">{(metrics.train_accuracy ?? metrics.train_r2 ?? 0).toFixed(4)}</span> on
                training data but only <span className="text-white font-medium">{(metrics.accuracy ?? metrics.r2_score ?? 0).toFixed(4)}</span> on
                cross-validation — a gap of <span className="text-white font-medium">{(overfitGap * 100).toFixed(1)}%</span>.
                The model has memorised training patterns and may generalise poorly to new data.
                {overfitGap > OVERFIT_ALERT && ' Consider collecting more data, reducing model complexity, or adding regularisation.'}
              </p>
            </div>
          </motion.div>
        )}

        {/* Metric cards */}
        <motion.div variants={containerVariants} className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-10">
          {metricCards.map((m, i) => (
            <motion.div key={m.label} variants={itemVariants} whileHover={{ y: -4 }}
              className="bg-slate-900/50 rounded-xl border border-purple-500/20 glass p-4">
              <p className="text-gray-400 text-xs mb-1">{m.label}</p>
              <p className="text-2xl font-bold text-purple-400">
                {m.value != null ? (m.value as number).toFixed(4) : 'N/A'}
              </p>
              <p className="text-gray-600 text-xs mt-1">{m.hint}</p>
            </motion.div>
          ))}
        </motion.div>

        {/* Train vs CV comparison bar */}
        <motion.div variants={itemVariants} className="bg-slate-900/50 rounded-xl border border-purple-500/20 glass p-6 mb-8">
          <div className="flex items-center gap-2 mb-4">
            <Info size={16} className="text-purple-400" />
            <h3 className="text-white font-semibold">Train vs CV Score Comparison</h3>
          </div>
          <div className="space-y-3">
            {[
              { label: 'Train Score', value: (metrics.train_accuracy ?? metrics.train_r2 ?? 0) as number, color: 'bg-orange-500' },
              { label: 'CV Score',    value: (metrics.accuracy ?? metrics.r2_score ?? 0) as number,        color: 'bg-purple-500' },
              { label: 'Test Score',  value: (metrics.test_accuracy ?? metrics.test_r2 ?? 0) as number,    color: 'bg-cyan-500' },
            ].map((bar) => (
              <div key={bar.label} className="flex items-center gap-3">
                <span className="text-gray-400 text-sm w-24 shrink-0">{bar.label}</span>
                <div className="flex-1 bg-gray-800 rounded-full h-3 overflow-hidden">
                  <motion.div
                    className={`h-full ${bar.color} rounded-full`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(0, Math.min(100, (bar.value ?? 0) * 100))}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                  />
                </div>
                <span className="text-white text-sm font-medium w-16 text-right">{(bar.value ?? 0).toFixed(4)}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Charts */}
        <motion.div variants={containerVariants} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <motion.div variants={itemVariants}>
            <ModelMetricsChart data={evalResults} />
          </motion.div>
          {isClassification && rocPoints.length > 0 && (
            <motion.div variants={itemVariants}>
              <ROCCurveChart data={rocPoints} />
            </motion.div>
          )}
          {shapData.length > 0 && (
            <motion.div variants={itemVariants}>
              <FeatureImportanceChart data={shapData} />
            </motion.div>
          )}
        </motion.div>

      </div>
    </motion.div>
  )
}
