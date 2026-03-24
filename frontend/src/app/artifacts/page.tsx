'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { useStoredDataset } from '@/hooks/useStoredDataset'
import { containerVariants, itemVariants } from '@/animations/variants'
import { apiService } from '@/services/api'
import { getLocalArtifact, getLocalArtifactVersions, deleteLocalArtifactVersion, type LocalArtifactPayload } from '@/hooks/useStoredDataset'
import {
  Package, Download, Trash2, RotateCcw, ChevronDown, ChevronUp,
  FileCode, BarChart3, Clock, Cpu, CheckCircle, AlertCircle,
  Copy, Check, Database, Layers
} from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────────
interface VersionMeta {
  version: string
  model_name: string
  task_type: string
  best_score: number | null
  timestamp: string
  training_time_s: number
  framework: string
  format: string
  model_size_kb?: number | null
}

interface TrainingRow {
  model_name: string
  composite_score: string
  cv_score: string
  test_score: string
  train_score: string
  rmse?: string
  mae?: string
  f1_score?: string
  cv_std: string
  overfit_gap: string
  best_params: string
}

interface Summary {
  metadata: VersionMeta
  experiment: any
  training_log: TrainingRow[]
  reproducibility: any
  inference_samples: any[] | null
  api_export_code: string | null
  drift_hooks: any
  model_file_exists: boolean
  model_size_kb: number | null
}

// Proper display names for known model identifiers
const MODEL_NAMES: Record<string, string> = {
  xgboost: 'XGBoost', xgb: 'XGBoost',
  random_forest: 'Random Forest', randomforest: 'Random Forest',
  gradient_boosting: 'Gradient Boosting', gradientboosting: 'Gradient Boosting',
  logistic_regression: 'Logistic Regression', logisticregression: 'Logistic Regression',
  linear_regression: 'Linear Regression', linearregression: 'Linear Regression',
  ridge: 'Ridge Regression', lasso: 'Lasso Regression',
  elasticnet: 'ElasticNet', svm: 'SVM', svr: 'SVR', svc: 'SVC',
  knn: 'K-Nearest Neighbors', decision_tree: 'Decision Tree', decisiontree: 'Decision Tree',
  adaboost: 'AdaBoost', lightgbm: 'LightGBM', lgbm: 'LightGBM',
  catboost: 'CatBoost', mlp: 'MLP', naive_bayes: 'Naive Bayes',
  extra_trees: 'Extra Trees', extratrees: 'Extra Trees',
}

function fmtModel(name: string | null | undefined): string {
  if (!name) return '—'
  const key = name.toLowerCase().replace(/[-\s]/g, '_')
  return MODEL_NAMES[key] ?? name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

// ── Code block with copy button ───────────────────────────────────────────────
function CodeBlock({ code, language = 'python' }: { code: string; language?: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <div className="relative rounded-xl overflow-hidden border border-slate-700">
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800/80 border-b border-slate-700">
        <div className="flex gap-1.5">
          {['#ef4444', '#f59e0b', '#10b981'].map(c => (
            <div key={c} className="w-2.5 h-2.5 rounded-full" style={{ background: c }} />
          ))}
        </div>
        <span className="text-xs text-gray-500 font-mono">{language}</span>
        <button onClick={copy} className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors">
          {copied ? <><Check size={12} className="text-green-400" /> Copied</> : <><Copy size={12} /> Copy</>}
        </button>
      </div>
      <pre className="p-4 text-xs text-gray-300 font-mono overflow-x-auto bg-black/60 max-h-80 overflow-y-auto leading-relaxed">
        {code}
      </pre>
    </div>
  )
}

// ── Metric pill ───────────────────────────────────────────────────────────────
function Pill({ label, value, color = 'text-cyan-400' }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col items-center px-3 py-2 rounded-lg bg-slate-800/60 border border-slate-700 min-w-[80px]">
      <span className={`text-sm font-bold font-mono ${color}`}>{value || '—'}</span>
      <span className="text-xs text-gray-500 mt-0.5">{label}</span>
    </div>
  )
}

// ── Single version card ───────────────────────────────────────────────────────
function VersionCard({
  meta, isActive, isLatest, datasetId,
  getLocalSummary,
  onRollback, onDelete,
}: {
  meta: VersionMeta
  isActive: boolean
  isLatest: boolean
  datasetId: string
  getLocalSummary: (version: string) => Summary | null
  onRollback: (v: string) => void
  onDelete: (v: string) => void
}) {
  const [open, setOpen] = useState(isLatest)
  const [tab, setTab] = useState<'logs' | 'code' | 'samples' | 'drift'>('logs')
  const [summary, setSummary] = useState<Summary | null>(null)
  const [loading, setLoading] = useState(false)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    if (!open) return
    const local = getLocalSummary(meta.version)
    if (local) {
      // Have full local data — use it, just refresh model_file_exists from API
      setSummary(local)
      apiService.getArtifactSummary(datasetId, meta.version)
        .then(apiSummary => setSummary(s => s ? {
          ...s,
          model_file_exists: apiSummary.model_file_exists,
          model_size_kb:     apiSummary.model_size_kb,
        } : s))
        .catch(() => {})
      return
    }
    // No local data — fetch fully from API
    setLoading(true)
    apiService.getArtifactSummary(datasetId, meta.version)
      .then(setSummary)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [open, datasetId, meta.version])

  const handleDownload = () => {
    const url = apiService.getModelDownloadUrl(datasetId, meta.version)
    const a = document.createElement('a')
    a.href = url
    a.download = `${meta.model_name}_${meta.version}.pkl`
    a.click()
  }

  const handleDownloadCode = () => {
    if (!summary?.api_export_code) return
    const blob = new Blob([summary.api_export_code], { type: 'text/plain' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `api_export_${meta.model_name}_${meta.version}.py`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const handleDelete = async () => {
    if (!confirm(`Delete version ${meta.version}? This cannot be undone.`)) return
    setDeleting(true)
    await onDelete(meta.version)
  }

  const ts = meta.timestamp ? new Date(meta.timestamp).toLocaleString() : '—'
  const score = meta.best_score != null ? meta.best_score.toFixed(4) : '—'
  const isClf = meta.task_type === 'classification'

  return (
    <motion.div
      variants={itemVariants}
      className={`rounded-xl border overflow-hidden transition-colors ${
        isActive ? 'border-cyan-500/50' : 'border-slate-700 hover:border-purple-500/40'
      }`}
      style={{ background: 'rgba(15,15,30,0.7)' }}
    >
      {/* Header row */}
      <div
        className="flex items-center gap-4 p-5 cursor-pointer select-none"
        onClick={() => setOpen(o => !o)}
      >
        {/* Version badge */}
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-cyan-500 flex items-center justify-center shrink-0">
          <span className="text-white font-bold text-sm">{meta.version}</span>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-white font-semibold">{fmtModel(meta.model_name)}</span>
            {isActive && (
              <span className="px-2 py-0.5 rounded-full text-xs bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                Active
              </span>
            )}
            {isLatest && !isActive && (
              <span className="px-2 py-0.5 rounded-full text-xs bg-purple-500/20 text-purple-400 border border-purple-500/30">
                Latest
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 mt-1 text-xs text-gray-500 flex-wrap">
            <span className="flex items-center gap-1"><Clock size={10} /> {ts}</span>
            <span className="flex items-center gap-1"><Cpu size={10} /> {meta.training_time_s?.toFixed(1)}s</span>
            <span className="flex items-center gap-1"><Database size={10} /> {meta.framework}</span>
            {meta.model_size_kb != null && (
              <span className="flex items-center gap-1"><Package size={10} /> {meta.model_size_kb} KB</span>
            )}
          </div>
        </div>

        {/* Score */}
        <div className="text-right shrink-0">
          <div className="text-lg font-bold text-purple-400 font-mono">{score}</div>
          <div className="text-xs text-gray-500">best score</div>
        </div>

        {/* Chevron */}
        <div className="text-gray-500 shrink-0">
          {open ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </div>
      </div>

      {/* Expanded body */}
      {open && (
        <div className="overflow-hidden">
          <div className="border-t border-slate-700/60 px-5 pb-5 pt-4 space-y-4">

              {/* Action buttons */}
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-purple-600/20 border border-purple-500/40 text-purple-300 hover:bg-purple-600/30 transition-colors"
                >
                  <Download size={14} /> Download .pkl
                </button>
                <button
                  onClick={handleDownloadCode}
                  disabled={!summary?.api_export_code}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-cyan-600/20 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-600/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <FileCode size={14} /> Download Code
                </button>
                {!isActive && (
                  <button
                    onClick={() => onRollback(meta.version)}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-cyan-600/20 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-600/30 transition-colors"
                  >
                    <RotateCcw size={14} /> Set Active
                  </button>
                )}
                <button
                  onClick={handleDelete}
                  disabled={deleting}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-red-600/20 border border-red-500/40 text-red-400 hover:bg-red-600/30 transition-colors disabled:opacity-50"
                >
                  <Trash2 size={14} /> {deleting ? 'Deleting…' : 'Delete'}
                </button>
              </div>

              {loading && (
                <div className="flex items-center gap-2 text-gray-500 text-sm py-4">
                  <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}>
                    <Cpu size={14} />
                  </motion.div>
                  Loading artifact data…
                </div>
              )}

              {summary && (
                <>
                  {/* Tab bar */}
                  <div className="flex gap-1 border-b border-slate-700">
                    {([
                      { key: 'logs',    label: 'Training Logs',   icon: BarChart3 },
                      { key: 'code',    label: 'API Export',      icon: FileCode },
                      { key: 'samples', label: 'Inference',       icon: Layers },
                      { key: 'drift',   label: 'Drift Hooks',     icon: Database },
                    ] as const).map(({ key, label, icon: Icon }) => (
                      <button
                        key={key}
                        onClick={() => setTab(key)}
                        className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition-colors ${
                          tab === key
                            ? 'border-purple-500 text-purple-300'
                            : 'border-transparent text-gray-500 hover:text-gray-300'
                        }`}
                      >
                        <Icon size={12} /> {label}
                      </button>
                    ))}
                  </div>

                  {/* Tab panels — all rendered, visibility toggled via CSS for instant switching */}
                  <div className={tab === 'logs' ? 'block space-y-3' : 'hidden'}>
                    <div className="flex flex-wrap gap-2">
                      <Pill label="CV Score" value={(() => { const s = summary.experiment?.best_score ?? (summary.metadata as any)?.best_score ?? meta.best_score; const n = parseFloat(s); return isNaN(n) ? '—' : n.toFixed(4) })()} color="text-purple-400" />
                      {summary.training_log[0] && (
                        <>
                          <Pill label="Test Score"  value={parseFloat(summary.training_log.find(r => r.model_name === meta.model_name)?.test_score ?? '0').toFixed(4)} color="text-cyan-400" />
                          <Pill label="Train Score" value={parseFloat(summary.training_log.find(r => r.model_name === meta.model_name)?.train_score ?? '0').toFixed(4)} color="text-green-400" />
                          {!isClf && summary.training_log.find(r => r.model_name === meta.model_name)?.rmse && (
                            <Pill label="RMSE" value={parseFloat(summary.training_log.find(r => r.model_name === meta.model_name)!.rmse!).toLocaleString('en-US', { maximumFractionDigits: 2 })} color="text-orange-400" />
                          )}
                          {!isClf && summary.training_log.find(r => r.model_name === meta.model_name)?.mae && (
                            <Pill label="MAE" value={parseFloat(summary.training_log.find(r => r.model_name === meta.model_name)!.mae!).toLocaleString('en-US', { maximumFractionDigits: 2 })} color="text-yellow-400" />
                          )}
                        </>
                      )}
                      <Pill label="Train Time" value={`${meta.training_time_s?.toFixed(1)}s`} color="text-gray-300" />
                    </div>
                    {summary.training_log.length > 0 && (
                      <div className="overflow-x-auto rounded-lg border border-slate-700">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-slate-700 bg-slate-800/60">
                              <th className="text-left px-3 py-2 text-gray-400 font-medium">Model</th>
                              <th className="text-right px-3 py-2 text-gray-400 font-medium">Composite</th>
                              <th className="text-right px-3 py-2 text-gray-400 font-medium">CV</th>
                              <th className="text-right px-3 py-2 text-gray-400 font-medium">Test</th>
                              <th className="text-right px-3 py-2 text-gray-400 font-medium">Train</th>
                              <th className="text-right px-3 py-2 text-gray-400 font-medium">Std</th>
                              <th className="text-right px-3 py-2 text-gray-400 font-medium">Gap</th>
                            </tr>
                          </thead>
                          <tbody>
                            {summary.training_log
                              .slice()
                              .filter(row => parseFloat(row.composite_score) > -999)
                              .sort((a, b) => parseFloat(b.composite_score) - parseFloat(a.composite_score))
                              .map((row, i) => (
                                <tr
                                  key={row.model_name}
                                  className={`border-b border-slate-700/50 ${
                                    row.model_name === meta.model_name ? 'bg-purple-500/10' : i % 2 === 0 ? 'bg-transparent' : 'bg-slate-800/20'
                                  }`}
                                >
                                  <td className="px-3 py-2 font-medium text-white flex items-center gap-1.5">
                                    {row.model_name === meta.model_name && <CheckCircle size={10} className="text-purple-400 shrink-0" />}
                                    <span>{fmtModel(row.model_name)}</span>
                                  </td>
                                  <td className="px-3 py-2 text-right font-mono text-purple-300">{parseFloat(row.composite_score).toFixed(4)}</td>
                                  <td className="px-3 py-2 text-right font-mono text-cyan-300">{parseFloat(row.cv_score || '0').toFixed(4)}</td>
                                  <td className="px-3 py-2 text-right font-mono text-green-300">{parseFloat(row.test_score || '0').toFixed(4)}</td>
                                  <td className="px-3 py-2 text-right font-mono text-orange-300">{parseFloat(row.train_score || '0').toFixed(4)}</td>
                                  <td className="px-3 py-2 text-right font-mono text-gray-400">{parseFloat(row.cv_std || '0').toFixed(4)}</td>
                                  <td className={`px-3 py-2 text-right font-mono ${parseFloat(row.overfit_gap || '0') > 0.15 ? 'text-red-400' : parseFloat(row.overfit_gap || '0') > 0.05 ? 'text-yellow-400' : 'text-green-400'}`}>
                                    {(parseFloat(row.overfit_gap || '0') * 100).toFixed(1)}%
                                  </td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                    {summary.experiment?.best_params && Object.keys(summary.experiment.best_params).length > 0 && (
                      <div>
                        <p className="text-xs text-gray-500 mb-2">Best Hyperparameters</p>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(summary.experiment.best_params).map(([k, v]) => (
                            <span key={k} className="px-2 py-1 rounded-md bg-slate-800 border border-purple-500/20 text-xs font-mono text-cyan-300">
                              {k}: <span className="text-white">{String(v)}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className={tab === 'code' ? 'block space-y-3' : 'hidden'}>
                    <p className="text-xs text-gray-500">
                      Drop-in FastAPI inference endpoint. Place alongside <code className="text-cyan-400">best_model.pkl</code> and run with <code className="text-cyan-400">uvicorn api_export:app</code>.
                    </p>
                    {summary.api_export_code
                      ? <CodeBlock code={summary.api_export_code} language="python" />
                      : <p className="text-gray-500 text-sm">No API export available.</p>
                    }
                  </div>

                  <div className={tab === 'samples' ? 'block space-y-3' : 'hidden'}>
                    {summary.inference_samples && summary.inference_samples.length > 0 ? (
                      <>
                        <p className="text-xs text-gray-500">5 sample predictions from the training set.</p>
                        <div className="space-y-2">
                          {summary.inference_samples.map((s: any, i: number) => (
                            <div key={i} className="rounded-lg border border-slate-700 p-3 bg-slate-800/30">
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-xs text-gray-500">Sample {i + 1}</span>
                                <div className="flex items-center gap-3 text-xs">
                                  <span className="text-gray-400">Actual: <span className="text-white font-mono">{String(s.actual)}</span></span>
                                  <span className={`font-medium font-mono ${Math.abs(s.prediction - s.actual) / (Math.abs(s.actual) || 1) < 0.1 ? 'text-green-400' : 'text-orange-400'}`}>
                                    Pred: {typeof s.prediction === 'number' ? s.prediction.toFixed(4) : s.prediction}
                                  </span>
                                </div>
                              </div>
                              <div className="flex flex-wrap gap-1">
                                {Object.entries(s.input).slice(0, 6).map(([k, v]) => (
                                  <span key={k} className="px-1.5 py-0.5 rounded bg-slate-700/60 text-xs font-mono text-gray-400">
                                    {k}: <span className="text-gray-200">{String(v)}</span>
                                  </span>
                                ))}
                                {Object.keys(s.input).length > 6 && (
                                  <span className="px-1.5 py-0.5 rounded bg-slate-700/60 text-xs text-gray-500">+{Object.keys(s.input).length - 6} more</span>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </>
                    ) : (
                      <p className="text-gray-500 text-sm">Inference samples not available — model was not saved as .pkl.</p>
                    )}
                  </div>

                  <div className={tab === 'drift' ? 'block space-y-3' : 'hidden'}>
                    {summary.drift_hooks ? (
                      <>
                        <p className="text-xs text-gray-500">
                          Baseline feature statistics for drift detection. Compare incoming batch stats against these thresholds.
                        </p>
                        <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                          <div className="rounded-lg border border-slate-700 p-3 bg-slate-800/30">
                            <span className="text-gray-500">Baseline rows</span>
                            <div className="text-white font-mono font-bold mt-1">{summary.drift_hooks.n_rows?.toLocaleString()}</div>
                          </div>
                          <div className="rounded-lg border border-slate-700 p-3 bg-slate-800/30">
                            <span className="text-gray-500">Features tracked</span>
                            <div className="text-white font-mono font-bold mt-1">{Object.keys(summary.drift_hooks.feature_stats || {}).length}</div>
                          </div>
                        </div>
                        <CodeBlock
                          code={JSON.stringify(summary.drift_hooks.drift_thresholds, null, 2)}
                          language="json"
                        />
                      </>
                    ) : (
                      <p className="text-gray-500 text-sm">Drift hooks not available.</p>
                    )}
                  </div>
                </>
              )}
          </div>
        </div>
      )}
    </motion.div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function ArtifactsPage() {
  const router = useRouter()
  const { datasetId, mounted } = useStoredDataset()
  const [versions, setVersions] = useState<VersionMeta[]>([])
  const [activeVersion, setActiveVersion] = useState<string | null>(null)
  const [resolvedId, setResolvedId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [localPayloads, setLocalPayloads] = useState<Record<string, LocalArtifactPayload>>({})

  // Convert ArtifactPayload from Firestore into VersionMeta + Summary shapes
  const payloadToMeta = (p: LocalArtifactPayload): VersionMeta => ({
    version:         p.version,
    model_name:      p.model_name,
    task_type:       p.task_type,
    best_score:      p.best_score,
    timestamp:       p.timestamp,
    training_time_s: p.training_time_s,
    framework:       (p.metadata as any)?.framework ?? 'sklearn',
    format:          'pkl',
    model_size_kb:   p.model_size_kb,
  })

  const payloadToSummary = (p: LocalArtifactPayload): Summary => {
    const normalizeRow = (r: any): TrainingRow => {
      const m = r.metrics ?? {}
      const isClf = p.task_type === 'classification'
      return {
        model_name:      String(r.model_name ?? ''),
        composite_score: String(r.composite_score ?? r.score ?? ''),
        cv_score:        String(r.cv_score ?? m[isClf ? 'accuracy' : 'r2_score'] ?? ''),
        test_score:      String(r.test_score ?? m[isClf ? 'test_accuracy' : 'test_r2'] ?? ''),
        train_score:     String(r.train_score ?? m[isClf ? 'train_accuracy' : 'train_r2'] ?? ''),
        f1_score:        String(r.f1_score ?? m.f1_score ?? ''),
        rmse:            String(r.rmse ?? m.rmse ?? ''),
        mae:             String(r.mae ?? m.mae ?? ''),
        cv_std:          String(r.cv_std ?? m.cv_std ?? ''),
        overfit_gap:     String(r.overfit_gap ?? m.overfit_gap ?? ''),
        best_params:     typeof r.best_params === 'string' ? r.best_params
                           : JSON.stringify(r.best_params ?? r.params ?? m.best_params ?? {}),
      }
    }

    // training_log: stored CSV rows → evaluation_results → experiment_log.models_evaluated
    let trainingLog: TrainingRow[] = []
    if (Array.isArray(p.training_log) && p.training_log.length > 0) {
      trainingLog = p.training_log.map(normalizeRow)
    } else if (Array.isArray(p.evaluation_results) && p.evaluation_results.length > 0) {
      trainingLog = p.evaluation_results.map(normalizeRow)
    } else {
      const models = (p.experiment_log as any)?.models_evaluated
      if (Array.isArray(models) && models.length > 0) trainingLog = models.map(normalizeRow)
    }

    // api_export_code: stored value → generate client-side
    const apiCode = p.api_export_code ?? (() => {
      const features = p.selected_features ?? []
      const fields = features.map(f => `    ${f.replace(/\W/g, '_')}: float`).join('\n')
      return `"""
Auto-generated FastAPI inference endpoint
Model   : ${p.model_name}
Version : ${p.version}
Task    : ${p.task_type}
"""
import pickle, os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TestLabs Model API — ${p.model_name}")

with open(os.path.join(os.path.dirname(__file__), "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
${fields}

@app.post("/predict")
def predict(data: InputData):
    result = model.predict([list(data.dict().values())])[0]
    return {"prediction": float(result), "model": "${p.model_name}", "version": "${p.version}"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "${p.model_name}", "version": "${p.version}"}
`
    })()

    return {
      metadata:          (p.metadata as any) ?? { best_score: p.best_score, model_name: p.model_name, task_type: p.task_type, timestamp: p.timestamp, training_time_s: p.training_time_s, framework: 'sklearn' },
      experiment:        { best_score: p.best_score, best_params: p.best_params, ...(p.experiment_log as any ?? {}) },
      training_log:      trainingLog,
      reproducibility:   p.reproducibility as any,
      inference_samples: p.inference_samples as any[] ?? null,
      api_export_code:   apiCode,
      drift_hooks:       p.drift_hooks as any,
      model_file_exists: p.model_file_exists,
      model_size_kb:     p.model_size_kb,
    }
  }

  const loadVersions = async () => {
    if (!datasetId) return
    setLoading(true)
    setError(null)
    try {
      // 1. Try backend disk first
      const data = await apiService.getArtifactVersions(datasetId)
      let versions: VersionMeta[] = (data.versions || []).filter((v: VersionMeta) => v.model_name)
      let active = data.latest || null
      let resolved = datasetId

      if (versions.length === 0) {
        const { datasets } = await apiService.getArtifactDatasets()
        for (const ds of datasets) {
          if (ds === datasetId) continue
          const fallback = await apiService.getArtifactVersions(ds)
          const fallbackVersions = (fallback.versions || []).filter((v: VersionMeta) => v.model_name)
          if (fallbackVersions.length > 0) {
            versions = fallbackVersions
            active = fallback.latest
            resolved = ds
            break
          }
        }
      }

      // 2. Fall back to localStorage if backend disk is empty
      if (versions.length === 0) {
        const versionList = getLocalArtifactVersions(datasetId)
        if (versionList.length > 0) {
          const payloadMap: Record<string, LocalArtifactPayload> = {}
          versionList.forEach(v => {
            const p = getLocalArtifact(datasetId, v)
            if (p) payloadMap[v] = p
          })
          setLocalPayloads(payloadMap)
          versions = Object.values(payloadMap).map(payloadToMeta)
          active = versions[0]?.version ?? null
          resolved = datasetId
        }
      }

      setVersions(versions)
      setActiveVersion(active)
      setResolvedId(resolved)
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to load artifacts')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (datasetId && mounted) loadVersions()
  }, [datasetId, mounted])

  const handleRollback = async (version: string) => {
    if (!resolvedId) return
    await apiService.rollbackVersion(resolvedId, version)
    setActiveVersion(version)
  }

  const handleDelete = async (version: string) => {
    if (!resolvedId) return
    deleteLocalArtifactVersion(resolvedId, version)
    await apiService.deleteVersion(resolvedId, version).catch(() => {})
    await loadVersions()
  }

  if (!mounted) return null

  if (!datasetId) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Package size={40} className="text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 mb-4">No dataset selected</p>
          <button onClick={() => router.push('/upload')}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-lg text-white font-medium">
            Upload a Dataset
          </button>
        </div>
      </div>
    )
  }

  return (
    <motion.div initial="hidden" animate="visible" variants={containerVariants} className="min-h-screen bg-black p-8">
      <div className="max-w-5xl mx-auto">

        {/* Header */}
        <motion.div variants={itemVariants} className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            <Package size={28} className="text-cyan-400" />
            <h1 className="text-3xl font-bold text-white">Artifacts & Model Versions</h1>
          </div>
          <p className="text-gray-400 text-sm">Trained models, experiment logs, API export code, and drift baselines</p>
        </motion.div>

        {/* Loading */}
        {loading && (
          <motion.div variants={itemVariants} className="flex items-center gap-3 text-gray-400 py-12 justify-center">
            <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}>
              <Cpu size={20} />
            </motion.div>
            Loading artifacts…
          </motion.div>
        )}

        {/* Error */}
        {error && (
          <motion.div variants={itemVariants} className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 mb-6 flex items-center gap-3">
            <AlertCircle size={18} className="text-red-400 shrink-0" />
            <span className="text-red-300 text-sm">{error}</span>
          </motion.div>
        )}

        {/* Empty state */}
        {!loading && !error && versions.length === 0 && (
          <motion.div variants={itemVariants}
            className="rounded-xl border border-slate-700 p-16 text-center"
            style={{ background: 'rgba(15,15,30,0.5)' }}>
            <Package size={48} className="mx-auto text-gray-600 mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No Artifacts Yet</h3>
            <p className="text-gray-400 mb-6 text-sm">Run the pipeline to generate model artifacts and versions</p>
            <button onClick={() => router.push('/pipeline')}
              className="px-6 py-3 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-lg text-white font-medium hover:shadow-lg hover:shadow-cyan-500/30 transition-all">
              Go to Pipeline
            </button>
          </motion.div>
        )}

        {/* Version list */}
        {!loading && versions.length > 0 && resolvedId && (
          <motion.div variants={containerVariants} className="space-y-4">
            {versions.map((v, i) => (
              <VersionCard
                key={v.version}
                meta={v}
                isActive={v.version === activeVersion}
                isLatest={i === 0}
                datasetId={resolvedId}
                getLocalSummary={(version) => {
                  const p = localPayloads[version] ?? getLocalArtifact(resolvedId, version)
                  return p ? payloadToSummary(p) : null
                }}
                onRollback={handleRollback}
                onDelete={handleDelete}
              />
            ))}
          </motion.div>
        )}

      </div>
    </motion.div>
  )
}
