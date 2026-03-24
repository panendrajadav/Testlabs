'use client'

import { useState } from 'react'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Zap, TrendingUp, MessageSquare,
  LogOut, Home, Package, Plus, ChevronDown,
  ChevronRight, CheckCircle, XCircle, Loader2, FlaskConical,
  Pencil, Trash2, X, Check,
} from 'lucide-react'
import { useExperiments, useStoredDataset, setStoredDataset, renameExperiment, deleteExperiment, StoredDataset } from '@/hooks/useStoredDataset'
import { getAuth, logout, useAuth } from '@/hooks/useAuth'
import { useQueryClient } from '@tanstack/react-query'

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
function fmtModel(name?: string | null) {
  if (!name) return '—'
  const key = name.toLowerCase().replace(/[-\s]/g, '_')
  return MODEL_NAMES[key] ?? name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

const alwaysNav = [
  { id: 'dashboard', label: 'Dashboard', icon: Home, href: '/' },
]

const experimentNav = [
  { id: 'pipeline',  label: 'Pipeline',     icon: Zap,          href: '/pipeline' },
  { id: 'models',    label: 'Models',       icon: TrendingUp,   href: '/models' },
  { id: 'artifacts', label: 'Artifacts',    icon: Package,      href: '/artifacts' },
  { id: 'chat',      label: 'Dataset Chat', icon: MessageSquare,href: '/chat' },
]

function StatusDot({ status }: { status?: StoredDataset['status'] }) {
  if (status === 'completed') return <CheckCircle size={10} className="text-green-400 shrink-0" />
  if (status === 'failed')    return <XCircle size={10} className="text-red-400 shrink-0" />
  if (status === 'running')   return (
    <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}>
      <Loader2 size={10} className="text-purple-400 shrink-0" />
    </motion.div>
  )
  return <div className="w-2 h-2 rounded-full bg-slate-600 shrink-0" />
}

function ExperimentCard({ exp, isActive, onSelect }: {
  exp: StoredDataset
  isActive: boolean
  onSelect: () => void
}) {
  const [renaming, setRenaming] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [nameVal, setNameVal] = useState(exp.filename)

  const commitRename = () => {
    const trimmed = nameVal.trim()
    if (trimmed && trimmed !== exp.filename) renameExperiment(exp.dataset_id, trimmed)
    setRenaming(false)
  }

  if (confirming) {
    return (
      <div className="rounded-lg border border-red-500/30 bg-red-500/5 px-2 py-2">
        <p className="text-xs text-red-300 mb-2 leading-snug">
          Permanently delete <span className="font-semibold text-white">{exp.filename}</span>?<br />
          <span className="text-red-400/70">Removes dataset, model &amp; all artifacts.</span>
        </p>
        <div className="flex gap-1.5">
          <button
            onClick={() => { deleteExperiment(exp.dataset_id) }}
            className="flex-1 text-xs py-1 rounded bg-red-600 hover:bg-red-500 text-white font-medium transition-colors"
          >
            Delete
          </button>
          <button
            onClick={() => setConfirming(false)}
            className="flex-1 text-xs py-1 rounded bg-slate-700 hover:bg-slate-600 text-gray-300 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className={`rounded-lg transition-colors group ${
      isActive ? 'bg-purple-600/10 border border-purple-500/20' : 'hover:bg-white/5'
    }`}>
      {renaming ? (
        <div className="flex items-center gap-1 px-2 py-1.5" onClick={e => e.stopPropagation()}>
          <input
            autoFocus
            value={nameVal}
            onChange={e => setNameVal(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') commitRename(); if (e.key === 'Escape') setRenaming(false) }}
            className="flex-1 text-xs bg-slate-800 border border-purple-500/40 rounded px-2 py-1 text-white outline-none"
          />
          <button onClick={commitRename} className="text-green-400 hover:text-green-300 p-0.5"><Check size={12} /></button>
          <button onClick={() => setRenaming(false)} className="text-gray-500 hover:text-gray-300 p-0.5"><X size={12} /></button>
        </div>
      ) : (
        <div className="flex items-center gap-1 px-2 py-1.5 cursor-pointer" onClick={onSelect}>
          <StatusDot status={exp.status} />
          <span className={`text-xs font-medium truncate flex-1 ml-1 ${
            isActive ? 'text-purple-300' : 'text-white group-hover:text-purple-300'
          } transition-colors`}>
            {exp.filename}
          </span>
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
            <button
              onClick={e => { e.stopPropagation(); setNameVal(exp.filename); setRenaming(true) }}
              className="p-1 rounded text-gray-500 hover:text-cyan-400 hover:bg-cyan-500/10 transition-colors"
              title="Rename"
            >
              <Pencil size={11} />
            </button>
            <button
              onClick={e => { e.stopPropagation(); setConfirming(true) }}
              className="p-1 rounded text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
              title="Delete"
            >
              <Trash2 size={11} />
            </button>
          </div>
        </div>
      )}
      {!renaming && (
        <div className="flex items-center gap-2 px-3 pb-1.5">
          <span className="text-xs text-gray-600 truncate">
            {exp.best_model ? fmtModel(exp.best_model) : '—'}
          </span>
          {exp.best_score != null && (
            <span className="ml-auto text-xs font-mono text-cyan-600 shrink-0">
              {exp.best_score.toFixed(3)}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

export default function Sidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const queryClient = useQueryClient()
  const experiments = useExperiments()
  const [expOpen, setExpOpen] = useState(true)

  const { user } = useAuth()
  const { datasetId, mounted } = useStoredDataset()
  const hasExperiment = mounted && !!datasetId

  const handleNewExperiment = () => {
    setStoredDataset(null)
    queryClient.removeQueries({ queryKey: ['pipelineStatus'] })
    router.push(getAuth() ? '/upload' : '/login?redirect=/upload')
  }

  const handleSwitchExperiment = (exp: StoredDataset) => {
    setStoredDataset(exp)
    queryClient.removeQueries({ queryKey: ['pipelineStatus'] })
    router.push('/pipeline')
  }

  return (
    <aside className="w-64 bg-gradient-to-b from-slate-950 via-slate-900 to-black border-r border-purple-500/20 flex flex-col overflow-hidden">
      {/* Logo */}
      <div className="p-6 flex items-center gap-3 shrink-0">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
          className="w-8 h-8 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-lg shrink-0"
        />
        <div className="flex flex-col">
          <h1 className="text-lg font-bold text-white">AutoML</h1>
          <p className="text-xs text-purple-400">TestLabs</p>
        </div>
      </div>

      {/* New Experiment button */}
      <div className="px-4 mb-4 shrink-0">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={handleNewExperiment}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-gradient-to-r from-purple-600 to-cyan-600 text-white text-sm font-semibold shadow-lg shadow-purple-900/30 hover:shadow-purple-700/40 transition-shadow"
        >
          <Plus size={16} />
          New Experiment
        </motion.button>
      </div>

      {/* Nav */}
      <nav className="px-4 space-y-1 shrink-0">
        {alwaysNav.map((item, index) => {
          const Icon = item.icon
          const isActive = pathname === item.href
          return (
            <motion.div key={item.id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * 0.05 }}>
              <Link href={item.href} prefetch={true}>
                <motion.div whileHover={{ x: 4 }} className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-sm ${
                  isActive ? 'bg-purple-600/30 text-purple-300 border border-purple-500/50' : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}>
                  <Icon size={17} />
                  <span className="font-medium">{item.label}</span>
                  {isActive && <motion.div layoutId="activeIndicator" className="ml-auto w-1.5 h-1.5 bg-purple-400 rounded-full" />}
                </motion.div>
              </Link>
            </motion.div>
          )
        })}

        {/* Experiment-only nav — shown only when a dataset is active */}
        <AnimatePresence>
          {hasExperiment && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="space-y-1 overflow-hidden"
            >
              <p className="text-xs text-gray-600 uppercase tracking-wider px-3 pt-3 pb-1">Current Experiment</p>
              {experimentNav.map((item, index) => {
                const Icon = item.icon
                const isActive = pathname === item.href
                return (
                  <motion.div key={item.id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * 0.05 }}>
                    <Link href={item.href} prefetch={true}>
                      <motion.div whileHover={{ x: 4 }} className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-sm ${
                        isActive ? 'bg-purple-600/30 text-purple-300 border border-purple-500/50' : 'text-gray-400 hover:text-white hover:bg-white/5'
                      }`}>
                        <Icon size={17} />
                        <span className="font-medium">{item.label}</span>
                        {isActive && <motion.div layoutId="activeIndicator" className="ml-auto w-1.5 h-1.5 bg-purple-400 rounded-full" />}
                      </motion.div>
                    </Link>
                  </motion.div>
                )
              })}
            </motion.div>
          )}
        </AnimatePresence>
      </nav>

      {/* Divider */}
      <div className="mx-4 my-4 border-t border-slate-800 shrink-0" />

      {/* Experiments section */}
      <div className="flex-1 flex flex-col min-h-0 px-4">
        <button
          onClick={() => setExpOpen(o => !o)}
          className="flex items-center gap-2 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 hover:text-gray-300 transition-colors w-full"
        >
          <FlaskConical size={12} />
          Experiments
          <span className="ml-1 px-1.5 py-0.5 rounded bg-slate-800 text-gray-400 font-mono normal-case tracking-normal">
            {mounted ? experiments.length : ''}
          </span>
          <span className="ml-auto">
            {expOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          </span>
        </button>

        <AnimatePresence initial={false}>
          {expOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-y-auto flex-1 space-y-1 pr-1 min-h-0"
              style={{ maxHeight: '100%' }}
            >
              {experiments.length === 0 ? (
                <p className="text-xs text-gray-600 text-center py-4">No experiments yet</p>
              ) : (
                experiments.map((exp) => (
                  <ExperimentCard
                    key={exp.dataset_id}
                    exp={exp}
                    isActive={exp.dataset_id === datasetId}
                    onSelect={() => handleSwitchExperiment(exp)}
                  />
                ))
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Bottom */}
      <div className="p-4 space-y-1 border-t border-purple-500/20 shrink-0">
        {/* User info */}
        {user && (
          <div className="flex items-center gap-3 px-3 py-2 mb-1">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-cyan-500 flex items-center justify-center text-sm font-bold text-white shrink-0">
              P
            </div>
            <span className="text-sm text-gray-300 font-medium truncate">{user.username}</span>
          </div>
        )}
        <motion.button
          whileHover={{ x: 4 }}
          onClick={() => { logout(); router.push('/') }}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:text-red-400 hover:bg-red-500/5 transition-all text-sm"
        >
          <LogOut size={17} />
          <span>Logout</span>
        </motion.button>
      </div>
    </aside>
  )
}
