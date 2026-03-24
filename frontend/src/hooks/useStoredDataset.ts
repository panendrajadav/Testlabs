'use client'

import { useState, useEffect } from 'react'
import { apiService } from '@/services/api'

export interface StoredDataset {
  dataset_id: string
  filename: string
  rows: number
  columns: number
  column_names: string[]
  preview: Record<string, unknown>[]
  started_at?: string
  best_model?: string
  best_score?: number
  task_type?: string
  status?: 'running' | 'completed' | 'failed'
}

let _cache: StoredDataset | null = undefined as any
let _experimentsCache: StoredDataset[] | null = null

const _changeListeners = new Set<(d: StoredDataset | null) => void>()
const _expListeners = new Set<(d: StoredDataset[]) => void>()

function _loadExperiments(): StoredDataset[] {
  if (_experimentsCache !== null) return _experimentsCache
  try {
    const raw = localStorage.getItem('experiments')
    _experimentsCache = raw ? JSON.parse(raw) : []
  } catch { _experimentsCache = [] }
  return _experimentsCache!
}

export function setStoredDataset(data: StoredDataset | null) {
  _cache = data
  if (data) {
    localStorage.setItem('currentDataset', JSON.stringify(data))
    const exps = _loadExperiments()
    const idx = exps.findIndex(e => e.dataset_id === data.dataset_id)
    const entry = { ...data, started_at: data.started_at ?? new Date().toISOString() }
    if (idx >= 0) exps[idx] = entry
    else exps.unshift(entry)
    _experimentsCache = exps
    localStorage.setItem('experiments', JSON.stringify(exps))
    _expListeners.forEach(fn => fn(exps))
  } else {
    localStorage.removeItem('currentDataset')
  }
  window.dispatchEvent(new Event('datasetUpdated'))
  _changeListeners.forEach((fn) => fn(data))
}

export function updateExperimentResult(datasetId: string, patch: Partial<StoredDataset>) {
  const exps = _loadExperiments()
  const idx = exps.findIndex(e => e.dataset_id === datasetId)
  if (idx >= 0) {
    exps[idx] = { ...exps[idx], ...patch }
    _experimentsCache = exps
    localStorage.setItem('experiments', JSON.stringify(exps))
    _expListeners.forEach(fn => fn(exps))
  }
}

export function renameExperiment(datasetId: string, newName: string) {
  const exps = _loadExperiments()
  const idx = exps.findIndex(e => e.dataset_id === datasetId)
  if (idx >= 0) {
    exps[idx] = { ...exps[idx], filename: newName }
    _experimentsCache = exps
    localStorage.setItem('experiments', JSON.stringify(exps))
    if (_cache?.dataset_id === datasetId) {
      _cache = { ..._cache, filename: newName }
      localStorage.setItem('currentDataset', JSON.stringify(_cache))
      window.dispatchEvent(new Event('datasetUpdated'))
    }
    _expListeners.forEach(fn => fn(exps))
  }
}

export function deleteExperiment(datasetId: string) {
  const exps = _loadExperiments().filter(e => e.dataset_id !== datasetId)
  _experimentsCache = exps
  localStorage.setItem('experiments', JSON.stringify(exps))
  if (_cache?.dataset_id === datasetId) {
    _cache = null
    localStorage.removeItem('currentDataset')
    window.dispatchEvent(new Event('datasetUpdated'))
  }
  _expListeners.forEach(fn => fn(exps))
  // Remove all locally stored artifact versions and chat history for this dataset
  Object.keys(localStorage)
    .filter(k => k.startsWith(`artifact_${datasetId}_`) || k === `chat_${datasetId}`)
    .forEach(k => localStorage.removeItem(k))
  localStorage.removeItem(`artifact_versions_${datasetId}`)
  // Delete server-side data
  apiService.deleteDataset(datasetId).catch(console.error)
}

export function useExperiments() {
  const [experiments, setExperiments] = useState<StoredDataset[]>([])
  useEffect(() => {
    setExperiments(_loadExperiments())
    const refresh = () => setExperiments([..._loadExperiments()])
    _expListeners.add(setExperiments)
    window.addEventListener('storage', refresh)
    return () => {
      _expListeners.delete(setExperiments)
      window.removeEventListener('storage', refresh)
    }
  }, [])
  return experiments
}

export function useStoredDataset() {
  const [dataset, setDataset] = useState<StoredDataset | null>(() => {
    if (_cache !== undefined) return _cache
    return null
  })
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    if (_cache === undefined) {
      const raw = localStorage.getItem('currentDataset')
      _cache = raw ? JSON.parse(raw) : null
    }
    setDataset(_cache)
    setMounted(true)

    const refresh = () => {
      const raw = localStorage.getItem('currentDataset')
      _cache = raw ? JSON.parse(raw) : null
      setDataset(_cache)
    }

    _changeListeners.add(setDataset)
    window.addEventListener('datasetUpdated', refresh)
    window.addEventListener('storage', refresh)
    return () => {
      _changeListeners.delete(setDataset)
      window.removeEventListener('datasetUpdated', refresh)
      window.removeEventListener('storage', refresh)
    }
  }, [])

  return { dataset, datasetId: dataset?.dataset_id ?? null, mounted }
}

// ── Local artifact storage ────────────────────────────────────────────────────

export interface LocalArtifactPayload {
  version: string
  dataset_id: string
  model_name: string
  task_type: string
  best_score: number | null
  best_params: Record<string, unknown>
  timestamp: string
  training_time_s: number
  target_column: string
  selected_features: string[]
  justification: string | null
  is_underfit: boolean
  evaluation_results: Record<string, unknown>[]
  metadata: Record<string, unknown> | null
  experiment_log: Record<string, unknown> | null
  training_log: Record<string, unknown>[] | null
  reproducibility: Record<string, unknown> | null
  inference_samples: Record<string, unknown>[] | null
  drift_hooks: Record<string, unknown> | null
  api_export_code: string | null
  model_file_exists: boolean
  model_size_kb: number | null
  agent_logs: string[]
  eda_summary: Record<string, unknown> | null
  shap_values: Record<string, unknown> | null
  roc_data: Record<string, unknown> | null
  preprocessing_report: Record<string, unknown> | null
  savedAt: string
}

export function saveLocalArtifact(datasetId: string, version: string, payload: Omit<LocalArtifactPayload, 'savedAt'>) {
  const full: LocalArtifactPayload = { ...payload, savedAt: new Date().toISOString() }
  try {
    localStorage.setItem(`artifact_${datasetId}_${version}`, JSON.stringify(full))
    // Maintain ordered version list
    const listKey = `artifact_versions_${datasetId}`
    const existing: string[] = JSON.parse(localStorage.getItem(listKey) ?? '[]')
    if (!existing.includes(version)) {
      existing.unshift(version)
      localStorage.setItem(listKey, JSON.stringify(existing))
    }
  } catch (e) {
    console.warn('saveLocalArtifact: localStorage write failed', e)
  }
}

export function getLocalArtifact(datasetId: string, version: string): LocalArtifactPayload | null {
  try {
    const raw = localStorage.getItem(`artifact_${datasetId}_${version}`)
    return raw ? JSON.parse(raw) : null
  } catch { return null }
}

export function getLocalArtifactVersions(datasetId: string): string[] {
  try {
    return JSON.parse(localStorage.getItem(`artifact_versions_${datasetId}`) ?? '[]')
  } catch { return [] }
}

export function deleteLocalArtifactVersion(datasetId: string, version: string) {
  localStorage.removeItem(`artifact_${datasetId}_${version}`)
  const listKey = `artifact_versions_${datasetId}`
  const existing: string[] = JSON.parse(localStorage.getItem(listKey) ?? '[]')
  localStorage.setItem(listKey, JSON.stringify(existing.filter(v => v !== version)))
}

// ── Chat history storage ──────────────────────────────────────────────────────

export interface StoredMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  chart?: Record<string, unknown> | null
}

export function getChatHistory(datasetId: string): StoredMessage[] {
  try {
    const raw = localStorage.getItem(`chat_${datasetId}`)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

export function saveChatHistory(datasetId: string, messages: StoredMessage[]) {
  try {
    localStorage.setItem(`chat_${datasetId}`, JSON.stringify(messages))
  } catch (e) {
    console.warn('saveChatHistory: localStorage write failed', e)
  }
}

export function clearChatHistory(datasetId: string) {
  localStorage.removeItem(`chat_${datasetId}`)
}
