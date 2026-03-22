'use client'

import { useState, useEffect } from 'react'
import {
  fsSetExperiment, fsUpdateExperiment, fsDeleteExperiment, fsDeleteArtifacts,
  fsSetCurrentDataset,
} from '@/services/firestoreService'
import { getAuth } from '@/hooks/useAuth'
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
    // Upsert into experiments list
    const exps = _loadExperiments()
    const idx = exps.findIndex(e => e.dataset_id === data.dataset_id)
    const entry = { ...data, started_at: data.started_at ?? new Date().toISOString() }
    if (idx >= 0) exps[idx] = entry
    else exps.unshift(entry)
    _experimentsCache = exps
    localStorage.setItem('experiments', JSON.stringify(exps))
    _expListeners.forEach(fn => fn(exps))
    // Firestore sync (fire-and-forget)
    fsSetExperiment(entry).catch(console.error)
    const user = getAuth()
    if (user) fsSetCurrentDataset(user.username, entry).catch(console.error)
  } else {
    localStorage.removeItem('currentDataset')
    const user = getAuth()
    if (user) fsSetCurrentDataset(user.username, null).catch(console.error)
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
    // Firestore sync
    fsUpdateExperiment(datasetId, patch).catch(console.error)
  }
}

export function renameExperiment(datasetId: string, newName: string) {
  const exps = _loadExperiments()
  const idx = exps.findIndex(e => e.dataset_id === datasetId)
  if (idx >= 0) {
    exps[idx] = { ...exps[idx], filename: newName }
    _experimentsCache = exps
    localStorage.setItem('experiments', JSON.stringify(exps))
    // Also update currentDataset if it's the active one
    if (_cache?.dataset_id === datasetId) {
      _cache = { ..._cache, filename: newName }
      localStorage.setItem('currentDataset', JSON.stringify(_cache))
      window.dispatchEvent(new Event('datasetUpdated'))
    }
    _expListeners.forEach(fn => fn(exps))
    // Firestore sync
    fsUpdateExperiment(datasetId, { filename: newName }).catch(console.error)
  }
}

export function deleteExperiment(datasetId: string) {
  const exps = _loadExperiments().filter(e => e.dataset_id !== datasetId)
  _experimentsCache = exps
  localStorage.setItem('experiments', JSON.stringify(exps))
  // Clear active dataset if it was the deleted one
  if (_cache?.dataset_id === datasetId) {
    _cache = null
    localStorage.removeItem('currentDataset')
    window.dispatchEvent(new Event('datasetUpdated'))
  }
  _expListeners.forEach(fn => fn(exps))
  // Delete from Firestore (experiment doc + all artifact version docs)
  fsDeleteExperiment(datasetId).catch(console.error)
  fsDeleteArtifacts(datasetId).catch(console.error)
  // Delete all server-side data (CSV + artifacts + results) — fire-and-forget
  apiService.deleteDataset(datasetId).catch(console.error)
}

export function useExperiments() {
  const [experiments, setExperiments] = useState<StoredDataset[]>(() => _loadExperiments())
  useEffect(() => {
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
