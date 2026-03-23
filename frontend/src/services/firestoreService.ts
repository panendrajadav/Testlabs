import {
  doc, setDoc, getDoc, getDocs, deleteDoc, updateDoc,
  collection, query, orderBy, serverTimestamp, Timestamp,
} from 'firebase/firestore'
import { db } from '@/lib/firebase'
import type { StoredDataset } from '@/hooks/useStoredDataset'
import type { AuthUser } from '@/hooks/useAuth'

// ── Collection paths ──────────────────────────────────────────────────────────
// experiments/{dataset_id}              — lightweight experiment metadata
// artifacts/{dataset_id}/{version}      — full artifact data per version
// sessions/{username}                   — current dataset pointer + auth

const EXPERIMENTS = 'experiments'
const ARTIFACTS   = 'artifacts'
const SESSIONS    = 'sessions'

// ── Helpers ───────────────────────────────────────────────────────────────────

function stripUndefined<T extends object>(obj: T): T {
  return Object.fromEntries(
    Object.entries(obj)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => [
        k,
        v && typeof v === 'object' && !Array.isArray(v) ? stripUndefined(v as object) : v
      ])
  ) as T
}

// ── Experiments ───────────────────────────────────────────────────────────────

export async function fsSetExperiment(data: StoredDataset): Promise<void> {
  const ref = doc(db, EXPERIMENTS, data.dataset_id)
  await setDoc(ref, {
    ...stripUndefined(data),
    started_at: data.started_at ?? new Date().toISOString(),
    updatedAt:  serverTimestamp(),
  }, { merge: true })
}

export async function fsUpdateExperiment(
  datasetId: string,
  patch: Partial<StoredDataset>
): Promise<void> {
  const ref = doc(db, EXPERIMENTS, datasetId)
  await setDoc(ref, { ...stripUndefined(patch as object), updatedAt: serverTimestamp() }, { merge: true })
}

export async function fsDeleteExperiment(datasetId: string): Promise<void> {
  await deleteDoc(doc(db, EXPERIMENTS, datasetId))
}

export async function fsGetAllExperiments(): Promise<StoredDataset[]> {
  const q = query(collection(db, EXPERIMENTS), orderBy('updatedAt', 'desc'))
  const snap = await getDocs(q)
  return snap.docs.map(d => {
    const data = d.data()
    if (data.updatedAt instanceof Timestamp) delete data.updatedAt
    return data as StoredDataset
  })
}

// ── Artifacts (experiment log, training log, model code) ──────────────────────

export interface ArtifactPayload {
  // Identity
  version:            string
  dataset_id:         string
  // Best model summary
  model_name:         string
  task_type:          string
  best_score:         number | null
  best_params:        Record<string, unknown>
  timestamp:          string
  training_time_s:    number
  target_column:      string
  selected_features:  string[]
  justification:      string | null
  is_underfit:        boolean
  // All models evaluated (full metrics per model)
  evaluation_results: Record<string, unknown>[]
  // Artifact file contents
  metadata:           Record<string, unknown> | null
  experiment_log:     Record<string, unknown> | null
  training_log:       Record<string, unknown>[] | null
  reproducibility:    Record<string, unknown> | null
  inference_samples:  Record<string, unknown>[] | null
  drift_hooks:        Record<string, unknown> | null
  api_export_code:    string | null
  model_file_exists:  boolean
  model_size_kb:      number | null
  // Pipeline result extras
  agent_logs:         string[]
  eda_summary:        Record<string, unknown> | null
  shap_values:        Record<string, unknown> | null
  roc_data:           Record<string, unknown> | null
  preprocessing_report: Record<string, unknown> | null
}

export async function fsSetArtifacts(
  datasetId: string,
  version: string,
  payload: ArtifactPayload
): Promise<void> {
  // artifacts/{datasetId}/{version}  — subcollection doc
  const ref = doc(db, ARTIFACTS, datasetId, 'versions', version)
  await setDoc(ref, { ...stripUndefined(payload as object), savedAt: serverTimestamp() })
}

export async function fsDeleteArtifacts(datasetId: string): Promise<void> {
  // Delete all version docs under artifacts/{datasetId}/versions/
  const snap = await getDocs(collection(db, ARTIFACTS, datasetId, 'versions'))
  await Promise.all(snap.docs.map(d => deleteDoc(d.ref)))
  // Delete the parent doc if it exists
  await deleteDoc(doc(db, ARTIFACTS, datasetId))
}

// ── Active session (current dataset per user) ─────────────────────────────────

export async function fsSetCurrentDataset(
  username: string,
  dataset: StoredDataset | null
): Promise<void> {
  const ref = doc(db, SESSIONS, username)
  await setDoc(ref, {
    currentDataset: dataset ? stripUndefined(dataset as object) : null,
    updatedAt: serverTimestamp()
  }, { merge: true })
}

export async function fsGetCurrentDataset(username: string): Promise<StoredDataset | null> {
  const snap = await getDoc(doc(db, SESSIONS, username))
  if (!snap.exists()) return null
  return snap.data().currentDataset ?? null
}

// ── Auth session ──────────────────────────────────────────────────────────────

export async function fsSetAuthSession(user: AuthUser): Promise<void> {
  await setDoc(doc(db, SESSIONS, user.username), {
    username:  user.username,
    loggedIn:  true,
    updatedAt: serverTimestamp(),
  }, { merge: true })
}

export async function fsClearAuthSession(username: string): Promise<void> {
  await updateDoc(doc(db, SESSIONS, username), { loggedIn: false, updatedAt: serverTimestamp() })
}
