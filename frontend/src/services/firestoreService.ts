import { db } from '@/lib/firebase'
import { collection, doc, setDoc, getDoc, getDocs, query, orderBy, limit, deleteDoc } from 'firebase/firestore'

export interface ArtifactPayload {
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
}

export async function fsSetArtifacts(datasetId: string, version: string, payload: ArtifactPayload) {
  const docRef = doc(db, 'artifacts', datasetId, 'versions', version)
  await setDoc(docRef, { ...payload, savedAt: new Date().toISOString() })
}

export async function fsGetArtifacts(datasetId: string, version: string): Promise<ArtifactPayload | null> {
  const docRef = doc(db, 'artifacts', datasetId, 'versions', version)
  const snap = await getDoc(docRef)
  return snap.exists() ? (snap.data() as ArtifactPayload) : null
}

export async function fsGetAllArtifactVersions(datasetId: string): Promise<ArtifactPayload[]> {
  const versionsRef = collection(db, 'artifacts', datasetId, 'versions')
  const q = query(versionsRef, orderBy('timestamp', 'desc'))
  const snapshot = await getDocs(q)
  return snapshot.docs.map(doc => doc.data() as ArtifactPayload)
}

export async function fsDeleteArtifactVersion(datasetId: string, version: string) {
  const docRef = doc(db, 'artifacts', datasetId, 'versions', version)
  await deleteDoc(docRef)
}

export async function fsSetExperiment(datasetId: string, data: Record<string, unknown>) {
  const docRef = doc(db, 'experiments', datasetId)
  await setDoc(docRef, { ...data, updatedAt: new Date().toISOString() }, { merge: true })
}

export async function fsGetExperiment(datasetId: string) {
  const docRef = doc(db, 'experiments', datasetId)
  const snap = await getDoc(docRef)
  return snap.exists() ? snap.data() : null
}

export async function fsSetAuthSession(userId: string, data: Record<string, unknown>) {
  const docRef = doc(db, 'auth_sessions', userId)
  await setDoc(docRef, { ...data, lastActive: new Date().toISOString() })
}

export async function fsClearAuthSession(userId: string) {
  const docRef = doc(db, 'auth_sessions', userId)
  await deleteDoc(docRef)
}
