export interface UploadResponse {
  dataset_id: string
  filename: string
  rows: number
  columns: number
  column_names: string[]
  preview: Record<string, unknown>[]
}

export interface EvalResult {
  model: string
  model_name: string   // backend uses model_name
  score: number
  metrics: Record<string, number>
  params: Record<string, unknown>
}

export interface PipelineResult {
  target_column: string
  task_type: 'classification' | 'regression'
  best_model: string
  best_score: number
  best_params: Record<string, unknown>
  evaluation_results: EvalResult[]
  eda_summary: Record<string, unknown>
  eda_plots: Record<string, unknown>
  eda_insights: string
  selected_features: string[]
  roc_data: {
    type?: 'binary' | 'multiclass'
    fpr?: number[]
    tpr?: number[]
    auc?: number
    classes?: string[]
    curves?: Record<string, { fpr: number[]; tpr: number[]; auc: number }>
  } | null
  shap_values: { feature_names: string[]; mean_abs_shap: number[] } | null
  agent_logs: string[]
}

export interface PipelineStatus {
  dataset_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: string | null
  result: PipelineResult | null
  error: string | null
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  text: string
  chart?: Record<string, unknown> | null
}

// Legacy aliases kept for component compatibility
export type DatasetInfo = UploadResponse
export type ModelMetrics = Record<string, number>
