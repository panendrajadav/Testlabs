export const PIPELINE_STAGES = [
  { id: 'eda', name: 'EDA', label: 'Exploratory Data Analysis' },
  { id: 'preprocessing', name: 'Preprocessing', label: 'Data Preprocessing' },
  { id: 'feature-engineering', name: 'Feature Engineering', label: 'Feature Engineering' },
  { id: 'model-selection', name: 'Model Selection', label: 'Model Selection' },
  { id: 'hpo', name: 'HPO', label: 'Hyperparameter Optimization' },
  { id: 'evaluation', name: 'Evaluation', label: 'Model Evaluation' },
]

export const ANIMATION_DURATION = {
  fast: 0.2,
  normal: 0.3,
  slow: 0.5,
  verySlow: 1,
}

export const COLORS = {
  primary: '#8b5cf6',
  secondary: '#06b6d4',
  success: '#10b981',
  warning: '#f59e0b',
  error: '#ef4444',
  accent: '#ec4899',
}

export const API_ENDPOINTS = {
  upload: '/upload',
  pipelineStatus: '/pipeline-status',
  pipelineResults: '/pipeline-results',
  modelMetrics: '/model-metrics',
  datasetChat: '/dataset-chat',
  startPipeline: '/pipeline/start',
}
