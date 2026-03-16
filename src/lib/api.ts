import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

export const api = axios.create({
  baseURL: API_BASE_URL,
});

export interface PipelineResult {
  target_column: string;
  task_type: 'classification' | 'regression';
  best_model: string;
  best_score: number;
  best_params: Record<string, any>;
  evaluation_results: any[];
  eda_summary: any;
  eda_plots: any;
  eda_insights: string;
  selected_features: string[];
  roc_data?: any;
  shap_values?: any;
  agent_logs: string[];
}

export interface PipelineStatus {
  dataset_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: string;
  result?: PipelineResult;
  error?: string;
}

export interface ChatResponse {
  answer: string;
  chart?: any;
  code?: string;
}

export const uploadDataset = async (file: File): Promise<{ dataset_id: string }> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/dataset/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const runPipeline = async (datasetId: string, targetColumn?: string): Promise<PipelineStatus> => {
  const response = await api.post('/pipeline/run', {
    dataset_id: datasetId,
    target_column: targetColumn,
  });
  return response.data;
};

export const getPipelineStatus = async (datasetId: string): Promise<PipelineStatus> => {
  const response = await api.get(`/pipeline/status/${datasetId}`);
  return response.data;
};

export const chatWithAnalyst = async (datasetId: string, question: string): Promise<ChatResponse> => {
  const response = await api.post('/analyst/chat', {
    dataset_id: datasetId,
    question: question,
  });
  return response.data;
};
