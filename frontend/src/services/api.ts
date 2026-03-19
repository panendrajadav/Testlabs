import axios, { AxiosInstance } from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

class ApiService {
  private api: AxiosInstance

  constructor() {
    this.api = axios.create({ baseURL: `${API_BASE_URL}/api/v1` })
  }

  async uploadDataset(file: File, onProgress?: (progress: number) => void) {
    const formData = new FormData()
    formData.append('file', file)
    const response = await this.api.post('/dataset/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (e.total) onProgress?.(Math.round((e.loaded / e.total) * 100))
      },
    })
    return response.data
  }

  async runPipeline(datasetId: string, targetColumn?: string) {
    const response = await this.api.post('/pipeline/run', {
      dataset_id: datasetId,
      target_column: targetColumn ?? null,
    })
    return response.data
  }

  async getPipelineStatus(datasetId: string) {
    const response = await this.api.get(`/pipeline/status/${datasetId}`)
    return response.data
  }

  async chat(datasetId: string, question: string) {
    const response = await this.api.post('/analyst/chat', { dataset_id: datasetId, question })
    return response.data
  }
}

export const apiService = new ApiService()
