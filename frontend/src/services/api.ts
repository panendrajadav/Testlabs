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

  async getArtifactVersions(datasetId: string) {
    const response = await this.api.get(`/artifacts/${datasetId}/versions`)
    return response.data
  }

  async getArtifactDatasets() {
    const response = await this.api.get('/artifacts/datasets/all')
    return response.data
  }

  async getArtifactSummary(datasetId: string, version: string) {
    const response = await this.api.get(`/artifacts/${datasetId}/${version}/summary`)
    return response.data
  }

  async getArtifactMetadata(datasetId: string, version: string) {
    const response = await this.api.get(`/artifacts/${datasetId}/${version}/metadata`)
    return response.data
  }

  async getExperimentLog(datasetId: string, version: string) {
    const response = await this.api.get(`/artifacts/${datasetId}/${version}/experiment`)
    return response.data
  }

  async getTrainingLog(datasetId: string, version: string) {
    const response = await this.api.get(`/artifacts/${datasetId}/${version}/training-log`)
    return response.data
  }

  async getReproducibility(datasetId: string, version: string) {
    const response = await this.api.get(`/artifacts/${datasetId}/${version}/reproducibility`)
    return response.data
  }

  async getInferenceSamples(datasetId: string, version: string) {
    const response = await this.api.get(`/artifacts/${datasetId}/${version}/inference-samples`)
    return response.data
  }

  async getDriftHooks(datasetId: string, version: string) {
    const response = await this.api.get(`/artifacts/${datasetId}/${version}/drift-hooks`)
    return response.data
  }

  async getApiExport(datasetId: string, version: string) {
    const response = await this.api.get(`/artifacts/${datasetId}/${version}/api-export`)
    return response.data
  }

  async rollbackVersion(datasetId: string, version: string) {
    const response = await this.api.post(`/artifacts/${datasetId}/${version}/rollback`)
    return response.data
  }

  async deleteVersion(datasetId: string, version: string) {
    const response = await this.api.delete(`/artifacts/${datasetId}/${version}`)
    return response.data
  }

  async deleteDataset(datasetId: string) {
    const response = await this.api.delete(`/dataset/${datasetId}`)
    return response.data
  }

  async compareVersions(datasetId: string) {
    const response = await this.api.get(`/artifacts/${datasetId}/compare`)
    return response.data
  }

  getModelDownloadUrl(datasetId: string, version: string) {
    return `${API_BASE_URL}/api/v1/artifacts/${datasetId}/${version}/download`
  }
}

export const apiService = new ApiService()
