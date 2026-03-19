'use client'

import { useQuery, useMutation } from '@tanstack/react-query'
import { apiService } from '@/services/api'

export const usePipelineStatus = (datasetId: string | null, enabled = true) => {
  return useQuery({
    queryKey: ['pipelineStatus', datasetId],
    queryFn: () => apiService.getPipelineStatus(datasetId!),
    enabled: !!datasetId && enabled,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (status === 'completed' || status === 'failed') return false
      if (query.state.error) return false
      return 2000
    },
    retry: false,
    staleTime: 0,
    throwOnError: false,   // never let query errors bubble up as exceptions
  })
}

export const useUploadDataset = () => {
  return useMutation({
    mutationFn: ({ file, onProgress }: { file: File; onProgress?: (p: number) => void }) =>
      apiService.uploadDataset(file, onProgress),
  })
}

export const useRunPipeline = () => {
  return useMutation({
    mutationFn: ({ datasetId, targetColumn }: { datasetId: string; targetColumn?: string }) =>
      apiService.runPipeline(datasetId, targetColumn),
  })
}

export const useChat = () => {
  return useMutation({
    mutationFn: ({ datasetId, question }: { datasetId: string; question: string }) =>
      apiService.chat(datasetId, question),
  })
}
