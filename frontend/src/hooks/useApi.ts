'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiService } from '@/services/api'

const WS_BASE = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
  .replace(/^http/, 'ws')

// ── WebSocket hook with polling fallback ─────────────────────────────────────
export const usePipelineWebSocket = (datasetId: string | null, enabled = true) => {
  const queryClient = useQueryClient()
  const wsRef = useRef<WebSocket | null>(null)
  const [wsConnected, setWsConnected] = useState(false)
  const pingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const updateCache = useCallback(
    (data: any) => {
      queryClient.setQueryData(['pipelineStatus', datasetId], (prev: any) => ({
        ...(prev ?? {}),
        ...data,
      }))
    },
    [queryClient, datasetId]
  )

  useEffect(() => {
    if (!datasetId || !enabled) return

    let dead = false

    const connect = () => {
      if (dead) return
      const ws = new WebSocket(`${WS_BASE}/api/v1/pipeline/ws/${datasetId}`)
      wsRef.current = ws

      ws.onopen = () => {
        setWsConnected(true)
        pingRef.current = setInterval(() => ws.readyState === WebSocket.OPEN && ws.send('ping'), 20_000)
      }

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data)
          if (data !== 'pong') updateCache(data)
        } catch {}
      }

      ws.onclose = (e) => {
        setWsConnected(false)
        if (pingRef.current) clearInterval(pingRef.current)
        // Don't reconnect on clean close (1000 = normal, 1001 = going away)
        // This prevents the spam loop when the backend closes the socket after
        // the pipeline completes or when there's no active job.
        const cleanClose = e.code === 1000 || e.code === 1001
        if (!dead && !cleanClose) setTimeout(connect, 3000)
      }

      ws.onerror = () => { if (!dead) ws.close() }
    }

    connect()

    return () => {
      dead = true
      if (pingRef.current) clearInterval(pingRef.current)
      wsRef.current?.close()
    }
  }, [datasetId, enabled, updateCache])

  return wsConnected
}

// ── Polling fallback (used when WS not connected) ────────────────────────────
export const usePipelineStatus = (datasetId: string | null, enabled = true) => {
  const wsConnected = usePipelineWebSocket(datasetId, enabled)

  return useQuery({
    queryKey: ['pipelineStatus', datasetId],
    queryFn: () => apiService.getPipelineStatus(datasetId!),
    enabled: !!datasetId && enabled && !wsConnected,
    staleTime: (query) => {
      const status = (query.state.data as any)?.status
      return status === 'completed' || status === 'failed' ? Infinity : 0
    },
    refetchInterval: (query) => {
      if (wsConnected) return false
      const status = (query.state.data as any)?.status
      if (status === 'completed' || status === 'failed') return false
      if (query.state.error) return false
      return status === 'queued' ? 1500 : 3000
    },
    retry: false,
    throwOnError: false,
  })
}

export const useUploadDataset = () => {
  return useMutation({
    mutationFn: ({ file, onProgress }: { file: File; onProgress?: (p: number) => void }) =>
      apiService.uploadDataset(file, onProgress),
  })
}

export const useRunPipeline = () => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ datasetId, targetColumn }: { datasetId: string; targetColumn?: string }) =>
      apiService.runPipeline(datasetId, targetColumn),
    onSuccess: (data, { datasetId }) => {
      if (data?.status === 'completed') {
        queryClient.setQueryData(['pipelineStatus', datasetId], data)
      }
    },
  })
}

export const useChat = () => {
  return useMutation({
    mutationFn: ({ datasetId, question }: { datasetId: string; question: string }) =>
      apiService.chat(datasetId, question),
  })
}
