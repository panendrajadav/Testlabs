// State management could be extended here if needed
// Currently using localStorage and React hooks for state management

export interface AppState {
  currentDataset: any | null
  pipelineStatus: string
  uploadProgress: number
  chatHistory: any[]
}

export const InitialState: AppState = {
  currentDataset: null,
  pipelineStatus: 'idle',
  uploadProgress: 0,
  chatHistory: [],
}

