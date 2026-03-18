import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Stale time: 5 minutes - data is considered fresh for 5 minutes
      staleTime: 5 * 60 * 1000,
      // Cache time: 10 minutes - unused data is garbage collected after 10 minutes
      gcTime: 10 * 60 * 1000,
      // Retry: 3 times on failure
      retry: 3,
      // Retry delay: exponential backoff
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      // Refetch on window focus: false (avoid interrupting user)
      refetchOnWindowFocus: false,
    },
    mutations: {
      // Retry mutations once
      retry: 1,
    },
  },
})

// Query keys factory for consistent cache management
export const queryKeys = {
  // Health
  health: ['health'] as const,

  // Data Discovery
  datasets: (params?: Record<string, unknown>) => ['datasets', params] as const,
  datasetDetail: (id: string) => ['dataset', id] as const,

  // Training
  trainingStatus: (taskId: string) => ['training', 'status', taskId] as const,
  trainingJobs: ['training', 'jobs'] as const,

  // Models
  models: ['models'] as const,
  modelDetail: (id: string) => ['model', id] as const,

  // Metrics
  metrics: ['metrics'] as const,
  systemMetrics: (timeRange: string) => ['metrics', 'system', timeRange] as const,
} as const
