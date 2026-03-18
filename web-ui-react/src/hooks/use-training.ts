import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  api,
  TrainingParams,
  TrainingResponse,
} from '@/lib/api'
import { queryKeys } from '@/lib/query-client'

// Submit training mutation
export function useSubmitTraining() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: TrainingParams) => api.submitTraining(params),
    onSuccess: (data: TrainingResponse) => {
      // Invalidate training jobs cache
      queryClient.invalidateQueries({ queryKey: queryKeys.trainingJobs })
      // Prefetch the new training status
      queryClient.prefetchQuery({
        queryKey: queryKeys.trainingStatus(data.task_id),
        queryFn: () => api.getTrainingStatus(data.task_id),
      })
    },
  })
}

// Training status hook
export function useTrainingStatus(taskId: string | null) {
  return useQuery({
    queryKey: queryKeys.trainingStatus(taskId || ''),
    queryFn: () => api.getTrainingStatus(taskId!),
    // Only fetch if we have a taskId
    enabled: !!taskId,
    // Poll every 5 seconds while running
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (status === 'running') return 5000
      if (status === 'pending') return 3000
      return false // Don't refetch when completed or failed
    },
  })
}

// Cancel training mutation
export function useCancelTraining() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (taskId: string) => {
      const response = await fetch(`/api/v1/train/cancel/${taskId}`, {
        method: 'POST',
      })
      if (!response.ok) throw new Error('Failed to cancel training')
      return response.json() as Promise<{ success: boolean }>
    },
    onSuccess: (_, taskId) => {
      // Invalidate the specific training status
      queryClient.invalidateQueries({ queryKey: queryKeys.trainingStatus(taskId) })
      queryClient.invalidateQueries({ queryKey: queryKeys.trainingJobs })
    },
  })
}
