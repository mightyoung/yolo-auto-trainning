import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, ExportParams } from '@/lib/api'
import { queryKeys } from '@/lib/query-client'

// Get all models hook
export function useModels() {
  return useQuery({
    queryKey: queryKeys.models,
    queryFn: () => api.getModels(),
    // Refetch every 30 seconds
    refetchInterval: 30000,
  })
}

// Export model mutation
export function useExportModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (params: ExportParams) => api.exportModel(params),
    onSuccess: () => {
      // Invalidate models cache after export
      queryClient.invalidateQueries({ queryKey: queryKeys.models })
    },
  })
}

// Delete model mutation
export function useDeleteModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (modelId: string) => {
      const response = await fetch(`/api/v1/models/${modelId}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete model')
      return response.json() as Promise<{ success: boolean }>
    },
    onSuccess: () => {
      // Invalidate models cache after delete
      queryClient.invalidateQueries({ queryKey: queryKeys.models })
    },
  })
}
