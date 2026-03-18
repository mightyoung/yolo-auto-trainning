import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, SearchDatasetsParams } from '@/lib/api'
import { queryKeys } from '@/lib/query-client'

// Search datasets hook
export function useSearchDatasets(params: SearchDatasetsParams) {
  return useQuery({
    queryKey: queryKeys.datasets(params as unknown as Record<string, unknown>),
    queryFn: () => api.searchDatasets(params),
    // Don't refetch automatically - user-initiated search
    refetchOnWindowFocus: false,
    // Keep previous data while fetching new data
    placeholderData: (previousData) => previousData,
  })
}

// Download dataset mutation hook
export function useDownloadDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (params: { datasetId: string; targetPath: string }) => {
      const response = await fetch('/api/v1/data/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      if (!response.ok) throw new Error('Failed to download dataset')
      return response.json() as Promise<{ task_id: string }>
    },
    onSuccess: () => {
      // Invalidate datasets cache after successful download
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
    },
  })
}
