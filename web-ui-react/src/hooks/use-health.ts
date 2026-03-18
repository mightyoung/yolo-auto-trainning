import { useQuery } from '@tanstack/react-query'
import { api, HealthResponse } from '@/lib/api'
import { queryKeys } from '@/lib/query-client'

// Health check hook
export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => api.getHealth() as Promise<HealthResponse>,
    // Refetch every 30 seconds to keep health status up to date
    refetchInterval: 30000,
    // Don't refetch on window focus for health check
    refetchOnWindowFocus: false,
  })
}

// System status derived from health
export function useSystemStatus() {
  const { data: health, ...rest } = useHealth()

  return {
    ...rest,
    data: health
      ? {
          isOnline: health.status === 'ok',
          trainingApiAvailable: health.training_api === 'connected',
          redisAvailable: health.redis === 'connected',
        }
      : undefined,
  }
}
