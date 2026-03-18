import { useMutation } from '@tanstack/react-query'
import { api, AnalysisParams, AnalysisResponse } from '@/lib/api'

// Analyze data mutation hook
export function useAnalyzeData() {
  return useMutation({
    mutationFn: (params: AnalysisParams) => api.analyzeData(params),
    // Long timeout for analysis (5 minutes)
    mutationKey: ['analyzeData'],
  })
}
