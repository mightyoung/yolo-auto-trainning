import { useMutation } from '@tanstack/react-query'
import { api, LabelingParams, LabelingResponse } from '@/lib/api'

// Submit labeling mutation hook
export function useSubmitLabeling() {
  return useMutation({
    mutationFn: (params: LabelingParams) => api.submitLabeling(params),
    // Long timeout for labeling (5 minutes)
    mutationKey: ['submitLabeling'],
  })
}
