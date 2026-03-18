const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
  body?: unknown
  timeout?: number
}

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, timeout = 30000 } = options

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)

  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })

    clearTimeout(timeoutId)

    if (!response.ok) {
      const errorData = await response.json().catch(() => null)
      throw new ApiError(
        errorData?.message || `Request failed with status ${response.status}`,
        response.status,
        errorData
      )
    }

    return response.json()
  } catch (error) {
    clearTimeout(timeoutId)

    if (error instanceof ApiError) {
      throw error
    }

    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError('Request timeout', 408)
    }

    throw new ApiError('Network error', 0)
  }
}

export const api = {
  // Health check
  getHealth: () => request<HealthResponse>('/health'),

  // Data Discovery
  searchDatasets: (params: SearchDatasetsParams) =>
    request<SearchDatasetsResponse>('/api/v1/data/search', {
      method: 'POST',
      body: params,
    }),

  // Training
  submitTraining: (params: TrainingParams) =>
    request<TrainingResponse>('/api/v1/train/submit', {
      method: 'POST',
      body: params,
    }),
  getTrainingStatus: (taskId: string) =>
    request<TrainingStatusResponse>(`/api/v1/train/status/${taskId}`),

  // Models
  getModels: () => request<Model[]>('/api/v1/models'),
  exportModel: (params: ExportParams) =>
    request<ExportResponse>('/api/v1/deploy/export', {
      method: 'POST',
      body: params,
    }),

  // Auto Label
  submitLabeling: (params: LabelingParams) =>
    request<LabelingResponse>('/api/v1/label/submit', {
      method: 'POST',
      body: params,
    }),

  // Analysis
  analyzeData: (params: AnalysisParams) =>
    request<AnalysisResponse>('/api/v1/analysis/analyze', {
      method: 'POST',
      body: params,
      timeout: 300000,
    }),
}

// Type definitions
export interface HealthResponse {
  status: string
  training_api?: string
  redis?: string
}

export interface SearchDatasetsParams {
  query: string
  max_results?: number
  sources?: string[]
  min_images?: number
}

export interface SearchDatasetsResponse {
  datasets: Dataset[]
}

export interface Dataset {
  name: string
  source: string
  images: number
  license: string
  relevance_score: number
  url?: string
}

export interface TrainingParams {
  model: string
  data_yaml: string
  epochs: number
  imgsz?: number
}

export interface TrainingResponse {
  task_id: string
  estimated_time_minutes: number
}

export interface TrainingStatusResponse {
  task_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress?: number
  metrics?: Record<string, number>
}

export interface Model {
  id: string
  name: string
  path: string
  created_at: string
  metrics?: Record<string, number>
}

export interface ExportParams {
  model_path: string
  platform: string
  imgsz?: number
}

export interface ExportResponse {
  task_id: string
}

export interface LabelingParams {
  task_id: string
  input_folder: string
  classes: string[]
  base_model: string
  conf_threshold?: number
}

export interface LabelingResponse {
  task_id: string
}

export interface AnalysisParams {
  dataset_path: string
  analysis_type: 'quality' | 'distribution' | 'anomalies' | 'full'
  prompt?: string
}

export interface AnalysisResponse {
  status: 'completed' | 'failed'
  content?: string
  files?: { name: string; url: string }[]
  error?: string
}

export { ApiError }
