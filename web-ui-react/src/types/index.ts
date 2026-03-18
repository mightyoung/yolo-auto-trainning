// Re-export API types for centralized type management
export type {
  HealthResponse,
  SearchDatasetsParams,
  SearchDatasetsResponse,
  Dataset,
  TrainingParams,
  TrainingResponse,
  TrainingStatusResponse,
  Model,
  ExportParams,
  ExportResponse,
  LabelingParams,
  LabelingResponse,
  AnalysisParams,
  AnalysisResponse,
} from '@/lib/api'

// Additional application types

// Navigation
export type NavPage =
  | 'dashboard'
  | 'discovery'
  | 'training'
  | 'models'
  | 'analysis'
  | 'labeling'

// Training job status
export type TrainingStatus = 'pending' | 'running' | 'completed' | 'failed'

// Model export platforms
export type ExportPlatform = 'onnx' | 'tensorrt' | 'tflite' | 'coreml'

// Chart data types
export interface ChartDataPoint {
  x: number | string
  y: number
}

export interface TimeSeriesDataPoint {
  timestamp: string
  value: number
}

// Metrics
export interface SystemMetrics {
  gpu: number
  cpu: number
  memory: number
  disk: number
}

export interface TrainingMetrics {
  epoch: number
  loss: number
  accuracy: number
  val_loss?: number
  val_accuracy?: number
}

export interface ModelMetrics {
  map: number
  precision: number
  recall: number
  f1: number
}

// UI Types
export interface PaginationParams {
  page: number
  limit: number
}

export interface PaginatedResponse<T> {
  data: T[]
  total: number
  page: number
  limit: number
  totalPages: number
}

// Form types
export interface TrainingFormData {
  model: string
  dataset: string
  epochs: number
  imageSize: number
  batchSize: number
  learningRate: number
  optimizer: string
  augmentation: boolean
}

export interface DatasetSearchFilters {
  source?: string[]
  minImages?: number
  maxImages?: number
  license?: string[]
  task?: string[]
}

// Component props types
export interface CardProps {
  title: string
  description?: string
  icon?: React.ReactNode
  action?: React.ReactNode
  children: React.ReactNode
  className?: string
}

export interface TableColumn<T> {
  key: keyof T | string
  header: string
  render?: (item: T) => React.ReactNode
  sortable?: boolean
  width?: string
}

export interface TableProps<T> {
  data: T[]
  columns: TableColumn<T>[]
  loading?: boolean
  onRowClick?: (item: T) => void
  emptyMessage?: string
}

// Notification types
export type NotificationType = 'success' | 'error' | 'warning' | 'info'

export interface Notification {
  id: string
  type: NotificationType
  title: string
  message?: string
  duration?: number
}
