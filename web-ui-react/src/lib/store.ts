import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type NavPage = 'dashboard' | 'discovery' | 'training' | 'models' | 'analysis' | 'labeling'

interface AppState {
  // Navigation
  currentPage: NavPage
  setCurrentPage: (page: NavPage) => void

  // Sidebar
  sidebarCollapsed: boolean
  toggleSidebar: () => void
  setSidebarCollapsed: (collapsed: boolean) => void

  // Theme
  theme: 'light' | 'dark' | 'system'
  setTheme: (theme: 'light' | 'dark' | 'system') => void

  // API Configuration
  apiBaseUrl: string
  setApiBaseUrl: (url: string) => void

  // Training Jobs
  activeJobId: string | null
  setActiveJobId: (jobId: string | null) => void

  // UI State
  isLoading: boolean
  setIsLoading: (loading: boolean) => void

  // Notifications
  notifications: Notification[]
  addNotification: (notification: Omit<Notification, 'id'>) => void
  removeNotification: (id: string) => void
  clearNotifications: () => void
}

interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
  duration?: number
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Navigation
      currentPage: 'dashboard',
      setCurrentPage: (page) => set({ currentPage: page }),

      // Sidebar
      sidebarCollapsed: false,
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),

      // Theme
      theme: 'dark',
      setTheme: (theme) => set({ theme }),

      // API Configuration
      apiBaseUrl: 'http://localhost:8000',
      setApiBaseUrl: (url) => set({ apiBaseUrl: url }),

      // Training Jobs
      activeJobId: null,
      setActiveJobId: (jobId) => set({ activeJobId: jobId }),

      // UI State
      isLoading: false,
      setIsLoading: (loading) => set({ isLoading: loading }),

      // Notifications
      notifications: [],
      addNotification: (notification) =>
        set((state) => ({
          notifications: [
            ...state.notifications,
            { ...notification, id: Math.random().toString(36).substring(2, 9) },
          ],
        })),
      removeNotification: (id) =>
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
        })),
      clearNotifications: () => set({ notifications: [] }),
    }),
    {
      name: 'yolo-auto-training-storage',
      partialize: (state) => ({
        theme: state.theme,
        sidebarCollapsed: state.sidebarCollapsed,
        apiBaseUrl: state.apiBaseUrl,
      }),
    }
  )
)

// Training job store
interface TrainingJob {
  id: string
  model: string
  dataset: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  startedAt: string
  completedAt?: string
  metrics?: Record<string, number>
}

interface TrainingState {
  jobs: TrainingJob[]
  addJob: (job: TrainingJob) => void
  updateJob: (id: string, updates: Partial<TrainingJob>) => void
  removeJob: (id: string) => void
  getJob: (id: string) => TrainingJob | undefined
}

export const useTrainingStore = create<TrainingState>()((set, get) => ({
  jobs: [],
  addJob: (job) => set((state) => ({ jobs: [job, ...state.jobs] })),
  updateJob: (id, updates) =>
    set((state) => ({
      jobs: state.jobs.map((job) =>
        job.id === id ? { ...job, ...updates } : job
      ),
    })),
  removeJob: (id) =>
    set((state) => ({
      jobs: state.jobs.filter((job) => job.id !== id),
    })),
  getJob: (id) => get().jobs.find((job) => job.id === id),
}))
