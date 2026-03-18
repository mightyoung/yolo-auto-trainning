'use client'

import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Rocket,
  Search,
  Sparkles,
  BarChart3,
  Package,
  Activity,
  CheckCircle2,
  XCircle,
  Clock,
  TrendingUp,
  Cpu,
  Zap,
  Database,
  Gauge,
  Brain,
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { HoverCard } from '@/components/ui/hover-card'
import { Skeleton } from '@/components/ui/skeleton'
import { TrainingMetricsChart, SystemMetricsChart, ModelComparisonChart } from '@/components/charts'
import { useAppStore, NavPage } from '@/lib/store'
import { cn } from '@/lib/utils'

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08 },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 20, scale: 0.95 },
  show: { opacity: 1, y: 0, scale: 1 },
}

// Mock data
const mockSystemStatus = {
  trainingApi: { status: 'available', label: 'Available', uptime: '99.9%' },
  redis: { status: 'connected', label: 'Connected', uptime: '99.8%' },
  gpu: { status: 'available', label: '2/4 GPUs', usage: 45 },
}

const mockRecentJobs = [
  { id: '1', name: 'coco-detection-v2', model: 'yolo11m', status: 'running', progress: 65, epoch: 65, total: 100 },
  { id: '2', name: 'traffic-sign-v1', model: 'yolo11s', status: 'completed', progress: 100, mAP: 0.72 },
  { id: '3', name: 'person-detection', model: 'yolo11n', status: 'failed', progress: 0, error: 'Out of memory' },
]

const quickActions = [
  { title: 'Start Training', icon: <Rocket className="w-6 h-6" />, page: 'training' as NavPage, gradient: 'from-cyan-500 to-cyber-blue-500', shadow: 'shadow-cyan-500/25' },
  { title: 'Discover Data', icon: <Search className="w-6 h-6" />, page: 'discovery' as NavPage, gradient: 'from-purple-500 to-pink-500', shadow: 'shadow-purple-500/25' },
  { title: 'Auto Label', icon: <Sparkles className="w-6 h-6" />, page: 'labeling' as NavPage, gradient: 'from-amber-500 to-orange-500', shadow: 'shadow-amber-500/25' },
  { title: 'Analyze Data', icon: <BarChart3 className="w-6 h-6" />, page: 'analysis' as NavPage, gradient: 'from-green-500 to-emerald-500', shadow: 'shadow-green-500/25' },
]

const mockStats = {
  modelsTrained: 12,
  datasetsUsed: 8,
  trainingHours: 156,
  accuracy: 94.5,
}

// Chart data
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const mockTrainingData = [
  { epoch: 1, loss: 2.845, accuracy: 0.152, val_loss: 2.654 },
  { epoch: 10, loss: 1.923, accuracy: 0.312, val_loss: 1.856 },
  { epoch: 20, loss: 1.245, accuracy: 0.534, val_loss: 1.312 },
  { epoch: 30, loss: 0.876, accuracy: 0.678, val_loss: 0.945 },
  { epoch: 40, loss: 0.623, accuracy: 0.782, val_loss: 0.712 },
  { epoch: 50, loss: 0.478, accuracy: 0.845, val_loss: 0.534 },
  { epoch: 60, loss: 0.389, accuracy: 0.889, val_loss: 0.445 },
  { epoch: 70, loss: 0.312, accuracy: 0.923, val_loss: 0.378 },
  { epoch: 80, loss: 0.267, accuracy: 0.945, val_loss: 0.334 },
  { epoch: 90, loss: 0.234, accuracy: 0.961, val_loss: 0.298 },
  { epoch: 100, loss: 0.198, accuracy: 0.972, val_loss: 0.256 },
]

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const mockSystemMetrics = [
  { time: '00:00', gpu: 45, cpu: 32, memory: 58 },
  { time: '00:05', gpu: 62, cpu: 41, memory: 62 },
  { time: '00:10', gpu: 78, cpu: 55, memory: 65 },
  { time: '00:15', gpu: 85, cpu: 48, memory: 71 },
  { time: '00:20', gpu: 92, cpu: 62, memory: 68 },
  { time: '00:25', gpu: 88, cpu: 58, memory: 75 },
  { time: '00:30', gpu: 72, cpu: 45, memory: 72 },
  { time: '00:35', gpu: 65, cpu: 52, memory: 68 },
  { time: '00:40', gpu: 55, cpu: 38, memory: 65 },
  { time: '00:45', gpu: 42, cpu: 35, memory: 62 },
]

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const mockModelComparison = [
  { name: 'yolo11n', mAP: 0.38, precision: 0.72, recall: 0.65, size: 6 },
  { name: 'yolo11s', mAP: 0.48, precision: 0.81, recall: 0.74, size: 21 },
  { name: 'yolo11m', mAP: 0.52, precision: 0.85, recall: 0.78, size: 52 },
  { name: 'yolo11l', mAP: 0.55, precision: 0.87, recall: 0.82, size: 86 },
]

export default function DashboardPage() {
  const { setCurrentPage } = useAppStore()
  const [mounted, setMounted] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    setMounted(true)
    // Simulate data loading
    const timer = setTimeout(() => setIsLoading(false), 800)
    return () => clearTimeout(timer)
  }, [])

  // Show loading skeleton during SSR and initial load
  if (!mounted || isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-end justify-between">
          <div>
            <div className="h-10 w-48 bg-muted rounded animate-pulse mb-2" />
            <div className="h-6 w-72 bg-muted rounded animate-pulse" />
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[1,2,3,4].map(i => (
            <div key={i} className="h-32 bg-muted rounded-xl animate-pulse" />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 h-64 bg-muted rounded-xl animate-pulse" />
          <div className="space-y-4">
            <div className="h-24 bg-muted rounded-xl animate-pulse" />
            <div className="h-24 bg-muted rounded-xl animate-pulse" />
            <div className="h-24 bg-muted rounded-xl animate-pulse" />
          </div>
        </div>
      </div>
    )
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Page Header */}
      <motion.div variants={itemVariants} className="flex items-end justify-between">
        <div>
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text">
            Dashboard
          </h1>
          <p className="text-muted-foreground mt-2 text-lg">
            Monitor your ML training pipeline
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="success" className="animate-pulse-glow">
            <span className="w-2 h-2 rounded-full bg-current mr-2" />
            System Online
          </Badge>
        </div>
      </motion.div>

      {/* Bento Grid Layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Status Cards */}
        <motion.div variants={itemVariants} className="col-span-1">
          <HoverCard hoverEffect="glow" className="h-full">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="p-3 rounded-xl bg-gradient-to-br from-cyan-500/20 to-cyber-blue-500/20">
                  <Cpu className="w-6 h-6 text-cyan-400" />
                </div>
                <Badge variant="success">Online</Badge>
              </div>
              <div className="mt-4">
                <p className="text-sm text-muted-foreground">Training API</p>
                <p className="text-2xl font-bold font-mono">{mockSystemStatus.trainingApi.uptime}</p>
              </div>
            </CardContent>
          </HoverCard>
        </motion.div>

        <motion.div variants={itemVariants} className="col-span-1">
          <HoverCard hoverEffect="glow" className="h-full">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="p-3 rounded-xl bg-gradient-to-br from-green-500/20 to-emerald-500/20">
                  <Database className="w-6 h-6 text-green-400" />
                </div>
                <Badge variant="success">Connected</Badge>
              </div>
              <div className="mt-4">
                <p className="text-sm text-muted-foreground">Database</p>
                <p className="text-2xl font-bold font-mono">{mockSystemStatus.redis.uptime}</p>
              </div>
            </CardContent>
          </HoverCard>
        </motion.div>

        <motion.div variants={itemVariants} className="col-span-1">
          <HoverCard hoverEffect="glow" className="h-full">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="p-3 rounded-xl bg-gradient-to-br from-purple-500/20 to-pink-500/20">
                  <Gauge className="w-6 h-6 text-purple-400" />
                </div>
                <Badge variant="info">{mockSystemStatus.gpu.usage}%</Badge>
              </div>
              <div className="mt-4">
                <p className="text-sm text-muted-foreground">GPU Usage</p>
                <Progress value={mockSystemStatus.gpu.usage} className="h-2 mt-2" />
              </div>
            </CardContent>
          </HoverCard>
        </motion.div>

        <motion.div variants={itemVariants} className="col-span-1">
          <HoverCard hoverEffect="glow" className="h-full">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="p-3 rounded-xl bg-gradient-to-br from-amber-500/20 to-orange-500/20">
                  <Activity className="w-6 h-6 text-amber-400" />
                </div>
                <Badge variant="warning">1 Active</Badge>
              </div>
              <div className="mt-4">
                <p className="text-sm text-muted-foreground">Active Jobs</p>
                <p className="text-2xl font-bold font-mono">1 / 5</p>
              </div>
            </CardContent>
          </HoverCard>
        </motion.div>
      </div>

      {/* Quick Actions - Full Width */}
      <motion.div variants={itemVariants}>
        <Card className="border-none bg-transparent shadow-none">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {quickActions.map((action, index) => (
                <motion.div
                  key={action.title}
                  whileHover={{ scale: 1.03, y: -2 }}
                  whileTap={{ scale: 0.98 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 17 }}
                >
                  <Button
                    variant="outline"
                    className={cn(
                      'w-full h-24 flex flex-col items-center justify-center gap-3',
                      'border-2 bg-card/50 backdrop-blur-sm',
                      'hover:bg-card/80 transition-all duration-300'
                    )}
                    onClick={() => setCurrentPage(action.page)}
                    style={{
                      borderColor: index === 0 ? '#06d6a0' : index === 1 ? '#a855f7' : index === 2 ? '#f59e0b' : '#22c55e',
                    }}
                  >
                    <div className={cn(
                      'p-3 rounded-xl',
                      `bg-gradient-to-br ${action.gradient} ${action.shadow}`
                    )}>
                      {React.cloneElement(action.icon as React.ReactElement, { className: "w-6 h-6 text-black" })}
                    </div>
                    <span className="text-sm font-medium">{action.title}</span>
                  </Button>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Training Metrics Chart */}
      <motion.div variants={itemVariants}>
        <HoverCard hoverEffect="lift" className="h-full">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-cyan-400" />
                  Training Metrics
                </CardTitle>
                <CardDescription>Loss and accuracy over epochs</CardDescription>
              </div>
              <Badge variant="outline" className="gap-1">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                Live
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <TrainingMetricsChart data={mockTrainingData} showAccuracy />
          </CardContent>
        </HoverCard>
      </motion.div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Metrics */}
        <motion.div variants={itemVariants}>
          <HoverCard hoverEffect="lift" className="h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-purple-400" />
                System Resources
              </CardTitle>
              <CardDescription>GPU, CPU, and memory usage over time</CardDescription>
            </CardHeader>
            <CardContent>
              <SystemMetricsChart data={mockSystemMetrics} />
            </CardContent>
          </HoverCard>
        </motion.div>

        {/* Model Comparison */}
        <motion.div variants={itemVariants}>
          <HoverCard hoverEffect="lift" className="h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-amber-400" />
                Model Performance
              </CardTitle>
              <CardDescription>Compare different YOLO model variants</CardDescription>
            </CardHeader>
            <CardContent>
              <ModelComparisonChart data={mockModelComparison} />
            </CardContent>
          </HoverCard>
        </motion.div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Jobs - 2/3 width */}
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <HoverCard hoverEffect="lift" className="h-full">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Rocket className="w-5 h-5 text-cyan-400" />
                    Training Jobs
                  </CardTitle>
                  <CardDescription>Recent model training tasks</CardDescription>
                </div>
                <Button variant="ghost" size="sm">View All</Button>
              </div>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="flex items-center gap-4">
                      <Skeleton className="h-12 w-12 rounded-xl" />
                      <div className="flex-1 space-y-2">
                        <Skeleton className="h-4 w-3/4" />
                        <Skeleton className="h-3 w-1/2" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-3">
                  {mockRecentJobs.map((job) => (
                    <JobCard key={job.id} job={job} />
                  ))}
                </div>
              )}
            </CardContent>
          </HoverCard>
        </motion.div>

        {/* Stats - 1/3 width */}
        <motion.div variants={itemVariants} className="space-y-4">
          <HoverCard hoverEffect="glow">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-cyan-500/20">
                  <Brain className="w-5 h-5 text-cyan-400" />
                </div>
                <div>
                  <p className="font-semibold">Total Models</p>
                  <p className="text-sm text-muted-foreground">All time</p>
                </div>
              </div>
              <p className="text-4xl font-bold gradient-text">{mockStats.modelsTrained}</p>
            </CardContent>
          </HoverCard>

          <HoverCard hoverEffect="glow">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-purple-500/20">
                  <TrendingUp className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <p className="font-semibold">Avg. Accuracy</p>
                  <p className="text-sm text-muted-foreground">Last 30 days</p>
                </div>
              </div>
              <p className="text-4xl font-bold gradient-text">{mockStats.accuracy}%</p>
            </CardContent>
          </HoverCard>

          <HoverCard hoverEffect="glow">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-green-500/20">
                  <Zap className="w-5 h-5 text-green-400" />
                </div>
                <div>
                  <p className="font-semibold">Training Hours</p>
                  <p className="text-sm text-muted-foreground">This month</p>
                </div>
              </div>
              <p className="text-4xl font-bold gradient-text">{mockStats.trainingHours}h</p>
            </CardContent>
          </HoverCard>
        </motion.div>
      </div>
    </motion.div>
  )
}

// Job Card Component with enhanced styling
function JobCard({ job }: { job: { id: string; name: string; model: string; status: string; progress: number; epoch?: number; total?: number; mAP?: number; error?: string } }) {
  const statusConfig = {
    running: { color: 'cyan', label: 'Running', icon: <Activity className="w-3 h-3" />, bg: 'bg-cyan-500/20', text: 'text-cyan-400' },
    completed: { color: 'green', label: 'Completed', icon: <CheckCircle2 className="w-3 h-3" />, bg: 'bg-green-500/20', text: 'text-green-400' },
    failed: { color: 'red', label: 'Failed', icon: <XCircle className="w-3 h-3" />, bg: 'bg-red-500/20', text: 'text-red-400' },
    pending: { color: 'amber', label: 'Pending', icon: <Clock className="w-3 h-3" />, bg: 'bg-amber-500/20', text: 'text-amber-400' },
  }

  const config = statusConfig[job.status as keyof typeof statusConfig] || statusConfig.pending

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className="group flex items-center justify-between p-4 rounded-xl bg-muted/30 hover:bg-muted/50 transition-all duration-300 border border-transparent hover:border-primary/10"
    >
      <div className="flex items-center gap-4">
        <motion.div
          className={cn('p-3 rounded-xl', config.bg)}
          whileHover={{ scale: 1.1 }}
        >
          <Rocket className={cn('w-5 h-5', config.text)} />
        </motion.div>
        <div>
          <p className="font-medium group-hover:text-cyan-400 transition-colors">{job.name}</p>
          <div className="flex items-center gap-2 mt-1">
            <Badge variant="outline" className="text-xs">{job.model}</Badge>
            {job.epoch && (
              <span className="text-xs text-muted-foreground">
                Epoch {job.epoch}/{job.total}
              </span>
            )}
            {job.mAP && (
              <span className="text-xs text-muted-foreground">
                mAP: {job.mAP}
              </span>
            )}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        {job.status === 'running' && (
          <div className="w-40">
            <Progress value={job.progress} className="h-2" />
            <p className="text-xs text-muted-foreground mt-1 text-right">{job.progress}%</p>
          </div>
        )}
        {job.error && (
          <p className="text-xs text-red-400 max-w-[100px] truncate">{job.error}</p>
        )}
        <Badge
          variant={config.color === 'cyan' ? 'success' : config.color === 'green' ? 'success' : config.color === 'red' ? 'destructive' : 'warning'}
          className="gap-1"
        >
          {config.icon}
          <span>{config.label}</span>
        </Badge>
      </div>
    </motion.div>
  )
}
