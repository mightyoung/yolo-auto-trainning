'use client'

import React from 'react'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { cn } from '@/lib/utils'

// Custom tooltip component for dark theme
function CustomTooltip({ active, payload, label }: {
  active?: boolean
  payload?: Array<{ value: number; name: string; color: string }>
  label?: string
}) {
  if (active && payload && payload.length) {
    return (
      <div className="bg-card/95 backdrop-blur-xl border border-border/50 rounded-lg p-3 shadow-lg">
        <p className="text-xs text-muted-foreground mb-2">{label}</p>
        {payload.map((entry, index) => (
          <p key={index} className="text-sm font-medium" style={{ color: entry.color }}>
            {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
          </p>
        ))}
      </div>
    )
  }
  return null
}

// Training Metrics Line Chart
interface TrainingChartProps {
  data: Array<{ epoch: number; loss: number; accuracy: number; val_loss?: number }>
  className?: string
  showAccuracy?: boolean
}

export function TrainingMetricsChart({ data, className, showAccuracy = true }: TrainingChartProps) {
  return (
    <div className={cn("w-full h-[300px] overflow-hidden", className)}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#06d6a0" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#06d6a0" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.3)" />
          <XAxis
            dataKey="epoch"
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => value.toFixed(2)}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ paddingTop: '10px' }}
            formatter={(value) => <span className="text-xs text-muted-foreground">{value}</span>}
          />
          <Line
            type="monotone"
            dataKey="loss"
            name="Loss"
            stroke="#06d6a0"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6, fill: '#06d6a0' }}
          />
          {showAccuracy && (
            <Line
              type="monotone"
              dataKey="accuracy"
              name="Accuracy"
              stroke="#a855f7"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6, fill: '#a855f7' }}
            />
          )}
          {data[0]?.val_loss !== undefined && (
            <Line
              type="monotone"
              dataKey="val_loss"
              name="Val Loss"
              stroke="#f59e0b"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// System Usage Area Chart
interface SystemMetricsChartProps {
  data: Array<{ time: string; gpu: number; cpu: number; memory: number }>
  className?: string
}

export function SystemMetricsChart({ data, className }: SystemMetricsChartProps) {
  return (
    <div className={cn("w-full h-[250px] overflow-hidden", className)}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="colorGpu" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#06d6a0" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#06d6a0" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorCpu" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#00b4d8" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#00b4d8" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorMem" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#a855f7" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.3)" />
          <XAxis
            dataKey="time"
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ paddingTop: '10px' }}
            formatter={(value) => <span className="text-xs text-muted-foreground">{value}</span>}
          />
          <Area
            type="monotone"
            dataKey="gpu"
            name="GPU"
            stroke="#06d6a0"
            fillOpacity={1}
            fill="url(#colorGpu)"
            strokeWidth={2}
          />
          <Area
            type="monotone"
            dataKey="cpu"
            name="CPU"
            stroke="#00b4d8"
            fillOpacity={1}
            fill="url(#colorCpu)"
            strokeWidth={2}
          />
          <Area
            type="monotone"
            dataKey="memory"
            name="Memory"
            stroke="#a855f7"
            fillOpacity={1}
            fill="url(#colorMem)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

// Model Comparison Bar Chart
interface ModelComparisonChartProps {
  data: Array<{ name: string; mAP: number; precision: number; recall: number; size: number }>
  className?: string
}

export function ModelComparisonChart({ data, className }: ModelComparisonChartProps) {
  return (
    <div className={cn("w-full h-[300px] overflow-hidden", className)}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.3)" />
          <XAxis
            dataKey="name"
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => value.toFixed(2)}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ paddingTop: '10px' }}
            formatter={(value) => <span className="text-xs text-muted-foreground">{value}</span>}
          />
          <Bar dataKey="mAP" name="mAP" fill="#06d6a0" radius={[4, 4, 0, 0]} />
          <Bar dataKey="precision" name="Precision" fill="#00b4d8" radius={[4, 4, 0, 0]} />
          <Bar dataKey="recall" name="Recall" fill="#a855f7" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// Job Progress Chart
interface JobProgressChartProps {
  data: Array<{ name: string; progress: number; status: string }>
  className?: string
}

export function JobProgressChart({ data, className }: JobProgressChartProps) {
  return (
    <div className={cn("w-full h-[200px] overflow-hidden", className)}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.3)" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, 100]}
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `${value}%`}
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            width={100}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar
            dataKey="progress"
            name="Progress"
            radius={[0, 4, 4, 0]}
            fill="#06d6a0"
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
