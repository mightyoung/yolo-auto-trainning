'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Rocket, Loader2, CheckCircle2, AlertCircle, Cpu, Zap, Clock, HardDrive, Gauge } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { HoverCard } from '@/components/ui/hover-card'
import { Progress } from '@/components/ui/progress'

const models = [
  { id: 'yolo11n', name: 'YOLO11n', params: '2.6M', fps: '100+', gpu: '2GB', description: 'Fastest, for edge devices', gradient: 'from-cyan-500 to-teal-500' },
  { id: 'yolo11s', name: 'YOLO11s', params: '9.7M', fps: '60+', gpu: '4GB', description: 'Balanced speed and accuracy', gradient: 'from-cyan-500 to-blue-500' },
  { id: 'yolo11m', name: 'YOLO11m', params: '25.9M', fps: '40+', gpu: '8GB', description: 'Recommended for most cases', gradient: 'from-blue-500 to-purple-500' },
  { id: 'yolo11l', name: 'YOLO11l', params: '51.5M', fps: '25+', gpu: '12GB', description: 'Higher accuracy', gradient: 'from-purple-500 to-pink-500' },
  { id: 'yolo11x', name: 'YOLO11x', params: '97.2M', fps: '15+', gpu: '16GB', description: 'Maximum accuracy', gradient: 'from-pink-500 to-rose-500' },
]

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

export default function TrainingPage() {
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [epochs, setEpochs] = useState(100)
  const [batchSize, setBatchSize] = useState(16)
  const [imageSize, setImageSize] = useState(640)
  const [datasetPath, setDatasetPath] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = async () => {
    if (!selectedModel || !datasetPath) return

    setIsSubmitting(true)
    await new Promise(resolve => setTimeout(resolve, 2000))
    setIsSubmitting(false)
    setSubmitted(true)
  }

  const selectedModelInfo = models.find(m => m.id === selectedModel)

  // Calculate estimated time
  const estimatedTime = selectedModelInfo ? Math.round(epochs * (selectedModelInfo.id === 'yolo11n' ? 1 : selectedModelInfo.id === 'yolo11s' ? 1.5 : selectedModelInfo.id === 'yolo11m' ? 2 : selectedModelInfo.id === 'yolo11l' ? 3 : 4)) : 0

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={itemVariants}>
        <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text">
          Training
        </h1>
        <p className="text-muted-foreground mt-2 text-lg">
          Configure and start model training
        </p>
      </motion.div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Model Selection - 2/3 */}
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <HoverCard hoverEffect="lift" className="h-full border-primary/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Rocket className="w-5 h-5 text-cyan-400" />
                Model Selection
              </CardTitle>
              <CardDescription>Choose a YOLO model for your task</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {models.map((model, index) => (
                  <motion.button
                    key={model.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    whileHover={{ scale: 1.02, y: -4 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setSelectedModel(model.id)}
                    className={`relative p-5 rounded-xl border-2 text-left transition-all duration-300 overflow-hidden ${
                      selectedModel === model.id
                        ? 'border-cyan-500 bg-cyan-500/10 shadow-lg shadow-cyan-500/20'
                        : 'border-border hover:border-primary/30 hover:bg-muted/50'
                    }`}
                  >
                    {/* Gradient background */}
                    <div className={`absolute inset-0 bg-gradient-to-br ${model.gradient} opacity-0 transition-opacity duration-300 ${selectedModel === model.id ? 'opacity-5' : ''}`} />

                    <div className="relative">
                      <div className="flex items-center justify-between mb-3">
                        <span className="font-bold text-lg">{model.name}</span>
                        {selectedModel === model.id && (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                          >
                            <CheckCircle2 className="w-5 h-5 text-cyan-400" />
                          </motion.div>
                        )}
                      </div>
                      <div className="space-y-2 text-sm">
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground flex items-center gap-1">
                            <Cpu className="w-3 h-3" /> Params
                          </span>
                          <span className="font-mono">{model.params}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground flex items-center gap-1">
                            <Zap className="w-3 h-3" /> FPS
                          </span>
                          <span className="font-mono">{model.fps}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground flex items-center gap-1">
                            <HardDrive className="w-3 h-3" /> GPU
                          </span>
                          <span className="font-mono">{model.gpu}</span>
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground mt-3 pt-3 border-t border-border/50">
                        {model.description}
                      </p>
                    </div>
                  </motion.button>
                ))}
              </div>
            </CardContent>
          </HoverCard>
        </motion.div>

        {/* Resource Estimation - 1/3 */}
        <motion.div variants={itemVariants}>
          <HoverCard hoverEffect="glow" className="h-full border-primary/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Gauge className="w-5 h-5 text-cyan-400" />
                Resource Estimation
              </CardTitle>
              <CardDescription>Based on your selection</CardDescription>
            </CardHeader>
            <CardContent>
              {selectedModelInfo ? (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-4"
                >
                  {/* GPU Memory */}
                  <div className="p-4 rounded-xl bg-muted/50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-muted-foreground">GPU Memory</span>
                      <Badge variant="info">{selectedModelInfo.gpu}</Badge>
                    </div>
                    <Progress value={parseInt(selectedModelInfo.gpu) / 16 * 100} className="h-2" />
                  </div>

                  {/* Training Time */}
                  <div className="p-4 rounded-xl bg-muted/50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-muted-foreground flex items-center gap-1">
                        <Clock className="w-3 h-3" /> Est. Time
                      </span>
                      <span className="font-mono font-bold">~{estimatedTime} min</span>
                    </div>
                    <Progress value={50} className="h-2" />
                  </div>

                  {/* Parameters */}
                  <div className="p-4 rounded-xl bg-muted/50">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Parameters</span>
                      <span className="font-mono">{selectedModelInfo.params}</span>
                    </div>
                  </div>

                  {/* Batch Size Impact */}
                  <div className="p-4 rounded-xl bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/20">
                    <p className="text-sm">
                      <span className="text-cyan-400 font-medium">Tip: </span>
                      Use batch size {batchSize >= 16 ? '16' : '8'} for optimal memory usage
                    </p>
                  </div>
                </motion.div>
              ) : (
                <div className="py-12 text-center">
                  <motion.div
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <AlertCircle className="w-12 h-12 mx-auto text-muted-foreground/30" />
                  </motion.div>
                  <p className="mt-4 text-muted-foreground">Select a model</p>
                </div>
              )}
            </CardContent>
          </HoverCard>
        </motion.div>
      </div>

      {/* Configuration */}
      <motion.div variants={itemVariants}>
        <HoverCard hoverEffect="lift" className="border-primary/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CogIcon className="w-5 h-5 text-cyan-400" />
              Training Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-8">
            <div className="grid gap-8 md:grid-cols-3">
              {/* Epochs */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label className="text-base">Epochs</Label>
                  <Badge variant="outline">{epochs}</Badge>
                </div>
                <Slider
                  value={[epochs]}
                  onValueChange={([value]) => setEpochs(value)}
                  min={1}
                  max={500}
                  step={1}
                  className="py-2"
                />
                <div className="flex gap-2">
                  {[10, 50, 100, 200].map(v => (
                    <Button
                      key={v}
                      variant={epochs === v ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setEpochs(v)}
                      className="flex-1"
                    >
                      {v}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Batch Size */}
              <div className="space-y-4">
                <Label className="text-base">Batch Size</Label>
                <Select value={String(batchSize)} onValueChange={(v) => setBatchSize(Number(v))}>
                  <SelectTrigger className="h-12">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {[8, 16, 32, 64].map(v => (
                      <SelectItem key={v} value={String(v)}>{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">Higher = faster training, more memory</p>
              </div>

              {/* Image Size */}
              <div className="space-y-4">
                <Label className="text-base">Image Size</Label>
                <Select value={String(imageSize)} onValueChange={(v) => setImageSize(Number(v))}>
                  <SelectTrigger className="h-12">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {[320, 416, 512, 640, 1280].map(v => (
                      <SelectItem key={v} value={String(v)}>{v}px</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">Higher = more detail, slower</p>
              </div>
            </div>

            {/* Dataset Path */}
            <div className="space-y-3">
              <Label className="text-base">Dataset YAML Path</Label>
              <Input
                placeholder="e.g., datasets/coco8/data.yaml"
                value={datasetPath}
                onChange={(e) => setDatasetPath(e.target.value)}
                className="h-12 bg-muted/50 border-primary/10 focus:border-primary/30"
              />
              <p className="text-sm text-muted-foreground">
                Path to your dataset configuration file (YOLO format)
              </p>
            </div>
          </CardContent>
        </HoverCard>
      </motion.div>

      {/* Submit */}
      <motion.div variants={itemVariants}>
        <HoverCard hoverEffect={submitted ? 'glow' : 'lift'} className="border-primary/10">
          <CardContent className="pt-6">
            <AnimatePresence mode="wait">
              {submitted ? (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="text-center py-8"
                >
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: 'spring', stiffness: 200 }}
                    className="w-20 h-20 mx-auto rounded-full bg-gradient-to-br from-cyan-500 to-cyber-blue-500 flex items-center justify-center mb-4"
                  >
                    <CheckCircle2 className="w-10 h-10 text-white" />
                  </motion.div>
                  <h3 className="text-2xl font-bold mb-2">Training Started!</h3>
                  <p className="text-muted-foreground mb-2">
                    Your training job has been submitted successfully.
                  </p>
                  <p className="text-sm font-mono text-cyan-400 mb-6">
                    Task ID: training_{Date.now()}
                  </p>
                  <Button
                    variant="outline"
                    onClick={() => setSubmitted(false)}
                  >
                    Start Another Training
                  </Button>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex justify-end"
                >
                  <Button
                    size="lg"
                    onClick={handleSubmit}
                    disabled={!selectedModel || !datasetPath || isSubmitting}
                    className="cyber text-lg px-8"
                  >
                    {isSubmitting ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Submitting...
                      </>
                    ) : (
                      <>
                        <Rocket className="w-5 h-5 mr-2" />
                        Start Training
                      </>
                    )}
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>
          </CardContent>
        </HoverCard>
      </motion.div>
    </motion.div>
  )
}

// Simple Cog icon component
function CogIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  )
}
