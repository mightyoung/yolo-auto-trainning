'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Sparkles, Loader2, FolderOpen, Plus, X, ArrowRight,
  Target, Zap, Eye, CheckCircle2, FileImage
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { HoverCard } from '@/components/ui/hover-card'

const baseModels = [
  { id: 'grounded_sam', name: 'GroundedSAM', description: 'Best accuracy - Recommended', icon: <Target className="w-6 h-6" />, gradient: 'from-cyan-500 to-blue-500', speed: 'Slow', accuracy: 'Highest' },
  { id: 'grounding_dino', name: 'GroundingDINO', description: 'Fast, open-set detection', icon: <Zap className="w-6 h-6" />, gradient: 'from-amber-500 to-orange-500', speed: 'Fast', accuracy: 'High' },
  { id: 'owlv2', name: 'OWLv2', description: 'Zero-shot detection', icon: <Eye className="w-6 h-6" />, gradient: 'from-purple-500 to-pink-500', speed: 'Fastest', accuracy: 'Medium' },
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

export default function LabelingPage() {
  const [baseModel, setBaseModel] = useState('grounded_sam')
  const [confThreshold, setConfThreshold] = useState(0.3)
  const [classes, setClasses] = useState<string[]>([])
  const [newClass, setNewClass] = useState('')
  const [inputFolder, setInputFolder] = useState('')
  const [outputFolder, setOutputFolder] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleAddClass = () => {
    if (newClass.trim() && !classes.includes(newClass.trim())) {
      setClasses([...classes, newClass.trim()])
      setNewClass('')
    }
  }

  const handleRemoveClass = (cls: string) => {
    setClasses(classes.filter(c => c !== cls))
  }

  const handleSubmit = async () => {
    if (!inputFolder || classes.length === 0) return

    setIsSubmitting(true)
    await new Promise(resolve => setTimeout(resolve, 2000))
    setIsSubmitting(false)
  }

  const selectedModel = baseModels.find(m => m.id === baseModel)

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
          Auto Labeling
        </h1>
        <p className="text-muted-foreground mt-2 text-lg">
          AI-powered automatic image labeling
        </p>
      </motion.div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Base Model Selection */}
        <motion.div variants={itemVariants}>
          <HoverCard hoverEffect="lift" className="h-full border-primary/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-purple-400" />
                Base Model
              </CardTitle>
              <CardDescription>Select the foundation model for labeling</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {baseModels.map((model, index) => (
                <motion.button
                  key={model.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ scale: 1.02, x: 4 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setBaseModel(model.id)}
                  className={`w-full p-4 rounded-xl border-2 text-left transition-all duration-300 ${
                    baseModel === model.id
                      ? 'border-purple-500 bg-purple-500/10 shadow-lg shadow-purple-500/20'
                      : 'border-border hover:border-purple-500/30 hover:bg-muted/50'
                  }`}
                >
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-xl bg-gradient-to-br ${model.gradient}`}>
                      {React.cloneElement(model.icon as React.ReactElement, { className: "w-6 h-6 text-white" })}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold">{model.name}</h3>
                        {baseModel === model.id && (
                          <CheckCircle2 className="w-5 h-5 text-purple-400" />
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">{model.description}</p>
                      <div className="flex gap-4 mt-3">
                        <Badge variant="outline" className="text-xs">
                          Speed: {model.speed}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          Accuracy: {model.accuracy}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </motion.button>
              ))}
            </CardContent>
          </HoverCard>
        </motion.div>

        {/* Configuration */}
        <motion.div variants={itemVariants} className="space-y-6">
          {/* Confidence */}
          <HoverCard hoverEffect="lift" className="border-primary/10">
            <CardHeader>
              <CardTitle>Confidence Threshold</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label className="text-lg">{confThreshold}</Label>
                <Badge variant={confThreshold > 0.5 ? 'warning' : 'success'}>
                  {confThreshold > 0.5 ? 'High' : 'Balanced'}
                </Badge>
              </div>
              <Slider
                value={[confThreshold]}
                onValueChange={([value]) => setConfThreshold(value)}
                min={0.1}
                max={0.9}
                step={0.05}
              />
              <p className="text-sm text-muted-foreground">
                Lower = more detections, more false positives. Higher = fewer but more confident.
              </p>
            </CardContent>
          </HoverCard>

          {/* Classes */}
          <HoverCard hoverEffect="lift" className="border-primary/10">
            <CardHeader>
              <CardTitle>Classes to Detect</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  placeholder="Enter class name"
                  value={newClass}
                  onChange={(e) => setNewClass(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAddClass()}
                  className="bg-muted/50"
                />
                <Button onClick={handleAddClass} className="bg-gradient-to-r from-purple-500 to-pink-500">
                  <Plus className="w-4 h-4" />
                </Button>
              </div>
              <div className="flex flex-wrap gap-2 min-h-[60px] p-4 rounded-xl bg-muted/30">
                <AnimatePresence>
                  {classes.map((cls) => (
                    <motion.span
                      key={cls}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 0, opacity: 0 }}
                      className="inline-flex items-center gap-1 px-4 py-2 rounded-full bg-gradient-to-r from-purple-500/20 to-pink-500/20 text-purple-400 text-sm font-medium"
                    >
                      {cls}
                      <button
                        onClick={() => handleRemoveClass(cls)}
                        className="hover:text-white transition-colors"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </motion.span>
                  ))}
                </AnimatePresence>
                {classes.length === 0 && (
                  <p className="text-muted-foreground text-sm">Add classes to detect...</p>
                )}
              </div>
            </CardContent>
          </HoverCard>
        </motion.div>
      </div>

      {/* Data Paths */}
      <motion.div variants={itemVariants}>
        <HoverCard hoverEffect="lift" className="border-primary/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FolderOpen className="w-5 h-5 text-cyan-400" />
              Data Paths
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-3">
                <Label className="text-base">Input Folder</Label>
                <div className="relative">
                  <FileImage className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    placeholder="e.g., datasets/my_images"
                    value={inputFolder}
                    onChange={(e) => setInputFolder(e.target.value)}
                    className="pl-10 h-12 bg-muted/50 border-primary/10 focus:border-primary/30"
                  />
                </div>
                <p className="text-sm text-muted-foreground">Images to label</p>
              </div>
              <div className="space-y-3">
                <Label className="text-base">Output Folder (Optional)</Label>
                <div className="relative">
                  <FolderOpen className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <Input
                    placeholder="e.g., datasets/labeled"
                    value={outputFolder}
                    onChange={(e) => setOutputFolder(e.target.value)}
                    className="pl-10 h-12 bg-muted/50 border-primary/10 focus:border-primary/30"
                  />
                </div>
                <p className="text-sm text-muted-foreground">Where to save labeled dataset</p>
              </div>
            </div>

            <div className="flex justify-end">
              <Button
                size="lg"
                onClick={handleSubmit}
                disabled={!inputFolder || classes.length === 0 || isSubmitting}
                className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-400 hover:to-pink-400"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5 mr-2" />
                    Start Auto Labeling
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </HoverCard>
      </motion.div>

      {/* Output Format */}
      <motion.div variants={itemVariants}>
        <HoverCard hoverEffect="glow" className="border-primary/10">
          <CardHeader>
            <CardTitle>Output Format</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="p-6 rounded-xl bg-muted/30 font-mono text-sm space-y-2">
              <p className="text-purple-400">YOLO format:</p>
              <div className="pl-4 space-y-1 text-muted-foreground">
                <p>project/</p>
                <p className="pl-4">├── images/</p>
                <p className="pl-4">├── labels/</p>
                <p className="pl-4">└── data.yaml</p>
              </div>
            </div>
          </CardContent>
        </HoverCard>
      </motion.div>
    </motion.div>
  )
}
