'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FlaskConical, Loader2, AlertCircle, FileText, Brain,
  BarChart3, PieChart, TrendingUp, Sparkles, ArrowRight,
  CheckCircle2, XCircle, Clock
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { HoverCard } from '@/components/ui/hover-card'
import { Progress } from '@/components/ui/progress'

const analysisTypes = [
  { id: 'quality', name: 'Quality Analysis', description: 'Detect missing values, outliers, duplicates', icon: <BarChart3 className="w-6 h-6" />, gradient: 'from-cyan-500 to-blue-500' },
  { id: 'distribution', name: 'Distribution Analysis', description: 'Explore correlations and patterns', icon: <PieChart className="w-6 h-6" />, gradient: 'from-purple-500 to-pink-500' },
  { id: 'anomalies', name: 'Anomaly Detection', description: 'Find unusual data points', icon: <TrendingUp className="w-6 h-6" />, gradient: 'from-amber-500 to-orange-500' },
  { id: 'full', name: 'Comprehensive', description: 'Full analysis with all features', icon: <Brain className="w-6 h-6" />, gradient: 'from-green-500 to-emerald-500' },
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

export default function AnalysisPage() {
  const [analysisType, setAnalysisType] = useState('quality')
  const [datasetPath, setDatasetPath] = useState('')
  const [useCustomPrompt, setUseCustomPrompt] = useState(false)
  const [customPrompt, setCustomPrompt] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState(0)

  const handleAnalyze = async () => {
    if (!datasetPath) return

    setIsAnalyzing(true)
    setAnalysisProgress(0)

    // Simulate progress
    for (let i = 0; i <= 100; i += 10) {
      await new Promise(resolve => setTimeout(resolve, 500))
      setAnalysisProgress(i)
    }

    setIsAnalyzing(false)
  }

  const selectedType = analysisTypes.find(t => t.id === analysisType)

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
          Data Analysis
        </h1>
        <p className="text-muted-foreground mt-2 text-lg">
          AI-powered data analysis with DeepAnalyze
        </p>
      </motion.div>

      {/* Analysis Type Selection */}
      <motion.div variants={itemVariants}>
        <HoverCard hoverEffect="lift" className="border-primary/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FlaskConical className="w-5 h-5 text-cyan-400" />
              Select Analysis Type
            </CardTitle>
            <CardDescription>Choose what type of analysis you need</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {analysisTypes.map((type, index) => (
                <motion.button
                  key={type.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  whileHover={{ scale: 1.02, y: -4 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setAnalysisType(type.id)}
                  className={`relative p-5 rounded-xl border-2 text-left transition-all duration-300 overflow-hidden ${
                    analysisType === type.id
                      ? 'border-cyan-500 bg-cyan-500/10 shadow-lg shadow-cyan-500/20'
                      : 'border-border hover:border-primary/30 hover:bg-muted/50'
                  }`}
                >
                  <div className={`absolute inset-0 bg-gradient-to-br ${type.gradient} opacity-0 transition-opacity duration-300 ${analysisType === type.id ? 'opacity-5' : ''}`} />

                  <div className="relative">
                    <div className={`p-3 rounded-xl bg-gradient-to-br ${type.gradient} inline-flex mb-4`}>
                      {React.cloneElement(type.icon as React.ReactElement, { className: "w-6 h-6 text-white" })}
                    </div>
                    <h3 className="font-semibold mb-1">{type.name}</h3>
                    <p className="text-sm text-muted-foreground">{type.description}</p>

                    {analysisType === type.id && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="absolute top-2 right-2"
                      >
                        <CheckCircle2 className="w-5 h-5 text-cyan-400" />
                      </motion.div>
                    )}
                  </div>
                </motion.button>
              ))}
            </div>
          </CardContent>
        </HoverCard>
      </motion.div>

      {/* Configuration */}
      <motion.div variants={itemVariants}>
        <HoverCard hoverEffect="lift" className="border-primary/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-cyan-400" />
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Dataset Path */}
            <div className="space-y-3">
              <Label className="text-base">Dataset Path</Label>
              <div className="relative">
                <FileText className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="e.g., datasets/my_data.csv or datasets/"
                  value={datasetPath}
                  onChange={(e) => setDatasetPath(e.target.value)}
                  className="pl-10 h-12 bg-muted/50 border-primary/10 focus:border-primary/30"
                />
              </div>
              <p className="text-sm text-muted-foreground">
                Path to a data file (CSV, Excel, JSON) or directory containing data files
              </p>
            </div>

            {/* Custom Prompt Toggle */}
            <div className="flex items-center gap-3 p-4 rounded-xl bg-muted/30">
              <Checkbox
                id="custom-prompt"
                checked={useCustomPrompt}
                onCheckedChange={(checked) => setUseCustomPrompt(checked as boolean)}
              />
              <Label htmlFor="custom-prompt" className="cursor-pointer font-medium">
                Use custom analysis prompt
              </Label>
            </div>

            {useCustomPrompt && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-3"
              >
                <Label className="text-base">Custom Prompt</Label>
                <Textarea
                  placeholder="Enter your analysis requirements..."
                  value={customPrompt}
                  onChange={(e) => setCustomPrompt(e.target.value)}
                  rows={4}
                  className="bg-muted/50 border-primary/10 focus:border-primary/30"
                />
              </motion.div>
            )}

            {/* Progress or Submit */}
            <AnimatePresence mode="wait">
              {isAnalyzing ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="p-6 rounded-xl bg-muted/30"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                      >
                        <Loader2 className="w-6 h-6 text-cyan-400" />
                      </motion.div>
                      <div>
                        <p className="font-medium">Analyzing your data...</p>
                        <p className="text-sm text-muted-foreground">{selectedType?.name}</p>
                      </div>
                    </div>
                    <Badge variant="outline" className="font-mono">{analysisProgress}%</Badge>
                  </div>
                  <Progress value={analysisProgress} className="h-2" />
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
                    onClick={handleAnalyze}
                    disabled={!datasetPath}
                    className="cyber"
                  >
                    <FlaskConical className="w-4 h-4 mr-2" />
                    Analyze Data
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </motion.div>
              )}
            </AnimatePresence>
          </CardContent>
        </HoverCard>
      </motion.div>

      {/* Analysis Capabilities */}
      <motion.div variants={itemVariants}>
        <HoverCard hoverEffect="glow" className="border-primary/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-cyan-400" />
              Analysis Capabilities
            </CardTitle>
            <CardDescription>What DeepAnalyze can do for your data</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {analysisTypes.map((type, index) => (
                <motion.div
                  key={type.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ scale: 1.02 }}
                  className={`p-5 rounded-xl bg-gradient-to-br ${type.gradient}/10 border border-current/10 hover:border-current/30 transition-all`}
                >
                  <div className={`p-3 rounded-xl bg-gradient-to-br ${type.gradient} inline-flex mb-4`}>
                    {React.cloneElement(type.icon as React.ReactElement, { className: "w-5 h-5 text-white" })}
                  </div>
                  <h4 className="font-semibold mb-1">{type.name}</h4>
                  <p className="text-sm text-muted-foreground">{type.description}</p>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </HoverCard>
      </motion.div>
    </motion.div>
  )
}
