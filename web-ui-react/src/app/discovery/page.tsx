'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Download, ExternalLink, Filter, Loader2, Sparkles, FileImage, Users, Globe } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { HoverCard } from '@/components/ui/hover-card'
import { Skeleton } from '@/components/ui/skeleton'

// Mock data
const mockDatasets = [
  { id: '1', name: 'COCO 2017', source: 'Roboflow', images: 123287, license: 'MIT', relevance: 0.95 },
  { id: '2', name: 'VOC Detection', source: 'Kaggle', images: 21503, license: 'GPL-3', relevance: 0.88 },
  { id: '3', name: 'Open Images V7', source: 'HuggingFace', images: 9011219, license: 'Apache-2.0', relevance: 0.82 },
  { id: '4', name: 'BDD100K', source: 'Roboflow', images: 100000, license: 'CC-BY-NC-4.0', relevance: 0.78 },
  { id: '5', name: 'KITTI Vision', source: 'Kaggle', images: 14999, license: 'Apache-2.0', relevance: 0.72 },
]

const sources = ['Roboflow', 'Kaggle', 'HuggingFace']

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

const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
}

export default function DiscoveryPage() {
  const [query, setQuery] = useState('')
  const [selectedSources, setSelectedSources] = useState<string[]>(sources)
  const [minImages, setMinImages] = useState(1000)
  const [isSearching, setIsSearching] = useState(false)
  const [results, setResults] = useState<typeof mockDatasets | null>(null)

  const handleSearch = async () => {
    setIsSearching(true)
    await new Promise(resolve => setTimeout(resolve, 1500))
    setResults(mockDatasets.filter(d =>
      d.name.toLowerCase().includes(query.toLowerCase()) &&
      selectedSources.includes(d.source) &&
      d.images >= minImages
    ))
    setIsSearching(false)
  }

  const toggleSource = (source: string) => {
    setSelectedSources(prev =>
      prev.includes(source)
        ? prev.filter(s => s !== source)
        : [...prev, source]
    )
  }

  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'Roboflow': return <Globe className="w-4 h-4" />
      case 'Kaggle': return <FileImage className="w-4 h-4" />
      case 'HuggingFace': return <Sparkles className="w-4 h-4" />
      default: return <Globe className="w-4 h-4" />
    }
  }

  const getSourceColor = (source: string) => {
    switch (source) {
      case 'Roboflow': return 'from-cyan-500 to-blue-500'
      case 'Kaggle': return 'from-green-500 to-emerald-500'
      case 'HuggingFace': return 'from-purple-500 to-pink-500'
      default: return 'from-gray-500 to-slate-500'
    }
  }

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
          Data Discovery
        </h1>
        <p className="text-muted-foreground mt-2 text-lg">
          Search and discover datasets from multiple sources
        </p>
      </motion.div>

      {/* Search Card */}
      <motion.div variants={itemVariants}>
        <HoverCard hoverEffect="lift" className="border-primary/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="w-5 h-5 text-cyan-400" />
              Search Datasets
            </CardTitle>
            <CardDescription>Find the perfect dataset for your model</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Search Input */}
            <div className="flex gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search datasets (e.g., car detection, traffic sign...)"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  className="pl-9 h-12 bg-muted/50 border-primary/10 focus:border-primary/30"
                />
              </div>
              <Button
                onClick={handleSearch}
                disabled={isSearching}
                size="lg"
                className="cyber"
              >
                {isSearching ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4 mr-2" />
                    Search
                  </>
                )}
              </Button>
            </div>

            {/* Filters */}
            <div className="flex flex-wrap items-center gap-4 p-4 rounded-xl bg-muted/30">
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium">Sources:</span>
              </div>
              <div className="flex gap-3">
                {sources.map(source => (
                  <motion.button
                    key={source}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => toggleSource(source)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
                      selectedSources.includes(source)
                        ? `bg-gradient-to-r ${getSourceColor(source)} text-white shadow-lg`
                        : 'bg-muted text-muted-foreground hover:bg-muted/80'
                    }`}
                  >
                    {getSourceIcon(source)}
                    {source}
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Min Images */}
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium">Min Images:</span>
              <Input
                type="number"
                value={minImages}
                onChange={(e) => setMinImages(Number(e.target.value))}
                className="w-32 bg-muted/50"
              />
            </div>
          </CardContent>
        </HoverCard>
      </motion.div>

      {/* Results */}
      <AnimatePresence mode="wait">
        {results && (
          <motion.div
            variants={cardVariants}
            initial="hidden"
            animate="show"
            exit="exit"
          >
            <HoverCard hoverEffect="lift" className="border-primary/10">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <FileImage className="w-5 h-5 text-cyan-400" />
                      Search Results
                    </CardTitle>
                    <CardDescription>Found {results.length} datasets</CardDescription>
                  </div>
                  <Badge variant="outline" className="gap-1">
                    <Sparkles className="w-3 h-3" />
                    AI Recommended
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  <AnimatePresence>
                    {results.map((dataset, index) => (
                      <motion.div
                        key={dataset.id}
                        variants={cardVariants}
                        initial="hidden"
                        animate="show"
                        transition={{ delay: index * 0.05 }}
                      >
                        <DatasetCard dataset={dataset} getSourceIcon={getSourceIcon} getSourceColor={getSourceColor} />
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              </CardContent>
            </HoverCard>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Placeholder */}
      {!results && !isSearching && (
        <motion.div variants={itemVariants}>
          <Card className="border-none bg-transparent shadow-none">
            <CardContent className="py-20 text-center">
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.2 }}
              >
                <div className="w-24 h-24 mx-auto rounded-2xl bg-gradient-to-br from-cyan-500/20 to-cyber-blue-500/20 flex items-center justify-center mb-6">
                  <Search className="w-10 h-10 text-muted-foreground/50" />
                </div>
              </motion.div>
              <h3 className="text-xl font-semibold mb-2">Discover Datasets</h3>
              <p className="text-muted-foreground max-w-md mx-auto">
                Search across Roboflow, Kaggle, and HuggingFace to find the perfect dataset for your YOLO training
              </p>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Loading */}
      {isSearching && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="py-20"
        >
          <Card className="border-none bg-transparent shadow-none">
            <CardContent className="flex flex-col items-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
              >
                <Loader2 className="w-12 h-12 text-cyan-400" />
              </motion.div>
              <p className="mt-4 text-muted-foreground">Searching across {selectedSources.length} sources...</p>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </motion.div>
  )
}

function DatasetCard({
  dataset,
  getSourceIcon,
  getSourceColor,
}: {
  dataset: { id: string; name: string; source: string; images: number; license: string; relevance: number }
  getSourceIcon: (source: string) => React.ReactNode
  getSourceColor: (source: string) => string
}) {
  return (
    <motion.div
      whileHover={{ x: 4 }}
      className="group flex items-center justify-between p-5 rounded-xl bg-muted/30 hover:bg-muted/50 transition-all duration-300 border border-transparent hover:border-primary/10"
    >
      <div className="flex items-center gap-5">
        <motion.div
          whileHover={{ scale: 1.1, rotate: 5 }}
          className={`p-4 rounded-xl bg-gradient-to-br ${getSourceColor(dataset.source)}/20`}
        >
          {getSourceIcon(dataset.source)}
        </motion.div>
        <div>
          <p className="font-semibold text-lg group-hover:text-cyan-400 transition-colors">{dataset.name}</p>
          <div className="flex items-center gap-4 mt-2">
            <Badge variant="outline" className="gap-1">
              {getSourceIcon(dataset.source)}
              {dataset.source}
            </Badge>
            <span className="flex items-center gap-1 text-sm text-muted-foreground">
              <FileImage className="w-3 h-3" />
              {dataset.images.toLocaleString()} images
            </span>
            <span className="text-sm text-muted-foreground">
              {dataset.license}
            </span>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className="text-right">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="text-2xl font-bold gradient-text"
          >
            {Math.round(dataset.relevance * 100)}%
          </motion.div>
          <p className="text-xs text-muted-foreground">match</p>
        </div>
        <Button variant="outline" size="sm" className="gap-2">
          <Download className="w-4 h-4" />
          Download
        </Button>
      </div>
    </motion.div>
  )
}
