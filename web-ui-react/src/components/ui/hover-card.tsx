'use client'

import React from 'react'
import { motion, HTMLMotionProps } from 'framer-motion'
import { cn } from '@/lib/utils'

interface HoverCardProps {
  hoverEffect?: 'glow' | 'lift' | 'scale' | 'border'
  children: React.ReactNode
  className?: string
}

export function HoverCard({
  hoverEffect = 'lift',
  className,
  children,
}: HoverCardProps) {
  const hoverVariants = {
    glow: {
      whileHover: {
        boxShadow: '0 0 30px hsl(var(--primary) / 0.2), 0 0 60px hsl(var(--primary) / 0.1)',
      },
    },
    lift: {
      whileHover: {
        y: -4,
        transition: { duration: 0.2 },
      },
    },
    scale: {
      whileHover: {
        scale: 1.02,
        transition: { duration: 0.2 },
      },
    },
    border: {
      whileHover: {
        borderColor: 'hsl(var(--primary) / 0.5)',
        transition: { duration: 0.2 },
      },
    },
  }

  return (
    <motion.div
      className={cn(
        'rounded-xl border bg-card text-card-foreground shadow-card transition-all duration-300',
        className
      )}
      whileHover={hoverVariants[hoverEffect].whileHover}
    >
      {children}
    </motion.div>
  )
}
