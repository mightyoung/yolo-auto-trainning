'use client'

import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard,
  Search,
  Cog,
  Package,
  FlaskConical,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  Rocket,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useAppStore, NavPage } from '@/lib/store'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet'
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from '@/components/ui/tooltip'
import { useState } from 'react'

const navItems: { title: string; href: string; icon: React.ReactNode; id: NavPage }[] = [
  { title: 'Dashboard', href: '/', icon: <LayoutDashboard className="w-5 h-5" />, id: 'dashboard' },
  { title: 'Data Discovery', href: '/discovery', icon: <Search className="w-5 h-5" />, id: 'discovery' },
  { title: 'Training', href: '/training', icon: <Rocket className="w-5 h-5" />, id: 'training' },
  { title: 'Models', href: '/models', icon: <Package className="w-5 h-5" />, id: 'models' },
  { title: 'Analysis', href: '/analysis', icon: <FlaskConical className="w-5 h-5" />, id: 'analysis' },
  { title: 'Auto Label', href: '/labeling', icon: <Sparkles className="w-5 h-5" />, id: 'labeling' },
]

export function Sidebar() {
  const pathname = usePathname()
  const { sidebarCollapsed, toggleSidebar, currentPage, setCurrentPage } = useAppStore()
  const [mobileOpen, setMobileOpen] = useState(false)

  const NavContent = () => (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <motion.div
        className={cn(
          'flex items-center gap-3 px-4 py-6 border-b border-border/30',
          sidebarCollapsed ? 'justify-center' : ''
        )}
        layout
      >
        <motion.div
          whileHover={{ rotate: 360 }}
          transition={{ duration: 0.5 }}
          className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-cyber-blue-500 flex items-center justify-center shadow-lg shadow-cyan-500/20"
        >
          <Rocket className="w-5 h-5 text-black" />
        </motion.div>
        <AnimatePresence>
          {!sidebarCollapsed && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              className="flex flex-col"
            >
              <span className="font-bold text-lg tracking-tight">YOLO</span>
              <span className="text-xs text-muted-foreground">Auto-Training</span>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = currentPage === item.id || pathname === item.href

          return sidebarCollapsed ? (
            <TooltipProvider key={item.id}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Link
                    href={item.href}
                    onClick={() => setCurrentPage(item.id)}
                    className={cn(
                      'flex items-center justify-center p-3 rounded-lg transition-all duration-200 relative',
                      isActive
                        ? 'bg-cyan-500/10 text-cyan-400'
                        : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'
                    )}
                  >
                    {isActive && (
                      <motion.div
                        layoutId="sidebar-active-collapsed"
                        className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-cyan-500 rounded-r-full"
                      />
                    )}
                    {item.icon}
                  </Link>
                </TooltipTrigger>
                <TooltipContent side="right">
                  {item.title}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          ) : (
            <Link
              key={item.id}
              href={item.href}
              onClick={() => setCurrentPage(item.id)}
              className={cn(
                'flex items-center gap-3 px-3 py-3 rounded-lg transition-all duration-200 group relative',
                isActive
                  ? 'bg-cyan-500/10 text-cyan-400'
                  : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'
              )}
            >
              {isActive && (
                <motion.div
                  layoutId="sidebar-active"
                  className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-cyan-500 rounded-r-full"
                />
              )}
              <motion.span
                className={cn(
                  'transition-colors',
                  isActive ? 'text-cyan-400' : 'group-hover:text-foreground'
                )}
                whileHover={{ x: 2 }}
              >
                {item.icon}
              </motion.span>
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-sm font-medium"
              >
                {item.title}
              </motion.span>
            </Link>
          )
        })}
      </nav>

      {/* Settings */}
      <div className="px-3 py-4 border-t border-border/30">
        <Link
          href="/settings"
          onClick={() => setCurrentPage('dashboard')}
          className={cn(
            'flex items-center gap-3 px-3 py-3 rounded-lg transition-all duration-200',
            'text-muted-foreground hover:bg-muted/50 hover:text-foreground',
            sidebarCollapsed && 'justify-center'
          )}
        >
          <Cog className="w-5 h-5" />
          {!sidebarCollapsed && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-sm font-medium"
            >
              Settings
            </motion.span>
          )}
        </Link>
      </div>

      {/* Collapse Toggle */}
      <div className="hidden lg:block px-3 py-4 border-t border-border/30">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className={cn(
            'w-full justify-center transition-all duration-200',
            sidebarCollapsed ? '' : 'justify-start'
          )}
        >
          <motion.div
            animate={{ rotate: sidebarCollapsed ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronLeft className="w-4 h-4" />
          </motion.div>
          {!sidebarCollapsed && (
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="ml-2 text-sm"
            >
              Collapse
            </motion.span>
          )}
        </Button>
      </div>
    </div>
  )

  return (
    <>
      {/* Desktop Sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: sidebarCollapsed ? 72 : 260 }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        className={cn(
          'hidden lg:flex flex-col h-screen sticky top-0 bg-card/50 backdrop-blur-2xl border-r border-border/30 transition-all duration-300',
          'shadow-xl shadow-black/5'
        )}
      >
        <NavContent />
      </motion.aside>

      {/* Mobile Sidebar */}
      <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
        <SheetTrigger asChild>
          <Button variant="ghost" size="icon" className="fixed top-4 left-4 z-50 lg:hidden">
            <Rocket className="w-5 h-5" />
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="w-[260px] p-0">
          <NavContent />
        </SheetContent>
      </Sheet>
    </>
  )
}
