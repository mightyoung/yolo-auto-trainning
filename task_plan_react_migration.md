# Task Plan: YOLO Auto-Training React Frontend Migration

## Project Overview
- **Project Name**: YOLO Auto-Training Web UI Migration
- **Goal**: 将 Streamlit 前端迁移到 Next.js + React + shadcn/ui
- **Target Users**: AI/ML 工程师、数据科学家
- **Aesthetic Direction**: 科技感 + 专业性 (Dark Theme, Cyberpunk-inspired, High-tech)

---

## Phase 1: 需求分析与设计方向 ✅ (in_progress)

### Tasks
- [ ] 1.1 分析现有 Streamlit 功能清单
- [ ] 1.2 搜索 React + ML Dashboard 最佳实践
- [ ] 1.3 确定视觉设计方向 (配色、字体、布局)
- [ ] 1.4 创建设计稿/组件规划

### Status: completed

---

## Phase 2: 技术选型与项目搭建 ✅ (completed)

### Completed Tasks
- [x] 2.1 初始化 Next.js 14 项目 (App Router)
- [x] 2.2 集成 shadcn/ui 组件库
- [x] 2.3 配置 Tailwind CSS 主题
- [x] 2.4 设置项目目录结构

### Files Created
- package.json - 项目依赖配置
- next.config.mjs - Next.js 配置
- tailwind.config.ts - Tailwind 主题配置 (含自定义颜色和动画)
- tsconfig.json - TypeScript 配置
- postcss.config.mjs - PostCSS 配置
- components.json - shadcn 配置
- src/app/globals.css - 全局样式 (含 cyber grid 背景、玻璃拟态)
- src/lib/utils.ts - 工具函数
- src/lib/api.ts - API 客户端
- src/lib/store.ts - Zustand 状态管理
- src/components/ui/* - shadcn/ui 组件 (15+ 组件)
- src/components/layout/sidebar.tsx - 侧边栏导航
- src/components/layout/header.tsx - 顶部导航栏
- src/components/theme-provider.tsx - 主题提供者

---

## Phase 3: 页面开发 ✅ (completed)

### Completed Tasks
- [x] 3.1 Dashboard 页面 (page.tsx)
- [x] 3.2 Data Discovery 页面 (discovery/page.tsx)
- [x] 3.3 Training 页面 (training/page.tsx)
- [x] 3.4 Models 页面 (models/page.tsx)
- [x] 3.5 Data Analysis 页面 (analysis/page.tsx)
- [x] 3.6 Auto Label 页面 (labeling/page.tsx)

### Tasks
- [ ] 2.1 初始化 Next.js 14 项目 (App Router)
- [ ] 2.2 集成 shadcn/ui 组件库
- [ ] 2.3 配置 Tailwind CSS 主题
- [ ] 2.4 设置项目目录结构

---

## Phase 3: 核心组件开发

### Tasks
- [ ] 3.1 Layout 组件 (Sidebar, Header, Main Content)
- [ ] 3.2 Dashboard 页面组件
- [ ] 3.3 Data Discovery 页面组件
- [ ] 3.4 Training 页面组件
- [ ] 3.5 Models 页面组件
- [ ] 3.6 Data Analysis 页面组件
- [ ] 3.7 Auto Label 页面组件

---

## Phase 4: 状态管理与 API 集成

### Tasks
- [ ] 4.1 状态管理方案 (Zustand/Context)
- [ ] 4.2 API 客户端封装
- [ ] 4.3 WebSocket 实时更新 (可选)
- [ ] 4.4 错误处理与 Loading 状态

---

## Phase 5: 动画与交互优化

### Tasks
- [ ] 5.1 页面切换动画
- [ ] 5.2 微交互设计
- [ ] 5.3 Loading 骨架屏
- [ ] 5.4 响应式适配

---

## Phase 6: 测试与优化

### Tasks
- [ ] 6.1 单元测试
- [ ] 6.2 E2E 测试
- [ ] 6.3 性能优化
- [ ] 6.4 部署配置

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-16 | 使用 Next.js 14 + shadcn/ui | 成熟稳定，社区活跃 |
| 2026-03-16 | Dark Theme + Cyberpunk 风格 | 符合 ML/AI 产品定位 |
| 2026-03-16 | 使用 App Router | Next.js 官方推荐 |

---

## Dependencies
- next: ^14.0.0
- react: ^18.2.0
- shadcn/ui (latest)
- tailwindcss: ^3.4.0
- framer-motion: ^10.0.0
- @tanstack/react-query: ^5.0.0
- zustand: ^4.4.0
- lucide-react: (icons)
- recharts: (charts)
