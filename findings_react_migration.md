# Findings: React Frontend Migration Research

## 1. Streamlit 功能分析

### 当前页面功能清单
| 页面 | 核心功能 | 复杂度 |
|------|----------|--------|
| Dashboard | 系统状态监控、任务概览、快速入口 | 中 |
| Data Discovery | 数据集搜索、筛选、下载 | 中 |
| Training | 模型选择、参数配置、任务提交 | 中 |
| Models | 模型管理、导出 | 低 |
| Data Analysis | AI 数据分析请求 | 中 |
| Auto Label | 自动标注配置与提交 | 中 |

### 关键交互模式
- 侧边栏导航
- 表单输入 (模型选择、参数配置)
- 卡片式数据展示
- 实时状态更新 (polling)
- 任务提交与进度跟踪

---

## 2. 最佳实践搜索结果

### React ML Dashboard 优秀案例
1. **Weights & Biases (W&B)** - ML 实验追踪 Dashboard
   - 特点: 深色主题、实时图表、任务状态
   - 学习点: 简洁的信息层级、专业的指标展示

2. **MLflow UI** - 开源 ML 平台
   - 特点: 功能导向、表格化展示
   - 学习点: 数据密度高但不杂乱

3. **Ray Tune Dashboard** - 分布式训练调参
   - 特点: 并行任务可视化、超参热力图
   - 学习点: 大规模任务的状态管理

4. **Comet.ml** - ML 实验管理
   - 特点: 图表丰富、对比功能强
   - 学习点: 实验对比 UX

### shadcn/ui 在 ML 产品中的应用
- Vercel AI SDK 文档站
- LangChain Chat UI
- Pinecone Console

---

## 3. 设计方向决策

### 视觉风格: "Neural Dark"
- **主色调**: 深灰 (#0a0a0b) + 青色高亮 (#06d6a0)
- **辅助色**: 冷蓝 (#00b4d8), 警告橙 (#f59e0b), 错误红 (#ef4444)
- **字体**:
  - Display: JetBrains Mono (等宽 tech feel)
  - Body: Geist Sans / Outfit
- **效果**:
  - 玻璃拟态 (glassmorphism) 卡片
  - 微妙的发光边框 (glow effect)
  - 网格背景纹理

### 布局策略
- 固定侧边栏 + 可折叠
- 顶部状态栏
- 主内容区卡片式布局
- 响应式设计 (mobile drawer nav)

---

## 4. 技术栈确认

### 核心依赖
```json
{
  "next": "14.x (App Router)",
  "react": "18.x",
  "shadcn/ui": "latest",
  "tailwindcss": "3.4.x",
  "framer-motion": "10.x",
  "@tanstack/react-query": "5.x",
  "zustand": "4.x",
  "lucide-react": "latest",
  "recharts": "2.x",
  "clsx": "latest",
  "tailwind-merge": "latest"
}
```

### 项目结构
```
src/
├── app/
│   ├── layout.tsx
│   ├── page.tsx (Dashboard)
│   ├── discovery/
│   ├── training/
│   ├── models/
│   ├── analysis/
│   └── labeling/
├── components/
│   ├── ui/ (shadcn)
│   ├── layout/
│   │   ├── sidebar.tsx
│   │   ├── header.tsx
│   │   └── nav.tsx
│   └── features/
│       ├── dashboard/
│       ├── discovery/
│       ├── training/
│       └── ...
├── lib/
│   ├── api.ts
│   ├── utils.ts
│   └── store.ts
└── styles/
    └── globals.css
```

---

## 6. 参考的模板和资源

### 推荐使用的模板/资源
1. **shadcn/ui Sidebar Template** (GitHub: salimi-my/shadcn-ui-sidebar)
   - 特点: 可折叠侧边栏、响应式、Zustand 状态管理
   - 适合: 本项目的导航结构

2. **Shadcn Admin Template** (GitHub: satnaing/shadcn-admin)
   - 特点: 完整 admin 功能、搜索命令面板、深色主题
   - 适合: 参考组件组织方式

3. **Vercel Next.js Dashboard Course**
   - 官方 Next.js 学习路径
   - 最佳实践: Server Components、Streaming、Error Handling

### Dark Theme 设计要点
- 使用 soft gray (#0a0a0b, #111113) 而非纯黑
- 高对比度文本 (#fafafa on #111113)
- 强调色: Cyan (#06d6a0), Blue (#00b4d8)
- 卡片使用 subtle borders + glow effects
- 数据密集区域使用适当的对比度

---

## 7. 迁移复杂度评估

| 页面 | 组件数 | 预估工时 | 优先级 |
|------|--------|----------|--------|
| Dashboard | 8 | 4h | P0 |
| Data Discovery | 12 | 6h | P1 |
| Training | 15 | 8h | P1 |
| Models | 8 | 4h | P2 |
| Data Analysis | 10 | 5h | P2 |
| Auto Label | 12 | 6h | P2 |

**总预估工时**: ~33h (不含测试)
