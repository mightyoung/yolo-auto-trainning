# YOLO Auto-Training Web UI

Modern React frontend for YOLO Auto-Training platform, built with Next.js 14, shadcn/ui, and Tailwind CSS.

## Features

- **Dashboard** - System overview, quick actions, recent training jobs
- **Data Discovery** - Search datasets from Roboflow, Kaggle, HuggingFace
- **Training** - Model selection, parameter configuration, job submission
- **Models** - Trained model management and export
- **Data Analysis** - AI-powered data analysis with DeepAnalyze
- **Auto Label** - Automatic image labeling with foundation models

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **UI Library**: shadcn/ui
- **Styling**: Tailwind CSS v4
- **Animation**: Framer Motion
- **State Management**: Zustand
- **Data Fetching**: React Query (TanStack Query)
- **Icons**: Lucide React
- **Charts**: Recharts

## Getting Started

### Prerequisites

- Node.js 18+
- npm or pnpm

### Installation

```bash
# Install dependencies
npm install
# or
pnpm install
```

### Development

```bash
# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
# Build for production
npm run build

# Start production server
npm run start
```

## Project Structure

```
src/
├── app/                    # Next.js App Router pages
│   ├── page.tsx           # Dashboard
│   ├── discovery/         # Data Discovery
│   ├── training/          # Training
│   ├── models/            # Models
│   ├── analysis/         # Data Analysis
│   └── labeling/         # Auto Label
├── components/
│   ├── ui/               # shadcn/ui components
│   └── layout/           # Layout components
├── lib/
│   ├── api.ts            # API client
│   ├── store.ts          # Zustand store
│   └── utils.ts          # Utility functions
└── styles/
    └── globals.css        # Global styles
```

## Design System

### Colors

| Color | Hex | Usage |
|-------|-----|-------|
| Primary | #06d6a0 | Main accent, buttons |
| Secondary | #00b4d8 | Secondary actions |
| Background | #0a0a0b | Dark theme bg |
| Surface | #111113 | Card backgrounds |
| Purple | #a855f7 | Accent highlights |
| Amber | #f59e0b | Warnings |

### Typography

- **Display**: JetBrains Mono
- **Body**: Outfit

### Animations

- Page transitions with Framer Motion
- Hover effects with subtle scale/glow
- Loading states with skeleton screens
- Staggered reveal animations

### Components

- **HoverCard**: Card with hover effects (glow, lift, scale, border)
- **Skeleton**: Loading placeholders
- **Tooltip**: Contextual hints
- **Toast**: Notification system

## Configuration

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

## UI/UX Improvements (2025)

Based on latest design trends:

1. **Bento Grid Layout** - Organized content sections
2. **Gradient Mesh Backgrounds** - Rich visual atmosphere
3. **Glassmorphism** - Modern translucent effects
4. **Glow Effects** - Subtle neon highlights
5. **Skeleton Loading** - Better perceived performance
6. **Micro-interactions** - Engaging feedback
7. **Tooltips** - Contextual help
8. **Toast Notifications** - Non-blocking alerts

## License

MIT
