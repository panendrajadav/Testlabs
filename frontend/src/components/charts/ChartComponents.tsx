'use client'

import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from 'recharts'
import { motion } from 'framer-motion'

const COLORS = ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899']

interface ChartContainerProps {
  title: string
  children: React.ReactNode
  animate?: boolean
}

export function ChartContainer({ title, children, animate = true }: ChartContainerProps) {
  return (
    <motion.div
      initial={animate ? { opacity: 0, scale: 0.95 } : {}}
      whileInView={animate ? { opacity: 1, scale: 1 } : {}}
      transition={{ duration: 0.5 }}
      className="bg-gradient-to-br from-slate-900/50 to-slate-800/50 rounded-xl p-6 border border-purple-500/20 glass glow-hover"
    >
      <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
      <div className="w-full h-64">{children}</div>
    </motion.div>
  )
}

// Model Metrics Chart — accepts EvalResult[] (backend uses model_name) or Record<string,number>
export function ModelMetricsChart({ data }: { data: Array<{ model?: string; model_name?: string; score?: number }> | Record<string, number> }) {
  const chartData = Array.isArray(data)
    ? data.map((r) => ({ name: r.model_name ?? r.model ?? 'unknown', value: (((r.score ?? 0) * 100)).toFixed(2) }))
    : Object.entries(data).map(([key, value]) => ({ name: key.replace(/_/g, ' '), value: (value * 100).toFixed(2) }))

  return (
    <ChartContainer title="Model Metrics">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
        >
          <defs>
            <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#8b5cf6" />
              <stop offset="100%" stopColor="#06b6d4" />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(139, 92, 246, 0.1)" />
          <XAxis dataKey="name" stroke="rgba(255, 255, 255, 0.5)" style={{ fontSize: '12px' }} />
          <YAxis stroke="rgba(255, 255, 255, 0.5)" style={{ fontSize: '12px' }} />
          <Tooltip
            contentStyle={{
              background: 'rgba(0, 0, 0, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.5)',
              borderRadius: '8px',
            }}
            cursor={{ fill: 'rgba(139, 92, 246, 0.1)' }}
          />
          <Bar
            dataKey="value"
            fill="url(#colorGradient)"
            radius={[8, 8, 0, 0]}
            animationDuration={1000}
          />
        </BarChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}

// ROC Curve
export function ROCCurveChart({ data }: { data: Array<{ fpr: number; tpr: number }> }) {
  return (
    <ChartContainer title="ROC Curve">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(139, 92, 246, 0.1)" />
          <XAxis
            dataKey="fpr"
            stroke="rgba(255, 255, 255, 0.5)"
            style={{ fontSize: '12px' }}
            label={{ value: 'FPR', position: 'insideBottomRight', offset: -5 }}
          />
          <YAxis
            stroke="rgba(255, 255, 255, 0.5)"
            style={{ fontSize: '12px' }}
            label={{ value: 'TPR', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            contentStyle={{
              background: 'rgba(0, 0, 0, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.5)',
              borderRadius: '8px',
            }}
          />
          <Line
            type="monotone"
            dataKey="tpr"
            stroke="#8b5cf6"
            strokeWidth={2}
            dot={false}
            isAnimationActive={true}
          />
          <Line
            type="monotone"
            dataKey={() => 0.5}
            stroke="rgba(255, 255, 255, 0.3)"
            strokeDasharray="5 5"
            dot={false}
            name="Random Classifier"
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}

// Feature Importance
export function FeatureImportanceChart({
  data,
}: {
  data: Array<{ feature: string; importance: number }>
}) {
  const sortedData = [...data].sort((a, b) => b.importance - a.importance).slice(0, 10)

  return (
    <ChartContainer title="Top 10 Feature Importance">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={sortedData}
          margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
        >
          <defs>
            <linearGradient id="importanceGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#06b6d4" />
              <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(139, 92, 246, 0.1)" />
          <XAxis type="number" stroke="rgba(255, 255, 255, 0.5)" style={{ fontSize: '12px' }} />
          <YAxis
            dataKey="feature"
            type="category"
            stroke="rgba(255, 255, 255, 0.5)"
            style={{ fontSize: '12px' }}
            width={140}
          />
          <Tooltip
            contentStyle={{
              background: 'rgba(0, 0, 0, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.5)',
              borderRadius: '8px',
            }}
          />
          <Bar
            dataKey="importance"
            fill="url(#importanceGradient)"
            animationDuration={1000}
            radius={[0, 8, 8, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}

// Confusion Matrix
export function ConfusionMatrixChart({
  data,
}: {
  data: Array<Array<number>>
}) {
  const maxValue = Math.max(...data.flat())

  return (
    <ChartContainer title="Confusion Matrix">
      <div className="flex items-center justify-center h-full">
        <div className="inline-block">
          <table className="border-collapse border-2 border-purple-500">
            {data.map((row, i) => (
              <tr key={i}>
                {row.map((value, j) => {
                  const intensity = value / maxValue
                  return (
                    <td
                      key={j}
                      className="w-16 h-16 flex items-center justify-center border border-purple-500/30 font-semibold text-white relative"
                      style={{
                        background: `rgba(139, 92, 246, ${intensity * 0.8})`,
                      }}
                    >
                      {value}
                    </td>
                  )
                })}
              </tr>
            ))}
          </table>
        </div>
      </div>
    </ChartContainer>
  )
}

// Distribution Chart
export function DistributionChart({ data }: { data: Array<{ bin: string; count: number }> }) {
  return (
    <ChartContainer title="Data Distribution">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
        >
          <defs>
            <linearGradient id="colorDistribution" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(139, 92, 246, 0.1)" />
          <XAxis dataKey="bin" stroke="rgba(255, 255, 255, 0.5)" style={{ fontSize: '12px' }} />
          <YAxis stroke="rgba(255, 255, 255, 0.5)" style={{ fontSize: '12px' }} />
          <Tooltip
            contentStyle={{
              background: 'rgba(0, 0, 0, 0.8)',
              border: '1px solid rgba(139, 92, 246, 0.5)',
              borderRadius: '8px',
            }}
          />
          <Area
            type="monotone"
            dataKey="count"
            stroke="#8b5cf6"
            fillOpacity={1}
            fill="url(#colorDistribution)"
            animationDuration={1000}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}
