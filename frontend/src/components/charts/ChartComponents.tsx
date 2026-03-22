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

const MODEL_NAMES: Record<string, string> = {
  xgboost: 'XGBoost', xgb: 'XGBoost',
  random_forest: 'Random Forest', randomforest: 'Random Forest',
  gradient_boosting: 'Gradient Boosting', gradientboosting: 'Gradient Boosting',
  logistic_regression: 'Logistic Regression', logisticregression: 'Logistic Regression',
  linear_regression: 'Linear Regression', linearregression: 'Linear Regression',
  ridge: 'Ridge Regression', lasso: 'Lasso Regression',
  elasticnet: 'ElasticNet', svm: 'SVM', svr: 'SVR', svc: 'SVC',
  knn: 'K-Nearest Neighbors', decision_tree: 'Decision Tree', decisiontree: 'Decision Tree',
  adaboost: 'AdaBoost', lightgbm: 'LightGBM', lgbm: 'LightGBM',
  catboost: 'CatBoost', mlp: 'MLP', naive_bayes: 'Naive Bayes',
  extra_trees: 'Extra Trees', extratrees: 'Extra Trees',
}
function fmtModel(name?: string | null): string {
  if (!name) return '—'
  const key = name.toLowerCase().replace(/[-\s]/g, '_')
  return MODEL_NAMES[key] ?? name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

interface ChartContainerProps {
  title: string
  children: React.ReactNode
  animate?: boolean
  height?: string
}

export function ChartContainer({ title, children, animate = true, height = 'h-64' }: ChartContainerProps) {
  return (
    <motion.div
      initial={animate ? { opacity: 0, scale: 0.95 } : {}}
      whileInView={animate ? { opacity: 1, scale: 1 } : {}}
      transition={{ duration: 0.5 }}
      className="bg-gradient-to-br from-slate-900/50 to-slate-800/50 rounded-xl p-6 border border-purple-500/20 glass glow-hover"
    >
      <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
      <div className={`w-full ${height}`}>{children}</div>
    </motion.div>
  )
}

// Model Comparison Chart — grouped bars: test score + cv score per model, best model highlighted
export function ModelMetricsChart({
  data,
  bestModel,
  taskType = 'classification',
}: {
  data: Array<{ model?: string; model_name?: string; score?: number; metrics?: Record<string, number> }>
  bestModel?: string
  taskType?: string
}) {
  const isClassification = taskType === 'classification'

  const chartData = data.map((r) => {
    const raw = r.model_name ?? r.model ?? 'unknown'
    const name = fmtModel(raw)
    const m = r.metrics ?? {}
    return {
      name,
      isBest: (r.model_name ?? r.model) === bestModel,
      testScore:  parseFloat(((isClassification ? (m.test_accuracy ?? 0) : (m.test_r2 ?? 0)) * 100).toFixed(2)),
      cvScore:    parseFloat(((isClassification ? (m.accuracy ?? 0)      : (m.r2_score ?? 0)) * 100).toFixed(2)),
      f1:         isClassification ? parseFloat(((m.f1_score ?? 0) * 100).toFixed(2)) : undefined,
    }
  })

  const scoreLabel = isClassification ? 'Accuracy (%)' : 'R² Score (%)'

  return (
    <ChartContainer title="All Models Comparison" height="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
          <defs>
            <linearGradient id="testGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#06b6d4" />
              <stop offset="100%" stopColor="#0891b2" />
            </linearGradient>
            <linearGradient id="cvGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#8b5cf6" />
              <stop offset="100%" stopColor="#7c3aed" />
            </linearGradient>
            <linearGradient id="f1Grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="100%" stopColor="#059669" />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.1)" />
          <XAxis
            dataKey="name"
            stroke="rgba(255,255,255,0.5)"
            style={{ fontSize: '10px' }}
            angle={-35}
            textAnchor="end"
            interval={0}
            tick={({ x, y, payload }) => (
              <text
                x={x} y={y + 4}
                textAnchor="end"
                transform={`rotate(-35, ${x}, ${y})`}
                style={{ fontSize: 10, fill: chartData.find(d => d.name === payload.value)?.isBest ? '#f59e0b' : 'rgba(255,255,255,0.6)' }}
              >
                {payload.value}{chartData.find(d => d.name === payload.value)?.isBest ? ' ★' : ''}
              </text>
            )}
          />
          <YAxis
            stroke="rgba(255,255,255,0.5)"
            style={{ fontSize: '11px' }}
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
            label={{ value: scoreLabel, angle: -90, position: 'insideLeft', style: { fill: 'rgba(255,255,255,0.4)', fontSize: 10 } }}
          />
          <Tooltip
            contentStyle={{ background: 'rgba(0,0,0,0.9)', border: '1px solid rgba(139,92,246,0.5)', borderRadius: '8px' }}
            formatter={(value: number, name: string) => [`${value}%`, name]}
            cursor={{ fill: 'rgba(139,92,246,0.08)' }}
          />
          <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
          <Bar dataKey="testScore" name="Test Score" fill="url(#testGrad)" radius={[4,4,0,0]} animationDuration={800}>
            {chartData.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.isBest ? '#f59e0b' : 'url(#testGrad)'}
                stroke={entry.isBest ? '#fbbf24' : 'none'}
                strokeWidth={entry.isBest ? 1.5 : 0}
              />
            ))}
          </Bar>
          <Bar dataKey="cvScore" name="CV Score" fill="url(#cvGrad)" radius={[4,4,0,0]} animationDuration={900} />
          {isClassification && (
            <Bar dataKey="f1" name="F1 Score" fill="url(#f1Grad)" radius={[4,4,0,0]} animationDuration={1000} />
          )}
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
