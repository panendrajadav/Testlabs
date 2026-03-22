'use client'

import { motion } from 'framer-motion'
import { usePipelineStatus } from '@/hooks/useApi'
import { useStoredDataset } from '@/hooks/useStoredDataset'
import { containerVariants, itemVariants } from '@/animations/variants'
import { ShimmerLoader } from '@/components/loaders/LoadingAnimations'
import { ModelMetricsChart, ROCCurveChart, FeatureImportanceChart } from '@/components/charts/ChartComponents'
import { AlertTriangle, CheckCircle, Info } from 'lucide-react'
import type { PipelineStatus } from '@/types'

const OVERFIT_WARN  = 0.05
const OVERFIT_ALERT = 0.15

// Proper display names for known model identifiers
const MODEL_NAMES: Record<string, string> = {
  xgboost:             'XGBoost',
  xgb:                 'XGBoost',
  random_forest:       'Random Forest',
  randomforest:        'Random Forest',
  gradient_boosting:   'Gradient Boosting',
  gradientboosting:    'Gradient Boosting',
  logistic_regression: 'Logistic Regression',
  logisticregression:  'Logistic Regression',
  linear_regression:   'Linear Regression',
  linearregression:    'Linear Regression',
  ridge:               'Ridge Regression',
  lasso:               'Lasso Regression',
  elasticnet:          'ElasticNet',
  svm:                 'SVM',
  svr:                 'SVR',
  svc:                 'SVC',
  knn:                 'K-Nearest Neighbors',
  decision_tree:       'Decision Tree',
  decisiontree:        'Decision Tree',
  adaboost:            'AdaBoost',
  lightgbm:            'LightGBM',
  lgbm:                'LightGBM',
  catboost:            'CatBoost',
  mlp:                 'MLP',
  naive_bayes:         'Naive Bayes',
  naivebayes:          'Naive Bayes',
  extra_trees:         'Extra Trees',
  extratrees:          'Extra Trees',
}

function fmtModel(name: string | null | undefined): string {
  if (!name) return '—'
  const key = name.toLowerCase().replace(/[-\s]/g, '_')
  return MODEL_NAMES[key] ?? name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function OverfitBadge({ gap }: { gap: number }) {
  if (gap <= OVERFIT_WARN) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-green-500/20 text-green-400 border border-green-500/30">
        <CheckCircle size={12} /> Good generalisation
      </span>
    )
  }
  if (gap <= OVERFIT_ALERT) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-yellow-500/20 text-yellow-400 border border-yellow-500/30">
        <AlertTriangle size={12} /> Mild overfitting (gap {(gap * 100).toFixed(1)}%)
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-red-500/20 text-red-400 border border-red-500/30">
      <AlertTriangle size={12} /> Severe overfitting (gap {(gap * 100).toFixed(1)}%)
    </span>
  )
}

export default function ModelsPage() {
  const { datasetId, mounted } = useStoredDataset()

  const { data: job, isLoading } = usePipelineStatus(datasetId)
  const status = job as PipelineStatus | undefined
  const result = status?.result

  if (!mounted) return null

  if (!datasetId) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-400">Please upload a dataset first</p>
      </div>
    )
  }

  if (isLoading || status?.status === 'running' || status?.status === 'queued') {
    return (
      <div className="min-h-screen bg-black p-8">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="bg-slate-900/50 rounded-xl p-6"><ShimmerLoader count={3} /></div>
          ))}
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-400">No results yet. Run the pipeline first.</p>
      </div>
    )
  }

  const isClassification = result.task_type === 'classification'
  const evalResults = result.evaluation_results ?? []
  const bestResult = evalResults.find((r) => (r.model_name ?? r.model) === result.best_model)
  const metrics = bestResult?.metrics ?? {}
  const overfitGap = (metrics.overfit_gap as number) ?? 0
  const testSetSize = (metrics.test_set_size as number) ?? null
  const trainScore = (metrics.train_accuracy ?? metrics.train_r2 ?? 0) as number
  const testScore  = (metrics.test_accuracy  ?? metrics.test_r2  ?? 0) as number
  const isUnreliable = testSetSize !== null && testSetSize < 30
  const isLuckySplit = testScore > trainScore + 0.01
  const justification = (result as any).justification as string | undefined
  const isUnderfit = (result as any).is_underfit as boolean | undefined
  const bestParams = (metrics.best_params ?? result.best_params ?? {}) as Record<string, any>

  // Format a metric value based on its type
  const fmt = (value: number | null | undefined, type: 'score' | 'error' | 'std') => {
    if (value == null) return 'N/A'
    // For all metrics, use consistent high precision internally, but display as requested:
    if (type === 'score') return value.toFixed(4)  // R², accuracy scores
    if (type === 'error') {
      // Error metrics (MAE, RMSE): always 2 decimal places for clarity
      return value.toFixed(2)
    }
    // Standard deviation: 4 decimals for precision
    return value.toFixed(4)
  }

  // Primary metric cards
  const metricCards = isClassification
    ? [
        { label: 'CV Accuracy',    value: metrics.accuracy,      hint: 'Cross-validated',      type: 'score' as const },
        { label: 'Test Accuracy',  value: metrics.test_accuracy,  hint: 'Held-out 20%',         type: 'score' as const },
        { label: 'Train Accuracy', value: metrics.train_accuracy, hint: 'Training set',          type: 'score' as const },
        { label: 'F1 Score',       value: metrics.f1_score,       hint: 'Weighted, test set',   type: 'score' as const },
        { label: 'ROC-AUC',        value: metrics.roc_auc,        hint: 'Test set',              type: 'score' as const },
        { label: 'CV Std Dev',     value: metrics.cv_std,         hint: 'Lower = more stable',  type: 'std'   as const },
      ]
    : [
        { label: 'CV R²',      value: metrics.r2_score, hint: 'Cross-validated',    type: 'score' as const },
        { label: 'Test R²',    value: metrics.test_r2,  hint: 'Held-out 20%',       type: 'score' as const },
        { label: 'Train R²',   value: metrics.train_r2, hint: 'Training set',        type: 'score' as const },
        { label: 'MAE',        value: metrics.mae,      hint: 'Avg error (same unit as target)', type: 'error' as const },
        { label: 'RMSE',       value: metrics.rmse,     hint: 'Avg squared error (same unit as target)', type: 'error' as const },
        { label: 'CV Std Dev', value: metrics.cv_std,   hint: 'Lower = more stable', type: 'std'   as const },
      ]

  const rocPoints = (() => {
    const roc = result.roc_data
    if (!roc) return []
    if (roc.fpr && roc.tpr) return roc.fpr.map((fpr, i) => ({ fpr, tpr: roc.tpr![i] }))
    // multiclass: use first class curve
    if (roc.curves) {
      const firstKey = Object.keys(roc.curves)[0]
      if (firstKey) {
        const c = roc.curves[firstKey]
        return c.fpr.map((fpr, i) => ({ fpr, tpr: c.tpr[i] }))
      }
    }
    return []
  })()

  const shapData = result.shap_values
    ? result.shap_values.feature_names.map((f, i) => ({ feature: f, importance: result.shap_values!.mean_abs_shap[i] }))
    : []

  return (
    <motion.div initial="hidden" animate="visible" variants={containerVariants} className="min-h-screen bg-black p-8">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <motion.div variants={itemVariants} className="mb-8">
          <h1 className="text-4xl font-bold text-gradient mb-1">Model Results</h1>
          <div className="flex flex-wrap items-center gap-3 mt-2">
            <p className="text-gray-400">
              Best model: <span className="text-white font-semibold">{fmtModel(result.best_model)}</span>
              {' '}— CV Score: <span className="text-cyan-400 font-semibold">{result.best_score != null ? result.best_score.toFixed(4) : 'N/A'}</span>
            </p>
            <OverfitBadge gap={overfitGap} />
          </div>
        </motion.div>

        {/* Small / lucky test set warning */}
        {(isUnreliable || isLuckySplit) && (
          <motion.div variants={itemVariants}
            className="rounded-xl border border-blue-500/40 bg-blue-500/10 p-4 mb-8 glass flex gap-3">
            <Info className="text-blue-400 shrink-0 mt-0.5" size={18} />
            <div>
              <p className="font-semibold text-sm text-blue-300">Results May Be Unreliable</p>
              <p className="text-gray-400 text-sm mt-1">
                {isLuckySplit
                  ? <>Test score (<span className="text-white font-medium">{testScore.toFixed(4)}</span>) exceeds train score (<span className="text-white font-medium">{trainScore.toFixed(4)}</span>) — this usually means the test set is too small to be statistically meaningful.</>
                  : <>Test set contains only <span className="text-white font-medium">{testSetSize} rows</span> — scores may vary significantly on a different split.</>
                }
                {testSetSize !== null && <> Upload a larger dataset (200+ rows) for reliable evaluation.</>}
              </p>
            </div>
          </motion.div>
        )}

        {/* Overfitting warning banner */}
        {overfitGap > OVERFIT_WARN && (
          <motion.div variants={itemVariants}
            className={`rounded-xl border p-4 mb-8 glass flex gap-3 ${
              overfitGap > OVERFIT_ALERT
                ? 'border-red-500/40 bg-red-500/10'
                : 'border-yellow-500/40 bg-yellow-500/10'
            }`}>
            <AlertTriangle className={overfitGap > OVERFIT_ALERT ? 'text-red-400 shrink-0 mt-0.5' : 'text-yellow-400 shrink-0 mt-0.5'} size={18} />
            <div>
              <p className={`font-semibold text-sm ${overfitGap > OVERFIT_ALERT ? 'text-red-300' : 'text-yellow-300'}`}>
                {overfitGap > OVERFIT_ALERT ? 'Severe Overfitting Detected' : 'Mild Overfitting Detected'}
              </p>
              <p className="text-gray-400 text-sm mt-1">
                The model scores <span className="text-white font-medium">{(metrics.train_accuracy ?? metrics.train_r2 ?? 0).toFixed(4)}</span> on
                training data but only <span className="text-white font-medium">{(metrics.test_accuracy ?? metrics.test_r2 ?? 0).toFixed(4)}</span> on
                the held-out test set — a gap of <span className="text-white font-medium">{(overfitGap * 100).toFixed(1)}%</span>.
                The model has memorised training patterns and may generalise poorly to new data.
                {overfitGap > OVERFIT_ALERT && ' Consider collecting more data, reducing model complexity, or adding regularisation.'}
              </p>
            </div>
          </motion.div>
        )}

        {/* Underfitting banner */}
        {isUnderfit && (
          <motion.div variants={itemVariants}
            className="rounded-xl border border-orange-500/40 bg-orange-500/10 p-4 mb-8 glass flex gap-3">
            <AlertTriangle className="text-orange-400 shrink-0 mt-0.5" size={18} />
            <div>
              <p className="font-semibold text-sm text-orange-300">Underfitting Detected</p>
              <p className="text-gray-400 text-sm mt-1">
                Both CV score (<span className="text-white font-medium">{(metrics.accuracy ?? metrics.r2_score ?? 0).toFixed(4)}</span>) and
                test score (<span className="text-white font-medium">{testScore.toFixed(4)}</span>) are below 0.60.
                The model is too simple for this data. Consider adding more features or using a more complex model.
              </p>
            </div>
          </motion.div>
        )}

        {/* Justification card */}
        {justification && (
          <motion.div variants={itemVariants}
            className="rounded-xl border border-purple-500/30 bg-purple-500/5 p-5 mb-8 glass">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle size={16} className="text-purple-400" />
              <h3 className="text-white font-semibold text-sm">Why This Model Was Selected</h3>
            </div>
            <p className="text-gray-300 text-sm leading-relaxed">{justification?.replace(new RegExp(Object.keys(MODEL_NAMES).join('|'), 'gi'), m => fmtModel(m))}</p>
            {Object.keys(bestParams).length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {Object.entries(bestParams).map(([k, v]) => (
                  <span key={k} className="px-2 py-1 rounded-md bg-slate-800 border border-purple-500/20 text-xs font-mono text-cyan-300">
                    {k}: <span className="text-white">{String(v)}</span>
                  </span>
                ))}
              </div>
            )}
          </motion.div>
        )}

        {/* Metric cards */}
        <motion.div variants={containerVariants} className="mb-10">
          <div className="flex items-center gap-2 mb-4">
            <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold bg-gradient-to-r from-purple-500/20 to-cyan-500/20 border border-purple-500/30 text-purple-300">
              {isClassification ? 'Classification Metrics' : 'Regression Performance Metrics'}
            </span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {metricCards.map((m, i) => (
              <motion.div key={m.label} variants={itemVariants} whileHover={{ y: -4 }}
                className={`rounded-xl border glass p-4 transition-colors ${
                  m.type === 'error'
                    ? 'bg-rose-900/20 border-rose-500/30 hover:border-rose-500/50'
                    : m.type === 'std'
                    ? 'bg-amber-900/20 border-amber-500/30 hover:border-amber-500/50'
                    : 'bg-slate-900/50 border-purple-500/20 hover:border-purple-500/40'
                }`}>
                <p className={`text-xs mb-1 ${
                  m.type === 'error' ? 'text-rose-400' : m.type === 'std' ? 'text-amber-400' : 'text-gray-400'
                }`}>{m.label}</p>
                <p className={`font-bold text-2xl ${
                  m.type === 'error' ? 'text-rose-300' : m.type === 'std' ? 'text-amber-300' : 'text-purple-400'
                }`}>
                  {fmt(m.value as number | null | undefined, m.type)}
                </p>
                <p className={`text-xs mt-1 ${
                  m.type === 'error' ? 'text-rose-600' : m.type === 'std' ? 'text-amber-600' : 'text-gray-600'
                }`}>{m.hint}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Regression Error Metrics Explanation */}
        {!isClassification && metrics.mae != null && metrics.rmse != null && (() => {
          const mae  = metrics.mae  as number
          const rmse = metrics.rmse as number
          const targetMean: number | null = (() => {
            try {
              // target_distribution plot has raw x values for regression
              const plot = (result as any).eda_plots?.target_distribution
              if (plot?.x && Array.isArray(plot.x) && plot.x.length > 0) {
                const vals = plot.x as number[]
                return vals.reduce((a: number, b: number) => a + b, 0) / vals.length
              }
              return null
            } catch { return null }
          })()

          const maePct  = targetMean ? (mae  / Math.abs(targetMean)) * 100 : null
          const rmsePct = targetMean ? (rmse / Math.abs(targetMean)) * 100 : null
          const rmseVsMae = rmse / mae

          // Verdict based on MAE % of mean
          const verdict = maePct == null ? null
            : maePct < 5   ? { label: 'Excellent', color: 'text-green-400',  bg: 'bg-green-500/10  border-green-500/30',  emoji: '🟢' }
            : maePct < 10  ? { label: 'Good',      color: 'text-cyan-400',   bg: 'bg-cyan-500/10   border-cyan-500/30',   emoji: '🟢' }
            : maePct < 20  ? { label: 'Acceptable', color: 'text-yellow-400', bg: 'bg-yellow-500/10 border-yellow-500/30', emoji: '🟡' }
            :                { label: 'Needs Work', color: 'text-red-400',    bg: 'bg-red-500/10    border-red-500/30',    emoji: '🔴' }

          const outlierNote = rmseVsMae > 1.5
            ? 'RMSE is significantly higher than MAE — this means a few predictions are very far off (large outlier errors). The model is mostly accurate but occasionally makes big mistakes.'
            : 'RMSE is close to MAE — errors are consistent with no extreme outliers. The model makes similar-sized mistakes across all predictions.'

          return (
            <motion.div variants={itemVariants}
              className={`rounded-xl border p-5 mb-8 glass ${
                verdict ? verdict.bg : 'bg-cyan-500/5 border-cyan-500/30'
              }`}>
              <div className="flex items-start gap-3">
                <Info size={18} className="text-cyan-400 shrink-0 mt-0.5" />
                <div className="w-full">
                  <div className="flex items-center gap-3 mb-3">
                    <h3 className="text-white font-semibold text-sm">What do MAE & RMSE mean for your model?</h3>
                    {verdict && (
                      <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${verdict.bg} ${verdict.color}`}>
                        {verdict.emoji} {verdict.label}
                      </span>
                    )}
                  </div>

                  <div className="space-y-3 text-sm">
                    {/* MAE plain English */}
                    <div className="p-3 rounded-lg bg-black/30">
                      <p className="text-rose-300 font-medium mb-1">MAE = {mae.toFixed(2)}</p>
                      <p className="text-gray-300">
                        On average, your model's prediction is off by <span className="text-white font-semibold">{mae.toFixed(2)}</span> units from the real value.
                        {maePct != null && (
                          <> That's roughly <span className={`font-semibold ${
                            maePct < 10 ? 'text-green-400' : maePct < 20 ? 'text-yellow-400' : 'text-red-400'
                          }`}>{maePct.toFixed(1)}% of the average target value</span> ({targetMean!.toFixed(2)}).
                          {maePct < 10
                            ? ' This is a small error — the model is predicting quite accurately.'
                            : maePct < 20
                            ? ' This is a moderate error — acceptable for most use cases.'
                            : ' This is a large error — the model may need more data or better features.'}
                          </>
                        )}
                      </p>
                    </div>

                    {/* RMSE plain English */}
                    <div className="p-3 rounded-lg bg-black/30">
                      <p className="text-rose-300 font-medium mb-1">RMSE = {rmse.toFixed(2)}</p>
                      <p className="text-gray-300">
                        RMSE punishes big mistakes more than small ones.
                        {rmsePct != null && (
                          <> At <span className={`font-semibold ${
                            rmsePct < 10 ? 'text-green-400' : rmsePct < 20 ? 'text-yellow-400' : 'text-red-400'
                          }`}>{rmsePct.toFixed(1)}% of the average target</span>, </>)}
                        {' '}{outlierNote}
                      </p>
                    </div>

                    {/* Simple analogy */}
                    <div className="p-3 rounded-lg bg-purple-500/10 border border-purple-500/20">
                      <p className="text-purple-300 font-medium mb-1">💡 Simple analogy</p>
                      <p className="text-gray-300">
                        Think of MAE like the average GPS error — if your GPS is off by {mae.toFixed(0)} metres on average, that's your MAE.
                        RMSE is like the worst-case GPS error — it's higher because a few very wrong predictions pull it up.
                        {maePct != null && maePct < 10 && ' Your model is like a GPS that\'s almost always spot-on.'}
                        {maePct != null && maePct >= 10 && maePct < 20 && ' Your model is like a GPS that\'s usually close but sometimes a street off.'}
                        {maePct != null && maePct >= 20 && ' Your model is like a GPS that sometimes takes you to the wrong neighbourhood.'}
                      </p>
                    </div>

                    {/* Why are they high? */}
                    {maePct != null && maePct >= 10 && (
                      <div className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                        <p className="text-yellow-300 font-medium mb-1">Why might the error be high?</p>
                        <ul className="text-gray-300 space-y-1 list-disc list-inside">
                          <li>The target variable has a wide range — larger values naturally produce larger absolute errors</li>
                          <li>Some important features may be missing from the dataset</li>
                          <li>The relationship between features and target may be non-linear</li>
                          <li>More training data usually reduces error over time</li>
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          )
        })()}

        {/* Train vs CV comparison bar */}
        <motion.div variants={itemVariants} className="bg-slate-900/50 rounded-xl border border-purple-500/20 glass p-6 mb-8">
          <div className="flex items-center gap-2 mb-4">
            <Info size={16} className="text-purple-400" />
            <h3 className="text-white font-semibold">Train vs CV Score Comparison</h3>
          </div>
          <div className="space-y-3">
            {[
              { label: 'Train Score', value: (metrics.train_accuracy ?? metrics.train_r2 ?? 0) as number, color: 'bg-orange-500' },
              { label: 'CV Score',    value: (metrics.accuracy ?? metrics.r2_score ?? 0) as number,        color: 'bg-purple-500' },
              { label: 'Test Score',  value: (metrics.test_accuracy ?? metrics.test_r2 ?? 0) as number,    color: 'bg-cyan-500' },
              { label: 'CV Std Dev',  value: (metrics.cv_std ?? 0) as number,                              color: 'bg-rose-500', scale: 5 },
            ].map((bar) => (
              <div key={bar.label} className="flex items-center gap-3">
                <span className="text-gray-400 text-sm w-24 shrink-0">{bar.label}</span>
                <div className="flex-1 bg-gray-800 rounded-full h-3 overflow-hidden">
                  <motion.div
                    className={`h-full ${bar.color} rounded-full`}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(0, Math.min(100, (bar.value ?? 0) * (bar.scale ?? 1) * 100))}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                  />
                </div>
                <span className="text-white text-sm font-medium w-16 text-right">{(bar.value ?? 0).toFixed(4)}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Charts */}
        <motion.div variants={containerVariants} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <motion.div variants={itemVariants} className="lg:col-span-2">
            <ModelMetricsChart
              data={evalResults}
              bestModel={result.best_model ?? undefined}
              taskType={result.task_type ?? 'classification'}
            />
          </motion.div>
          {isClassification && rocPoints.length > 0 && (
            <motion.div variants={itemVariants}>
              <ROCCurveChart data={rocPoints} />
            </motion.div>
          )}
          {shapData.length > 0 && (
            <motion.div variants={itemVariants}>
              <FeatureImportanceChart data={shapData} />
            </motion.div>
          )}
        </motion.div>

      </div>
    </motion.div>
  )
}
