"use client";
import { motion } from "framer-motion";
import { PipelineResult } from "@/lib/api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
  LineChart, Line, CartesianGrid, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";

const COLORS = ["#3b82f6","#06b6d4","#8b5cf6","#f59e0b","#10b981","#ec4899","#f97316","#14b8a6","#a855f7"];

export default function ResultsDashboard({ result }: { result?: PipelineResult }) {
  if (!result) return (
    <div className="glass-panel p-8 flex flex-col items-center justify-center h-64 text-slate-500 text-sm gap-3">
      <div className="w-12 h-12 rounded-2xl border border-slate-700 flex items-center justify-center text-2xl">📊</div>
      <p>Run the pipeline to see model results</p>
    </div>
  );

  // ── Collect metrics from ALL evaluation results, fallback to best_score ──
  const allResults = result.evaluation_results ?? [];
  const bestResult = allResults.find(r => r.model_name === result.best_model) ?? allResults[0];
  const metrics = bestResult?.metrics ?? {};

  const isClassification = result.task_type === "classification";

  const ML_MODELS = new Set([
    "ridge","lasso","logistic regression","logistic_regression",
    "random forest","random_forest","xgboost","lightgbm",
    "svm","decision tree","decision_tree","knn",
    "gradient boosting","gradient_boosting","extra trees","extra_trees",
  ]);
  const accuracy = metrics.accuracy ?? metrics.train_accuracy ?? null;
  const f1       = metrics.f1_score ?? metrics.f1 ?? null;
  const rocAuc   = metrics.roc_auc ?? result.roc_data?.auc ?? null;

  // ── Regression metrics ──
  const r2   = metrics.r2_score ?? null;
  const rmse = metrics.rmse ?? null;
  const mae  = metrics.mae  ?? null;

  // ── Metric cards — switch by task type ──
  const metricCards = isClassification ? [
    { label: "Accuracy",   value: accuracy,          color: "#3b82f6", fmt: (v: number) => (v*100).toFixed(1)+"%" },
    { label: "F1 Score",   value: f1,                color: "#06b6d4", fmt: (v: number) => (v*100).toFixed(1)+"%" },
    { label: "ROC-AUC",    value: rocAuc,            color: "#8b5cf6", fmt: (v: number) => v.toFixed(3) },
    { label: "Best Score", value: result.best_score, color: "#10b981", fmt: (v: number) => (v*100).toFixed(1)+"%" },
  ] : [
    { label: "R² Score",   value: r2,                color: "#3b82f6", fmt: (v: number) => v.toFixed(4) },
    { label: "RMSE",       value: rmse,              color: "#f59e0b", fmt: (v: number) => v.toFixed(4) },
    { label: "MAE",        value: mae,               color: "#8b5cf6", fmt: (v: number) => v.toFixed(4) },
    { label: "Best Score", value: result.best_score, color: "#10b981", fmt: (v: number) => v.toFixed(4) },
  ];

  // ── Radar — switch by task type ──
  const radarData = isClassification ? [
    { metric: "Accuracy", value: accuracy != null ? +(accuracy*100).toFixed(1) : 0 },
    { metric: "F1 Score", value: f1       != null ? +(f1*100).toFixed(1)       : 0 },
    { metric: "ROC-AUC",  value: rocAuc   != null ? +(rocAuc*100).toFixed(1)   : 0 },
  ] : [
    // For regression: R² as %, RMSE/MAE normalized to 0-100 via sigmoid-like clamp
    { metric: "R²",    value: r2   != null ? +(Math.max(0, Math.min(1, r2))   * 100).toFixed(1) : 0 },
    { metric: "RMSE",  value: rmse != null ? +(Math.max(0, Math.min(100, rmse))).toFixed(1)     : 0 },
    { metric: "MAE",   value: mae  != null ? +(Math.max(0, Math.min(100, mae))).toFixed(1)      : 0 },
  ];

  // ── Bar chart — switch metric keys by task type ──
  const modelBarData = allResults
    .filter(r => ML_MODELS.has(r.model_name?.toLowerCase().replace(/_/g," ")) || ML_MODELS.has(r.model_name?.toLowerCase()))
    .map(r => isClassification ? ({
      name:       r.model_name.replace(/_/g," ").replace(/\b\w/g,(c:string)=>c.toUpperCase()),
      Accuracy:   r.metrics?.accuracy  != null ? parseFloat((r.metrics.accuracy *100).toFixed(1)) : null,
      "F1 Score": r.metrics?.f1_score  != null ? parseFloat((r.metrics.f1_score *100).toFixed(1))
                : r.metrics?.f1        != null ? parseFloat((r.metrics.f1       *100).toFixed(1)) : null,
      "ROC-AUC":  r.metrics?.roc_auc   != null ? parseFloat((r.metrics.roc_auc  *100).toFixed(1)) : null,
    }) : ({
      name:     r.model_name.replace(/_/g," ").replace(/\b\w/g,(c:string)=>c.toUpperCase()),
      "R²":     r.metrics?.r2_score != null ? parseFloat((r.metrics.r2_score*100).toFixed(1)) : null,
      "RMSE":   r.metrics?.rmse     != null ? parseFloat(r.metrics.rmse.toFixed(4))           : null,
      "MAE":    r.metrics?.mae      != null ? parseFloat(r.metrics.mae.toFixed(4))            : null,
    }));

  const barKeys = isClassification
    ? [
        { key: "Accuracy",  color: "#3b82f6" },
        { key: "F1 Score",  color: "#06b6d4" },
        { key: "ROC-AUC",   color: "#8b5cf6" },
      ]
    : [
        { key: "R²",   color: "#3b82f6" },
        { key: "RMSE", color: "#f59e0b" },
        { key: "MAE",  color: "#8b5cf6" },
      ];

  // ── ROC curve ──
  const fprArr: number[] = result.roc_data?.fpr ?? [];
  const tprArr: number[] = result.roc_data?.tpr ?? [];
  const rocData = result.roc_data?.type === "binary" && fprArr.length
    ? fprArr.map((x, i) => ({ fpr: +x.toFixed(3), tpr: +(tprArr[i] ?? 0).toFixed(3) }))
    : null;

  // ── SHAP ──
  const shapFeatures: string[] = result.shap_values?.feature_names ?? [];
  const shapScores: number[]   = result.shap_values?.mean_abs_shap  ?? [];
  const shapData = shapFeatures.slice(0, 10).map((f: string, i: number) => ({
    feature:    f.length > 14 ? f.slice(0, 14) + "…" : f,
    importance: +(shapScores[i] ?? 0).toFixed(4),
  }));

  return (
    <div className="space-y-5">

      {/* ── Metric cards ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {metricCards.map((m, i) => (
          <motion.div key={m.label}
            initial={{ opacity:0, y:16 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*.06 }}
            className="glass-panel p-4 text-center relative overflow-hidden"
            style={{ borderColor:`${m.color}30`, boxShadow:`0 0 20px ${m.color}15` }}>
            <div className="absolute inset-0 opacity-5 pointer-events-none"
              style={{ background:`radial-gradient(circle at 50% 0%, ${m.color}, transparent 70%)` }} />
            <p className="text-xs text-slate-500 mb-2">{m.label}</p>
            <p className="text-2xl font-bold tabular-nums" style={{ color: m.color }}>
              {m.value != null ? m.fmt(m.value) : <span className="text-slate-600 text-lg">N/A</span>}
            </p>
          </motion.div>
        ))}
      </div>

      {/* ── Best model banner ── */}
      <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:.3 }}
        className="glass-panel p-4 flex items-center gap-4"
        style={{ borderColor:"#10b98130", boxShadow:"0 0 20px #10b98115" }}>
        <div className="w-10 h-10 rounded-xl bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center text-lg">🏆</div>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-slate-500">Best Model</p>
          <p className="text-base font-bold text-emerald-400 capitalize">{result.best_model?.replace(/_/g," ") ?? "—"}</p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`text-[10px] px-2 py-0.5 rounded-full border font-medium ${
            isClassification
              ? "border-blue-500/30 bg-blue-500/10 text-blue-400"
              : "border-purple-500/30 bg-purple-500/10 text-purple-400"
          }`}>
            {isClassification ? "Classification" : "Regression"}
          </span>
          <div className="text-right">
            <p className="text-xs text-slate-500">Score</p>
            <p className="text-xl font-bold text-emerald-400">
              {result.best_score != null
                ? isClassification
                  ? (result.best_score * 100).toFixed(1) + "%"
                  : result.best_score.toFixed(4)
                : "—"}
            </p>
          </div>
        </div>
      </motion.div>

      {/* ── Triangle Radar + Model Comparison side by side ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        {/* Triangle Radar */}
        <motion.div initial={{ opacity:0, x:-20 }} animate={{ opacity:1, x:0 }} transition={{ delay:.35 }}
          className="glass-panel p-5">
          <p className="text-sm font-semibold text-slate-300 mb-1">Metrics Triangle</p>
          <p className="text-xs text-slate-500 mb-4">Comparative view of top 3 metrics</p>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius={80}>
              <PolarGrid stroke="rgba(99,179,237,0.12)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill:"#94a3b8", fontSize:11, fontWeight:500 }} />
              <PolarRadiusAxis angle={90} domain={isClassification ? [0,100] : [0,"auto"]} tick={{ fill:"#475569", fontSize:9 }} tickCount={4} />
              <Radar name="Score" dataKey="value" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.18} strokeWidth={2}
                dot={{ fill:"#06b6d4", r:4, strokeWidth:0 }} />
              <Tooltip
                contentStyle={{ background:"rgba(10,22,40,.95)", border:"1px solid rgba(99,179,237,.2)", borderRadius:8, fontSize:12 }}
                formatter={(v: number) => [isClassification ? `${v}%` : v, "Score"]} />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* All models bar chart */}
        <motion.div initial={{ opacity:0, x:20 }} animate={{ opacity:1, x:0 }} transition={{ delay:.4 }}
          className="glass-panel p-5">
          <p className="text-sm font-semibold text-slate-300 mb-1">All Models Comparison</p>
          <p className="text-xs text-slate-500 mb-4">Score (%) across every evaluated model</p>
          {modelBarData.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={modelBarData} margin={{ top:4, right:4, left:-18, bottom:36 }} barCategoryGap="20%" barGap={2}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.07)" />
                <XAxis dataKey="name" tick={{ fontSize:9, fill:"#64748b" }} angle={-35} textAnchor="end" interval={0} />
                <YAxis tick={{ fontSize:10, fill:"#64748b" }}
                  domain={isClassification ? [0,100] : ["auto","auto"]}
                  unit={isClassification ? "%" : ""} />
                <Tooltip
                  contentStyle={{ background:"rgba(10,22,40,.95)", border:"1px solid rgba(99,179,237,.2)", borderRadius:8, fontSize:12 }}
                  formatter={(v: number) => [isClassification ? `${v}%` : v]} />
                <Legend wrapperStyle={{ fontSize:10, color:"#94a3b8", paddingTop:8 }} />
                {barKeys.map(({ key, color }) => (
                  <Bar key={key} dataKey={key} fill={color} radius={[4,4,0,0]} maxBarSize={18} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[240px] flex items-center justify-center text-slate-600 text-sm">No model data yet</div>
          )}
        </motion.div>
      </div>

      {/* ── ROC Curve ── */}
      {rocData && (
        <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:.45 }}
          className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4">
            <p className="text-sm font-semibold text-slate-300">ROC Curve</p>
            <span className="text-xs px-2.5 py-1 rounded-full bg-cyan-500/15 text-cyan-400 border border-cyan-500/25">
              AUC = {result.roc_data?.auc?.toFixed(3)}
            </span>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={rocData} margin={{ top:0, right:0, left:-20, bottom:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.07)" />
              <XAxis dataKey="fpr" tick={{ fontSize:10, fill:"#64748b" }} label={{ value:"FPR", position:"insideBottomRight", offset:-4, fill:"#64748b", fontSize:10 }} />
              <YAxis tick={{ fontSize:10, fill:"#64748b" }} label={{ value:"TPR", angle:-90, position:"insideLeft", fill:"#64748b", fontSize:10 }} />
              <Tooltip contentStyle={{ background:"rgba(10,22,40,.95)", border:"1px solid rgba(99,179,237,.2)", borderRadius:8, fontSize:12 }} />
              <Line type="monotone" dataKey="tpr" stroke="#06b6d4" dot={false} strokeWidth={2.5} name="TPR" />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* ── SHAP ── */}
      {shapData.length > 0 && (
        <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:.5 }}
          className="glass-panel p-5">
          <p className="text-sm font-semibold text-slate-300 mb-4">Feature Importance (SHAP)</p>
          <ResponsiveContainer width="100%" height={Math.max(160, shapData.length * 22)}>
            <BarChart data={shapData} layout="vertical" margin={{ top:0, right:16, left:0, bottom:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.07)" />
              <XAxis type="number" tick={{ fontSize:10, fill:"#64748b" }} />
              <YAxis dataKey="feature" type="category" tick={{ fontSize:10, fill:"#94a3b8" }} width={100} />
              <Tooltip contentStyle={{ background:"rgba(10,22,40,.95)", border:"1px solid rgba(99,179,237,.2)", borderRadius:8, fontSize:12 }} />
              <Bar dataKey="importance" radius={[0,5,5,0]}>
                {shapData.map((_: unknown, i: number) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      )}
    </div>
  );
}
