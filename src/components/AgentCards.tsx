"use client";
import { motion } from "framer-motion";
import { PipelineResult } from "@/lib/api";
import { BarChart2, Cpu, Filter, GitBranch, Sliders, Trophy } from "lucide-react";

const AGENTS = [
  { key:"eda",   label:"EDA Agent",                 logPrefix:"EDA",          icon: BarChart2, color:"#3b82f6", desc:"Analyzes dataset statistics, distributions, and generates insights." },
  { key:"pre",   label:"Preprocessing Agent",       logPrefix:"Preprocessing", icon: Filter,    color:"#06b6d4", desc:"Handles missing values, encoding, scaling, and class imbalance." },
  { key:"feat",  label:"Feature Engineering Agent", logPrefix:"Feature",       icon: GitBranch, color:"#8b5cf6", desc:"Selects the most informative features using statistical tests." },
  { key:"model", label:"Model Selection Agent",     logPrefix:"Model",         icon: Cpu,       color:"#f59e0b", desc:"Picks the best algorithm based on task type and dataset size." },
  { key:"hpo",   label:"HPO Agent",                 logPrefix:"Hyperparameter",icon: Sliders,   color:"#10b981", desc:"Optimizes hyperparameters using Optuna with parallel trials." },
  { key:"eval",  label:"Evaluation Agent",          logPrefix:"Evaluation",    icon: Trophy,    color:"#ec4899", desc:"Computes accuracy, F1, ROC-AUC, and SHAP feature importance." },
];

const stageIndex = (progress?: string) => {
  if (!progress) return -1;
  const p = progress.toLowerCase();
  if (p.includes("eda"))                        return 0;
  if (p.includes("preprocess"))                 return 1;
  if (p.includes("feature"))                    return 2;
  if (p.includes("model select") || p === "model selection") return 3;
  if (p.includes("hyper") || p.includes("hpo")) return 4;
  if (p.includes("eval"))                       return 5;
  if (p.includes("done") || p.includes("complet")) return 6;
  return -1;
};

interface Props { result?: PipelineResult; currentStatus?: string; progress?: string; }

export default function AgentCards({ result, currentStatus, progress }: Props) {
  const activeIdx = currentStatus === "completed" ? 6 : stageIndex(progress);
  const allLogs: string[] = result?.agent_logs ?? [];

  // HPO+Eval are looping — detect if we're in the loop phase
  const isLooping = activeIdx >= 3 && currentStatus === "running";

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {AGENTS.map((agent, i) => {
        const Icon  = agent.icon;
        const done   = activeIdx > i;
        const active = activeIdx === i;
        const pending = activeIdx < i;

        // Filter logs belonging to this agent
        const agentLogs = allLogs.filter(l =>
          l.toLowerCase().includes(agent.logPrefix.toLowerCase())
        ).slice(-3); // last 3 relevant logs

        // Decide whether to show stats or logs
        const hasStats = done && result && (i === 0 || i === 2 || i === 5);

        // Loop badge: HPO (4) and Eval (5) loop multiple times
        const showLoopBadge = (i === 3 || i === 4 || i === 5) && (active || done) && isLooping;

        return (
          <motion.div key={agent.key}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
            whileHover={{ y: -4, scale: 1.02 }}
            className="glass-panel p-5 transition-all duration-300 relative overflow-hidden"
            style={{
              boxShadow: active ? `0 0 30px ${agent.color}33` : done ? `0 0 12px ${agent.color}18` : "none",
              borderColor: active ? `${agent.color}55` : done ? `${agent.color}33` : "rgba(99,179,237,0.12)",
            }}
          >
            {(active || done) && (
              <div className="absolute inset-0 opacity-5 pointer-events-none"
                style={{ background: `radial-gradient(circle at 30% 30%, ${agent.color}, transparent 70%)` }} />
            )}

            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg" style={{ background: `${agent.color}18` }}>
                  <Icon className="w-5 h-5" style={{ color: agent.color }} />
                </div>
                <div>
                  <p className="font-semibold text-sm text-slate-100">{agent.label}</p>
                  <p className="text-xs text-slate-500 mt-0.5">{agent.desc}</p>
                </div>
              </div>
              <div className="flex flex-col items-end gap-1">
                <StatusDot done={done} active={active} color={agent.color} />
                {showLoopBadge && (
                  <motion.span
                    className="text-[9px] px-1.5 py-0.5 rounded-full border"
                    style={{ borderColor: `${agent.color}40`, color: agent.color, background: `${agent.color}10` }}
                    animate={{ opacity: [1, 0.4, 1] }}
                    transition={{ duration: 1.2, repeat: Infinity }}
                  >
                    ↻ looping
                  </motion.span>
                )}
              </div>
            </div>

            {/* ── EDA stats ── */}
            {hasStats && i === 0 && (
              <div className="mt-3 grid grid-cols-2 gap-2">
                <Stat label="Rows"   value={result!.eda_summary?.n_rows}    color={agent.color} />
                <Stat label="Cols"   value={result!.eda_summary?.n_cols}    color={agent.color} />
                <Stat label="Task"   value={result!.task_type}              color={agent.color} />
                <Stat label="Target" value={result!.target_column}          color={agent.color} />
              </div>
            )}

            {/* ── Feature Engineering stats ── */}
            {hasStats && i === 2 && (
              <div className="mt-3">
                <Stat label="Features selected" value={result!.selected_features?.length} color={agent.color} />
              </div>
            )}

            {/* ── Evaluation stats ── */}
            {hasStats && i === 5 && (
              <div className="mt-3 grid grid-cols-2 gap-2">
                <Stat label="Best Model" value={result!.best_model?.replace(/_/g, " ")} color={agent.color} />
                <Stat label="Score"      value={result!.best_score?.toFixed(4)}          color={agent.color} />
              </div>
            )}

            {/* ── Logs fallback: show when done but no dedicated stats, or when active ── */}
            {!hasStats && (done || active) && agentLogs.length > 0 && (
              <div className="mt-3 space-y-1">
                {agentLogs.map((log, li) => (
                  <p key={li} className="text-[10px] text-slate-400 leading-relaxed truncate"
                    style={{ borderLeft: `2px solid ${agent.color}40`, paddingLeft: 6 }}>
                    {log.replace(/^[^:]+Agent:\s*/i, "")}
                  </p>
                ))}
              </div>
            )}

            {/* ── Active pulse ── */}
            {active && (
              <div className="mt-3 flex items-center gap-2 text-xs" style={{ color: agent.color }}>
                {[0,1,2].map(d => (
                  <motion.div key={d} className="w-1.5 h-1.5 rounded-full" style={{ background: agent.color }}
                    animate={{ opacity:[1,.2,1], scale:[1,.7,1] }}
                    transition={{ duration:.9, repeat:Infinity, delay:d*.2 }} />
                ))}
                <span>Processing…</span>
              </div>
            )}

            {pending && <div className="mt-3 text-xs text-slate-600">Waiting…</div>}
          </motion.div>
        );
      })}
    </div>
  );
}

function StatusDot({ done, active, color }: { done:boolean; active:boolean; color:string }) {
  if (done) return <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ background:`${color}20`, color }}>Done</span>;
  if (active) return (
    <motion.span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ background:`${color}20`, color }}
      animate={{ opacity:[1,.5,1] }} transition={{ duration:1, repeat:Infinity }}>Running</motion.span>
  );
  return <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-slate-800 text-slate-500">Pending</span>;
}

function Stat({ label, value, color }: { label:string; value:any; color:string }) {
  return (
    <div className="rounded-lg p-2" style={{ background:`${color}0d` }}>
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-sm font-semibold truncate" style={{ color }}>{value ?? "—"}</p>
    </div>
  );
}
