"use client";
import { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";
import { PipelineStatus } from "@/lib/api";
import { CheckCircle2, Loader2, RefreshCw } from "lucide-react";

const STAGES = [
  { label: "EDA",           icon: "🔍", color: "#3b82f6", glow: "rgba(59,130,246,0.6)"  },
  { label: "Preprocessing", icon: "⚙️", color: "#06b6d4", glow: "rgba(6,182,212,0.6)"   },
  { label: "Feature Eng",   icon: "🧬", color: "#8b5cf6", glow: "rgba(139,92,246,0.6)"  },
  { label: "Model Select",  icon: "🤖", color: "#f59e0b", glow: "rgba(245,158,11,0.6)"  },
  { label: "HPO",           icon: "⚡", color: "#10b981", glow: "rgba(16,185,129,0.6)"  },
  { label: "Evaluation",    icon: "🏆", color: "#ec4899", glow: "rgba(236,72,153,0.6)"  },
];

const targetFromProgress = (progress?: string, pipelineStatus?: string): number => {
  if (pipelineStatus === "completed") return STAGES.length;
  if (!progress) return -1;
  const p = progress.toLowerCase();
  if (p.includes("eda"))                                      return 0;
  if (p.includes("preprocess"))                               return 1;
  if (p.includes("feature"))                                  return 2;
  if (p.includes("model select") || p === "model selection") return 3;
  if (p.includes("hpo") || p.includes("hyper"))              return 4;
  if (p.includes("eval"))                                     return 5;
  if (p.includes("done") || p.includes("complet"))           return STAGES.length;
  return -1;
};

export default function PipelineVisualizer({ status }: { status: PipelineStatus | null }) {
  const targetIdx   = targetFromProgress(status?.progress, status?.status);
  const isRunning   = status?.status === "running" || status?.status === "queued";
  const isCompleted = status?.status === "completed";

  const [displayIdx, setDisplayIdx] = useState(-1);
  const [loopCount,  setLoopCount]  = useState(0);
  const prevTarget = useRef(-1);

  // Detect loop: when targetIdx drops back to 3 (Model Select) after being at 5 (Eval)
  useEffect(() => {
    if (prevTarget.current >= 5 && targetIdx === 3 && isRunning) {
      setLoopCount(c => c + 1);
      setDisplayIdx(3); // jump display back to Model Select instantly
    }
    prevTarget.current = targetIdx;
  }, [targetIdx, isRunning]);

  // Step displayIdx forward toward targetIdx one at a time
  useEffect(() => {
    if (targetIdx <= displayIdx) return;
    const timer = setTimeout(() => {
      setDisplayIdx(prev => prev + 1);
    }, displayIdx === -1 ? 300 : 550);
    return () => clearTimeout(timer);
  }, [targetIdx, displayIdx]);

  // Reset on new pipeline
  useEffect(() => {
    if (status?.status === "queued") {
      setDisplayIdx(-1);
      setLoopCount(0);
      prevTarget.current = -1;
    }
  }, [status?.status]);

  const activeIdx = displayIdx;
  const isLooping = loopCount > 0 && isRunning;
  const pct = isCompleted ? 100 : Math.round(Math.max(0, activeIdx + 1) / STAGES.length * 100);

  return (
    <div className="glass-panel p-8 overflow-hidden relative">
      {/* Background grid */}
      <div className="absolute inset-0 opacity-[0.03]"
        style={{ backgroundImage: "linear-gradient(rgba(99,179,237,1) 1px,transparent 1px),linear-gradient(90deg,rgba(99,179,237,1) 1px,transparent 1px)", backgroundSize: "40px 40px" }} />

      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <p className="text-sm font-semibold text-slate-200">AutoML Pipeline</p>
          <p className="text-xs text-slate-500 mt-0.5">LangGraph Multi-Agent Orchestration</p>
        </div>
        <div className="flex items-center gap-2">
          {isLooping && (
            <motion.div className="flex items-center gap-1.5 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 px-3 py-1.5 rounded-full"
              animate={{ opacity: [1, 0.5, 1] }} transition={{ duration: 1.2, repeat: Infinity }}>
              <RefreshCw className="w-3 h-3 animate-spin" />
              Loop {loopCount} — evaluating models
            </motion.div>
          )}
          {isRunning && !isLooping && (
            <motion.div className="flex items-center gap-2 text-xs text-blue-400 bg-blue-500/10 border border-blue-500/20 px-3 py-1.5 rounded-full"
              animate={{ opacity: [1, 0.6, 1] }} transition={{ duration: 1.5, repeat: Infinity }}>
              <Loader2 className="w-3 h-3 animate-spin" />
              {status?.progress || "Running…"}
            </motion.div>
          )}
          {isCompleted && (
            <div className="flex items-center gap-2 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-full">
              <CheckCircle2 className="w-3 h-3" />
              Completed
            </div>
          )}
        </div>
      </div>

      {/* Nodes row */}
      <div className="relative flex items-center justify-between" style={{ height: 140 }}>

        {/* Base track */}
        <div className="absolute left-7 right-7 top-[42px] h-0.5 z-0"
          style={{ background: "rgba(99,179,237,0.08)" }} />

        {/* Filled track */}
        {activeIdx >= 0 && (
          <motion.div
            className="absolute top-[42px] h-0.5 z-0 rounded-full"
            style={{ left: "28px", background: "linear-gradient(90deg,#3b82f6,#06b6d4,#8b5cf6,#f59e0b,#10b981,#ec4899)" }}
            animate={{ width: `calc(${Math.min(activeIdx, STAGES.length - 1) / (STAGES.length - 1) * 100}% - 56px)` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        )}

        {/* Travelling dot forward */}
        {isRunning && activeIdx >= 0 && activeIdx < STAGES.length - 1 && (
          <motion.div
            className="absolute top-[38px] w-2.5 h-2.5 rounded-full z-10 pointer-events-none"
            style={{
              background: STAGES[activeIdx + 1]?.color ?? "#3b82f6",
              boxShadow: `0 0 10px ${STAGES[activeIdx + 1]?.color ?? "#3b82f6"}`,
            }}
            animate={{
              left: [`${(activeIdx / (STAGES.length - 1)) * 100}%`, `${((activeIdx + 1) / (STAGES.length - 1)) * 100}%`],
              opacity: [0, 1, 1, 0],
            }}
            transition={{ duration: 0.55, repeat: Infinity, ease: "easeInOut" }}
          />
        )}

        {/* ── Loop back arc: Evaluation → Model Select ── */}
        {isLooping && (
          <svg className="absolute inset-0 w-full pointer-events-none z-20" style={{ height: 140, overflow: "visible" }}>
            {/* Arc path below the nodes */}
            <motion.path
              d={`M 83.33% 42px Q 58% 110px 50% 42px`}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="1.5"
              strokeDasharray="6 4"
              opacity={0.5}
              animate={{ opacity: [0.3, 0.7, 0.3] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
            {/* Travelling dot on the arc */}
            <motion.circle r="4" fill="#f59e0b"
              style={{ filter: "drop-shadow(0 0 4px #f59e0b)" }}
              animate={{
                offsetDistance: ["0%", "100%"],
                opacity: [0, 1, 1, 0],
              }}
              transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
            />
          </svg>
        )}

        {/* Loop badge on Model Select node when looping */}
        {isLooping && (
          <motion.div
            className="absolute z-30 text-[9px] font-bold px-1.5 py-0.5 rounded-full"
            style={{
              left: `calc(${3 / (STAGES.length - 1) * 100}% - 12px)`,
              top: "0px",
              background: "#f59e0b20",
              border: "1px solid #f59e0b60",
              color: "#f59e0b",
            }}
            animate={{ opacity: [1, 0.4, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          >
            ×{loopCount + 1}
          </motion.div>
        )}

        {STAGES.map((stage, i) => {
          const done   = activeIdx > i;
          const active = activeIdx === i;
          const idle   = activeIdx < i;
          // Nodes 3-5 stay "active-ish" during looping even when done
          const loopActive = isLooping && i >= 3 && i <= 5;

          return (
            <motion.div key={stage.label}
              className="flex flex-col items-center gap-2 z-10 relative"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.07, type: "spring", stiffness: 220 }}>

              <div className="relative">
                {/* Pulse rings — active */}
                {active && (<>
                  <motion.div className="absolute inset-0 rounded-full"
                    style={{ border: `2px solid ${stage.color}` }}
                    animate={{ scale: [1, 1.9], opacity: [0.7, 0] }}
                    transition={{ duration: 1.4, repeat: Infinity }} />
                  <motion.div className="absolute inset-0 rounded-full"
                    style={{ border: `2px solid ${stage.color}` }}
                    animate={{ scale: [1, 1.5], opacity: [0.5, 0] }}
                    transition={{ duration: 1.4, repeat: Infinity, delay: 0.35 }} />
                </>)}

                {/* Loop pulse on nodes 3-5 */}
                {loopActive && !active && (
                  <motion.div className="absolute inset-0 rounded-full"
                    style={{ border: `1.5px solid ${stage.color}` }}
                    animate={{ scale: [1, 1.6], opacity: [0.4, 0] }}
                    transition={{ duration: 2, repeat: Infinity, delay: i * 0.2 }} />
                )}

                {/* Orbital dot — done */}
                {done && (
                  <motion.div className="absolute -inset-2 rounded-full border"
                    style={{ borderColor: `${stage.color}35` }}
                    animate={{ rotate: 360 }}
                    transition={{ duration: 7, repeat: Infinity, ease: "linear" }}>
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full"
                      style={{ background: stage.color }} />
                  </motion.div>
                )}

                {/* Main circle */}
                <motion.div
                  className="w-14 h-14 rounded-full flex items-center justify-center text-xl relative z-10 border-2 select-none"
                  style={{
                    borderColor: done || active ? stage.color : "rgba(99,179,237,0.12)",
                    background: done
                      ? `radial-gradient(circle, ${stage.color}35, ${stage.color}10)`
                      : active
                      ? `radial-gradient(circle, ${stage.color}25, transparent)`
                      : "rgba(10,22,40,0.8)",
                    boxShadow: active
                      ? `0 0 28px ${stage.glow}, 0 0 56px ${stage.color}25, inset 0 0 18px ${stage.color}10`
                      : loopActive
                      ? `0 0 18px ${stage.color}50`
                      : done
                      ? `0 0 14px ${stage.color}45`
                      : "none",
                    transition: "all 0.4s ease",
                  }}
                  animate={active ? { scale: [1, 1.06, 1] } : loopActive ? { scale: [1, 1.03, 1] } : { scale: 1 }}
                  transition={active ? { duration: 1.1, repeat: Infinity } : loopActive ? { duration: 1.8, repeat: Infinity } : { type: "spring" }}>
                  {done ? (
                    <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 320 }}>
                      <CheckCircle2 className="w-6 h-6" style={{ color: stage.color }} />
                    </motion.div>
                  ) : (
                    <span style={{ filter: idle ? "grayscale(1) opacity(0.35)" : "none", transition: "filter 0.3s" }}>
                      {stage.icon}
                    </span>
                  )}
                </motion.div>
              </div>

              {/* Label */}
              <div className="text-center">
                <p className="text-xs font-semibold" style={{ color: done || active ? stage.color : "#475569", transition: "color 0.3s" }}>
                  {stage.label}
                </p>
                {active && (
                  <motion.p className="text-[10px] text-slate-500 mt-0.5"
                    animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 0.9, repeat: Infinity }}>
                    Processing…
                  </motion.p>
                )}
                {done && !loopActive && <p className="text-[10px] text-slate-600 mt-0.5">Done ✓</p>}
                {loopActive && done && (
                  <motion.p className="text-[10px] mt-0.5" style={{ color: stage.color }}
                    animate={{ opacity: [1, 0.4, 1] }} transition={{ duration: 1, repeat: Infinity }}>
                    Looping…
                  </motion.p>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Loop info bar */}
      {isLooping && (
        <motion.div
          className="mt-4 flex items-center gap-2 text-xs px-3 py-2 rounded-xl border"
          style={{ borderColor: "#f59e0b30", background: "#f59e0b08", color: "#f59e0b" }}
          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          <RefreshCw className="w-3 h-3 animate-spin flex-shrink-0" />
          <span>Model Selection → HPO → Evaluation loop is running. Each iteration evaluates a new ML model.</span>
        </motion.div>
      )}

      {/* Progress bar */}
      <div className="mt-4 space-y-1.5">
        <div className="flex justify-between text-xs text-slate-500">
          <span>Progress</span>
          <span>{pct}%</span>
        </div>
        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
          <motion.div className="h-full rounded-full"
            style={{ background: "linear-gradient(90deg,#3b82f6,#06b6d4,#8b5cf6,#ec4899)" }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }} />
        </div>
      </div>

      {/* Stage chips */}
      <div className="mt-4 flex flex-wrap gap-2">
        {STAGES.map((s, i) => {
          const done   = activeIdx > i;
          const active = activeIdx === i;
          const loopActive = isLooping && i >= 3 && i <= 5;
          return (
            <motion.span key={s.label}
              className="text-[10px] px-2.5 py-0.5 rounded-full border"
              animate={{
                borderColor: done || active || loopActive ? `${s.color}50` : "rgba(99,179,237,0.1)",
                color: done || active || loopActive ? s.color : "#475569",
                backgroundColor: done || active || loopActive ? `${s.color}12` : "transparent",
              }}
              transition={{ duration: 0.3 }}>
              {s.label}{loopActive ? " ↻" : ""}
            </motion.span>
          );
        })}
      </div>
    </div>
  );
}
