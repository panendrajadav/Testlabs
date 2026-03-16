"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload, Activity, Network, Bot, BarChart3, Database,
  Loader2, CheckCircle2, AlertCircle, Cpu, Zap, ChevronRight,
  FileText, X
} from "lucide-react";
import dynamic from "next/dynamic";
import PipelineVisualizer from "@/components/PipelineVisualizer";
import AgentCards from "@/components/AgentCards";
import DatasetChat from "@/components/DatasetChat";
import ResultsDashboard from "@/components/ResultsDashboard";
import { uploadDataset, runPipeline, getPipelineStatus, PipelineStatus } from "@/lib/api";

const NeuralBackground = dynamic(() => import("@/components/NeuralBackground"), { ssr: false });

const NAV_ITEMS = [
  { id:"hero",     label:"Dashboard",     icon: Activity },
  { id:"pipeline", label:"Pipeline",      icon: Network },
  { id:"agents",   label:"Agents",        icon: Cpu },
  { id:"results",  label:"Results",       icon: BarChart3 },
  { id:"chat",     label:"Dataset Chat",  icon: Bot },
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeNav, setActiveNav] = useState("hero");
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // Poll pipeline status
  useEffect(() => {
    if (!datasetId || (status?.status !== "queued" && status?.status !== "running")) return;
    const t = setInterval(async () => {
      try {
        const s = await getPipelineStatus(datasetId);
        setStatus(s);
        if (s.status === "completed" || s.status === "failed") clearInterval(t);
      } catch {}
    }, 2500);
    return () => clearInterval(t);
  }, [datasetId, status?.status]);

  // Intersection observer for nav highlight
  useEffect(() => {
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) setActiveNav(e.target.id); });
    }, { threshold: 0.4 });
    NAV_ITEMS.forEach(n => { const el = document.getElementById(n.id); if (el) obs.observe(el); });
    return () => obs.disconnect();
  }, []);

  const handleFile = async (f: File) => {
    if (!f.name.endsWith(".csv")) { setError("Only CSV files are supported"); return; }
    setFile(f); setIsUploading(true); setError(null); setDatasetId(null); setStatus(null);
    try {
      const { dataset_id } = await uploadDataset(f);
      setDatasetId(dataset_id);
      setIsUploading(false); // ← clear spinner immediately after upload
      const s = await runPipeline(dataset_id);
      setStatus(s);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setIsUploading(false);
    }
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (f) handleFile(f);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files?.[0]; if (f) handleFile(f);
  };

  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior:"smooth" });

  const statusColor = status?.status === "completed" ? "#10b981"
    : status?.status === "failed" ? "#ef4444"
    : status?.status === "running" ? "#3b82f6" : "#f59e0b";

  return (
    <div className="min-h-screen bg-[#020817] text-slate-100 overflow-x-hidden">
      <NeuralBackground />

      {/* Ambient glows */}
      <div className="fixed top-0 left-1/4 w-96 h-96 rounded-full bg-blue-600/8 blur-[120px] pointer-events-none z-0" />
      <div className="fixed bottom-0 right-1/4 w-96 h-96 rounded-full bg-purple-600/8 blur-[120px] pointer-events-none z-0" />

      {/* ── Sidebar ── */}
      <aside className="fixed left-0 top-0 h-full w-16 lg:w-56 z-50 flex flex-col py-6 px-3 border-r border-white/5"
        style={{ background:"rgba(2,8,23,0.85)", backdropFilter:"blur(20px)" }}>
        {/* Logo */}
        <div className="flex items-center gap-2 px-1 mb-8">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center flex-shrink-0">
            <Zap className="w-4 h-4 text-white" />
          </div>
          <span className="hidden lg:block font-bold text-sm gradient-text">TestLabs AI</span>
        </div>

        {/* Nav */}
        <nav className="flex-1 space-y-1">
          {NAV_ITEMS.map(item => {
            const Icon = item.icon;
            const active = activeNav === item.id;
            return (
              <button key={item.id} onClick={() => scrollTo(item.id)}
                className={`w-full flex items-center gap-3 px-2 py-2.5 rounded-xl text-sm font-medium transition-all duration-200
                  ${active ? "bg-blue-600/20 text-blue-400 border border-blue-500/25" : "text-slate-500 hover:text-slate-300 hover:bg-white/5"}`}>
                <Icon className="w-4 h-4 flex-shrink-0" />
                <span className="hidden lg:block">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Status pill */}
        {status && (
          <div className="hidden lg:flex items-center gap-2 px-2 py-2 rounded-xl bg-white/5 border border-white/8">
            <motion.div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: statusColor }}
              animate={{ opacity:[1,.3,1] }} transition={{ duration:1.5, repeat:Infinity }} />
            <span className="text-xs text-slate-400 truncate capitalize">{status.status}</span>
          </div>
        )}
      </aside>

      {/* ── Top bar ── */}
      <header className="fixed top-0 left-16 lg:left-56 right-0 h-14 z-40 flex items-center px-6 gap-4 border-b border-white/5"
        style={{ background:"rgba(2,8,23,0.85)", backdropFilter:"blur(20px)" }}>
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <FileText className="w-4 h-4" />
          <span>{file ? file.name : "No dataset loaded"}</span>
        </div>
        {status && (
          <div className="flex items-center gap-1.5 ml-auto text-xs px-3 py-1 rounded-full border"
            style={{ color: statusColor, borderColor:`${statusColor}40`, background:`${statusColor}12` }}>
            {(status.status === "running" || status.status === "queued") && (
              <Loader2 className="w-3 h-3 animate-spin" />
            )}
            {status.status === "completed" && <CheckCircle2 className="w-3 h-3" />}
            <span className="capitalize">{status.progress || status.status}</span>
          </div>
        )}
      </header>

      {/* ── Main content ── */}
      <main className="pl-16 lg:pl-56 pt-14">
        <div className="max-w-6xl mx-auto px-6 py-12 space-y-28">

          {/* ══ HERO ══ */}
          <section id="hero" className="min-h-[80vh] flex flex-col items-center justify-center text-center space-y-8 scroll-mt-14">
            <motion.div initial={{ opacity:0, y:-10 }} animate={{ opacity:1, y:0 }}
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium border border-blue-500/30 bg-blue-500/10 text-blue-400">
              <Activity className="w-3.5 h-3.5" />
              Powered by LangGraph Multi-Agent Orchestration
            </motion.div>

            <motion.h1 initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:.1 }}
              className="text-5xl md:text-7xl font-extrabold tracking-tight leading-tight">
              Automate the ML Lifecycle<br />
              with <span className="gradient-text">Intelligent Agents</span>
            </motion.h1>

            <motion.p initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:.2 }}
              className="max-w-2xl text-lg text-slate-400 leading-relaxed">
              Upload a dataset and let AI agents build and evaluate machine learning models automatically.
              Powered by Azure AI Foundry and LangGraph.
            </motion.p>

            {/* Upload zone */}
            <motion.div initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} transition={{ delay:.3 }}
              className="w-full max-w-lg">
              <div
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => fileRef.current?.click()}
                className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition-all duration-300
                  ${dragOver ? "border-blue-400 bg-blue-500/10 scale-[1.02]" : "border-white/15 hover:border-blue-500/50 hover:bg-white/3"}`}>
                <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={onFileChange} />
                {isUploading ? (
                  <div className="flex flex-col items-center gap-3">
                    <Loader2 className="w-10 h-10 text-blue-400 animate-spin" />
                    <p className="text-sm text-slate-400">Uploading & starting pipeline…</p>
                  </div>
                ) : file ? (
                  <div className="flex flex-col items-center gap-3">
                    <CheckCircle2 className="w-10 h-10 text-emerald-400" />
                    <p className="text-sm font-medium text-slate-200">{file.name}</p>
                    <p className="text-xs text-slate-500">Click to replace</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3">
                    <div className="w-14 h-14 rounded-2xl bg-blue-500/15 border border-blue-500/25 flex items-center justify-center">
                      <Upload className="w-6 h-6 text-blue-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-slate-200">Drop your CSV here</p>
                      <p className="text-xs text-slate-500 mt-1">or click to browse</p>
                    </div>
                  </div>
                )}
              </div>

              <AnimatePresence>
                {error && (
                  <motion.div initial={{ opacity:0, y:-8 }} animate={{ opacity:1, y:0 }} exit={{ opacity:0 }}
                    className="mt-3 flex items-center gap-2 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-2.5">
                    <AlertCircle className="w-4 h-4 flex-shrink-0" />
                    {error}
                    <button onClick={() => setError(null)} className="ml-auto"><X className="w-3.5 h-3.5" /></button>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            <motion.button initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:.4 }}
              onClick={() => scrollTo("pipeline")}
              className="flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200 transition-colors">
              View Pipeline <ChevronRight className="w-4 h-4" />
            </motion.button>

            {/* Stats row */}
            <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:.5 }}
              className="grid grid-cols-3 gap-4 w-full max-w-lg mt-4">
              {[
                { label:"Agents", value:"6", color:"#3b82f6" },
                { label:"Models", value:"9+", color:"#06b6d4" },
                { label:"Metrics", value:"5+", color:"#8b5cf6" },
              ].map(s => (
                <div key={s.label} className="glass-panel p-4 text-center rounded-xl">
                  <p className="text-2xl font-bold" style={{ color:s.color }}>{s.value}</p>
                  <p className="text-xs text-slate-500 mt-0.5">{s.label}</p>
                </div>
              ))}
            </motion.div>
          </section>

          {/* ══ PIPELINE ══ */}
          <section id="pipeline" className="space-y-6 scroll-mt-20">
            <SectionHeader icon={<Network className="w-5 h-5 text-blue-400" />} title="Agent Workflow Pipeline"
              sub="Real-time execution across 6 specialized LangGraph agents" />
            <PipelineVisualizer status={status} />
          </section>

          {/* ══ AGENTS ══ */}
          <section id="agents" className="space-y-6 scroll-mt-20">
            <SectionHeader icon={<Cpu className="w-5 h-5 text-cyan-400" />} title="Specialized Agents"
              sub="Each agent handles a distinct stage of the AutoML lifecycle" />
            <AgentCards result={status?.result} currentStatus={status?.status} progress={status?.progress} />
          </section>

          {/* ══ RESULTS + CHAT ══ */}
          <section id="results" className="space-y-6 scroll-mt-20">
            <SectionHeader icon={<BarChart3 className="w-5 h-5 text-purple-400" />} title="Model Results"
              sub="Metrics, ROC curves, SHAP feature importance, and model comparison" />
            <ResultsDashboard result={status?.result} />
          </section>

          <section id="chat" className="space-y-6 scroll-mt-20 pb-20">
            <SectionHeader icon={<Bot className="w-5 h-5 text-pink-400" />} title="Dataset Chat Analyst"
              sub="Ask questions about your data — get answers, charts, and code" />
            <DatasetChat datasetId={datasetId} />
          </section>
        </div>
      </main>
    </div>
  );
}

function SectionHeader({ icon, title, sub }: { icon: React.ReactNode; title: string; sub: string }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-xl font-bold text-slate-100">{title}</h2>
      </div>
      <p className="text-sm text-slate-500 ml-7">{sub}</p>
    </div>
  );
}
