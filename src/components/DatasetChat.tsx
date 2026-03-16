"use client";
import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Bot, User, BarChart2, Code2, Sparkles } from "lucide-react";
import { chatWithAnalyst, ChatResponse } from "@/lib/api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, ScatterChart, Scatter,
  PieChart, Pie, Cell, Legend,
} from "recharts";

interface Message {
  role: "user" | "assistant";
  content: string;
  chart?: Record<string, unknown> | null;
  code?: string | null;
  thinking?: boolean;
}

const SUGGESTIONS = [
  "What are the columns and their types?",
  "Show the distribution of each numeric column",
  "What is the correlation between features?",
  "Show a scatter plot of the first two numeric columns",
  "Give me summary statistics",
  "Are there any missing values?",
];

const THINKING_PHRASES = [
  "Analyzing dataset…", "Querying data…",
  "Building visualization…", "Computing statistics…",
  "Generating insights…",
];

const CHART_COLORS = ["#3b82f6","#06b6d4","#8b5cf6","#f59e0b","#10b981","#ec4899","#f97316","#14b8a6"];

export default function DatasetChat({ datasetId }: { datasetId: string | null }) {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hi! I'm your data analyst. Ask me anything about your dataset — statistics, visualizations, correlations, or any insights you need." }
  ]);
  const [input, setInput]     = useState("");
  const [loading, setLoading] = useState(false);
  const [thinkIdx, setThinkIdx] = useState(0);
  const [showChart, setShowChart] = useState<Record<number, boolean>>({});
  const [showCode,  setShowCode]  = useState<Record<number, boolean>>({});
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  useEffect(() => {
    if (!loading) return;
    const t = setInterval(() => setThinkIdx(i => (i + 1) % THINKING_PHRASES.length), 1600);
    return () => clearInterval(t);
  }, [loading]);

  const send = async (q?: string) => {
    const question = (q ?? input).trim();
    if (!question || !datasetId || loading) return;
    setInput("");
    setMessages(m => [...m, { role: "user", content: question }]);
    setLoading(true);
    setMessages(m => [...m, { role: "assistant", content: "", thinking: true }]);
    try {
      const res: ChatResponse = await chatWithAnalyst(datasetId, question);
      setMessages(m => {
        const copy = [...m];
        copy[copy.length - 1] = {
          role: "assistant",
          content: res.answer,
          chart: res.chart,
          code: res.code,
          thinking: false,
        };
        return copy;
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Request failed";
      setMessages(m => {
        const c = [...m];
        c[c.length - 1] = { role: "assistant", content: `Error: ${msg}`, thinking: false };
        return c;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel flex flex-col h-[600px]">
      {/* Header */}
      <div className="flex items-center gap-3 p-4 border-b border-white/5">
        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
          <Bot className="w-4 h-4 text-white" />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-100">Data Analyst</p>
          <p className="text-xs text-slate-500">{datasetId ? "Dataset loaded — ask anything" : "Upload a dataset first"}</p>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <motion.div className="w-2 h-2 rounded-full bg-emerald-400"
            animate={{ opacity: [1, .3, 1] }} transition={{ duration: 2, repeat: Infinity }} />
          <span className="text-xs text-slate-500">Live</span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence initial={false}>
          {messages.map((msg, idx) => (
            <motion.div key={idx}
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              transition={{ duration: .2 }}
              className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>

              <div className={`w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center
                ${msg.role === "assistant" ? "bg-gradient-to-br from-blue-500 to-cyan-500" : "bg-gradient-to-br from-purple-500 to-pink-500"}`}>
                {msg.role === "assistant"
                  ? <Bot className="w-3.5 h-3.5 text-white" />
                  : <User className="w-3.5 h-3.5 text-white" />}
              </div>

              <div className={`max-w-[82%] space-y-2 flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}>
                <div className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap
                  ${msg.role === "user"
                    ? "bg-blue-600/80 text-white rounded-tr-sm"
                    : "bg-slate-800/80 text-slate-200 rounded-tl-sm border border-white/5"}`}>
                  {msg.thinking ? (
                    <div className="flex items-center gap-2 text-slate-400">
                      {[0,1,2].map(d => (
                        <motion.div key={d} className="w-1.5 h-1.5 rounded-full bg-blue-400"
                          animate={{ opacity:[1,.2,1], y:[0,-3,0] }}
                          transition={{ duration:.7, repeat:Infinity, delay:d*.15 }} />
                      ))}
                      <span className="text-xs">{THINKING_PHRASES[thinkIdx]}</span>
                    </div>
                  ) : msg.content}
                </div>

                {/* Action buttons */}
                {msg.role === "assistant" && !msg.thinking && (msg.chart || msg.code) && (
                  <div className="flex gap-2 flex-wrap">
                    {msg.chart && (
                      <button onClick={() => setShowChart(s => ({ ...s, [idx]: !s[idx] }))}
                        className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-cyan-500/15 text-cyan-400 border border-cyan-500/25 hover:bg-cyan-500/25 transition-colors">
                        <BarChart2 className="w-3 h-3" />
                        {showChart[idx] ? "Hide Chart" : "Visualize Insight"}
                      </button>
                    )}
                    {msg.code && (
                      <button onClick={() => setShowCode(s => ({ ...s, [idx]: !s[idx] }))}
                        className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-purple-500/15 text-purple-400 border border-purple-500/25 hover:bg-purple-500/25 transition-colors">
                        <Code2 className="w-3 h-3" />
                        {showCode[idx] ? "Hide Code" : "View Code"}
                      </button>
                    )}
                  </div>
                )}

                {/* Chart */}
                {showChart[idx] && msg.chart && (
                  <motion.div initial={{ opacity:0, height:0 }} animate={{ opacity:1, height:"auto" }}
                    className="w-full glass-panel p-3 rounded-xl overflow-hidden min-w-[300px]">
                    <PlotlyChart data={msg.chart} />
                  </motion.div>
                )}

                {/* Code */}
                {showCode[idx] && msg.code && (
                  <motion.div initial={{ opacity:0, height:0 }} animate={{ opacity:1, height:"auto" }} className="w-full">
                    <pre className="text-xs text-slate-300 overflow-x-auto p-3 rounded-xl bg-black/40 border border-white/10">
                      <code>{msg.code}</code>
                    </pre>
                  </motion.div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Suggestions — only show when no conversation yet */}
        {messages.length === 1 && datasetId && (
          <motion.div initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ delay:.3 }}
            className="space-y-2">
            <p className="text-xs text-slate-500 flex items-center gap-1.5">
              <Sparkles className="w-3 h-3" /> Suggested questions
            </p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map(s => (
                <button key={s} onClick={() => send(s)}
                  className="text-xs px-3 py-1.5 rounded-lg bg-slate-800/60 border border-white/8 text-slate-400 hover:text-slate-200 hover:border-blue-500/30 hover:bg-blue-500/8 transition-all">
                  {s}
                </button>
              ))}
            </div>
          </motion.div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-white/5">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && !e.shiftKey && send()}
            placeholder={datasetId ? "Ask anything about your dataset…" : "Upload a dataset first"}
            disabled={!datasetId || loading}
            className="flex-1 bg-slate-800/60 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/20 disabled:opacity-40 transition-all"
          />
          <button onClick={() => send()} disabled={!datasetId || loading || !input.trim()}
            className="w-10 h-10 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-colors flex-shrink-0">
            <Send className="w-4 h-4 text-white" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Universal Plotly → Recharts renderer ──
function PlotlyChart({ data }: { data: Record<string, unknown> }) {
  const d = data as {
    data?: Array<{
      x?: unknown; y?: unknown; type?: string; name?: string;
      labels?: unknown; values?: unknown; z?: unknown;
    }>;
    layout?: { title?: string | { text?: string } };
  };

  if (!d.data?.length) return <p className="text-xs text-slate-500">No chart data</p>;

  const toArr = (v: unknown): unknown[] => Array.isArray(v) ? v : v != null ? [v] : [];
  const toNum = (v: unknown): number => typeof v === "number" ? v : parseFloat(String(v)) || 0;

  const title = typeof d.layout?.title === "string"
    ? d.layout.title
    : d.layout?.title?.text ?? "";

  const tooltipStyle = {
    background: "rgba(10,22,40,.95)",
    border: "1px solid rgba(99,179,237,.2)",
    borderRadius: 8,
    fontSize: 12,
  };

  // ── PIE ──
  const pieTrace = d.data.find(t => t.type === "pie");
  if (pieTrace) {
    const labels = toArr(pieTrace.labels);
    const values = toArr(pieTrace.values);
    const pieData = labels.map((l, i) => ({ name: String(l), value: toNum(values[i]) }));
    return (
      <div>
        {title && <p className="text-xs text-slate-400 mb-2">{title}</p>}
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`} labelLine={false}>
              {pieData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
            </Pie>
            <Tooltip contentStyle={tooltipStyle} />
            <Legend wrapperStyle={{ fontSize: 10, color: "#94a3b8" }} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // ── SCATTER ──
  const scatterTrace = d.data.find(t => t.type === "scatter" && toArr(t.x).length && toArr(t.y).length);
  if (scatterTrace) {
    const xs = toArr(scatterTrace.x);
    const ys = toArr(scatterTrace.y);
    const scatterData = xs.map((x, i) => ({ x: toNum(x), y: toNum(ys[i]) }));
    return (
      <div>
        {title && <p className="text-xs text-slate-400 mb-2">{title}</p>}
        <ResponsiveContainer width="100%" height={220}>
          <ScatterChart margin={{ top:4, right:4, left:-18, bottom:4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.08)" />
            <XAxis dataKey="x" type="number" tick={{ fontSize:10, fill:"#64748b" }} name={scatterTrace.name ?? "x"} />
            <YAxis dataKey="y" type="number" tick={{ fontSize:10, fill:"#64748b" }} name="y" />
            <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray:"3 3" }} />
            <Scatter data={scatterData} fill="#06b6d4" opacity={0.7} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // ── HISTOGRAM ──
  const histTrace = d.data.find(t => t.type === "histogram");
  if (histTrace) {
    const xs = toArr(histTrace.x);
    // bucket into 15 bins
    const nums = xs.map(toNum).filter(n => !isNaN(n));
    if (nums.length) {
      const min = Math.min(...nums), max = Math.max(...nums);
      const bins = 15;
      const size = (max - min) / bins || 1;
      const counts = Array(bins).fill(0);
      nums.forEach(n => { const b = Math.min(Math.floor((n - min) / size), bins - 1); counts[b]++; });
      const histData = counts.map((c, i) => ({ x: (min + i * size).toFixed(1), y: c }));
      return (
        <div>
          {title && <p className="text-xs text-slate-400 mb-2">{title}</p>}
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={histData} margin={{ top:4, right:4, left:-18, bottom:4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.08)" />
              <XAxis dataKey="x" tick={{ fontSize:9, fill:"#64748b" }} />
              <YAxis tick={{ fontSize:10, fill:"#64748b" }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="y" fill="#8b5cf6" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      );
    }
  }

  // ── BAR (default) ──
  const barTrace = d.data[0];
  const xs = toArr(barTrace.x);
  const ys = toArr(barTrace.y);
  if (!xs.length && !ys.length) return <p className="text-xs text-slate-500">Empty chart data</p>;

  const chartData = xs.length
    ? xs.map((x, i) => ({ x: String(x), y: toNum(ys[i]) }))
    : ys.map((y, i) => ({ x: String(i), y: toNum(y) }));

  const isLine = barTrace.type === "line" || barTrace.type === "lines";

  return (
    <div>
      {title && <p className="text-xs text-slate-400 mb-2">{title}</p>}
      <ResponsiveContainer width="100%" height={200}>
        {isLine ? (
          <LineChart data={chartData} margin={{ top:4, right:4, left:-18, bottom:4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.08)" />
            <XAxis dataKey="x" tick={{ fontSize:10, fill:"#64748b" }} />
            <YAxis tick={{ fontSize:10, fill:"#64748b" }} />
            <Tooltip contentStyle={tooltipStyle} />
            <Line type="monotone" dataKey="y" stroke="#3b82f6" dot={false} strokeWidth={2} />
          </LineChart>
        ) : (
          <BarChart data={chartData} margin={{ top:4, right:4, left:-18, bottom:4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,237,.08)" />
            <XAxis dataKey="x" tick={{ fontSize:10, fill:"#64748b" }} />
            <YAxis tick={{ fontSize:10, fill:"#64748b" }} />
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey="y" fill="#06b6d4" radius={[4,4,0,0]}>
              {chartData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
            </Bar>
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
