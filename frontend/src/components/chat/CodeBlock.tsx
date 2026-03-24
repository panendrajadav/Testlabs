'use client'

import { useState } from 'react'
import { Check, Copy } from 'lucide-react'

interface CodeBlockProps {
  code: string
  language?: string
}

export default function CodeBlock({ code, language = 'python' }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="rounded-xl overflow-hidden border border-slate-700/60 my-2" style={{ background: '#1e1e2e' }}>
      {/* macOS title bar */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-700/50" style={{ background: '#2a2a3d' }}>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <div className="w-3 h-3 rounded-full bg-yellow-400" />
          <div className="w-3 h-3 rounded-full bg-green-500" />
        </div>
        <span className="text-xs text-slate-400 font-mono">{language}</span>
        <button
          onClick={copy}
          className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition-colors px-2 py-1 rounded-md hover:bg-slate-700/50"
        >
          {copied
            ? <><Check size={11} className="text-green-400" /><span className="text-green-400">Copied</span></>
            : <><Copy size={11} /><span>Copy</span></>
          }
        </button>
      </div>
      {/* Code body */}
      <pre className="p-4 text-sm font-mono overflow-x-auto leading-relaxed" style={{ color: '#cdd6f4', maxHeight: '480px', overflowY: 'auto' }}>
        <code>{code}</code>
      </pre>
    </div>
  )
}
