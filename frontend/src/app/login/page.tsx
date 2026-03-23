'use client'

import { useState, useEffect, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { motion } from 'framer-motion'
import { Eye, EyeOff, Zap, AlertCircle, Lock, User } from 'lucide-react'
import { login, useAuth } from '@/hooks/useAuth'
import NeuralNetworkBackground from '@/components/visualization/NeuralNetworkBackground'

function LoginForm() {
  const router = useRouter()
  const params = useSearchParams()
  const { isLoggedIn } = useAuth()
  const redirect = params.get('redirect') || '/upload'

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPass, setShowPass] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (isLoggedIn) router.replace(redirect)
  }, [isLoggedIn])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    await new Promise(r => setTimeout(r, 600))
    const ok = login(username.trim(), password)
    if (ok) {
      router.replace(redirect)
    } else {
      setError('Invalid username or password')
      setLoading(false)
    }
  }

  return (
    <div className="relative min-h-screen bg-black flex items-center justify-center overflow-hidden">
      <NeuralNetworkBackground />

      {/* Orbs */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {[
          { left: '10%', top: '20%', w: 300, color: '#8b5cf6' },
          { left: '75%', top: '10%', w: 200, color: '#06b6d4' },
          { left: '60%', top: '65%', w: 260, color: '#7c3aed' },
        ].map((orb, i) => (
          <motion.div
            key={i}
            className="absolute rounded-full"
            style={{
              left: orb.left, top: orb.top,
              width: orb.w, height: orb.w,
              background: `radial-gradient(circle at 35% 35%, ${orb.color}33, transparent 70%)`,
              filter: 'blur(4px)',
            }}
            animate={{ y: [0, -20, 0], scale: [1, 1.08, 1] }}
            transition={{ duration: 8 + i * 2, repeat: Infinity, ease: 'easeInOut', delay: i * 1.5 }}
          />
        ))}
      </div>

      <div className="relative z-10 w-full max-w-md px-4">
        {/* Logo */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col items-center mb-10"
        >
          <div className="flex items-center gap-3 mb-3">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
              className="w-10 h-10 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-xl"
            />
            <span className="text-2xl font-bold text-white">TestLabs AutoML</span>
          </div>
          <p className="text-gray-400 text-sm">Sign in to start your experiment</p>
        </motion.div>

        {/* Card */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="rounded-2xl border border-purple-500/20 p-8"
          style={{ background: 'rgba(10,10,20,0.85)', backdropFilter: 'blur(20px)' }}
        >
          <h2 className="text-xl font-semibold text-white mb-6">Welcome back</h2>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Username */}
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1.5">Username</label>
              <div className="relative">
                <User size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input
                  type="text"
                  value={username}
                  onChange={e => { setUsername(e.target.value); setError('') }}
                  placeholder="Enter your username"
                  autoComplete="username"
                  required
                  className="w-full pl-9 pr-4 py-3 rounded-lg bg-slate-900/80 border border-slate-700 text-white placeholder-gray-600 text-sm focus:border-purple-500 focus:ring-0 transition-colors"
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1.5">Password</label>
              <div className="relative">
                <Lock size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input
                  type={showPass ? 'text' : 'password'}
                  value={password}
                  onChange={e => { setPassword(e.target.value); setError('') }}
                  placeholder="Enter your password"
                  autoComplete="current-password"
                  required
                  className="w-full pl-9 pr-10 py-3 rounded-lg bg-slate-900/80 border border-slate-700 text-white placeholder-gray-600 text-sm focus:border-purple-500 focus:ring-0 transition-colors"
                />
                <button
                  type="button"
                  onClick={() => setShowPass(s => !s)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 transition-colors"
                >
                  {showPass ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-2 text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2"
              >
                <AlertCircle size={14} className="shrink-0" />
                {error}
              </motion.div>
            )}

            {/* Submit */}
            <motion.button
              type="submit"
              disabled={loading}
              whileHover={{ scale: loading ? 1 : 1.02 }}
              whileTap={{ scale: loading ? 1 : 0.97 }}
              className="w-full py-3 rounded-lg bg-gradient-to-r from-purple-600 to-cyan-600 text-white font-semibold text-sm flex items-center justify-center gap-2 shadow-lg shadow-purple-900/40 disabled:opacity-60 transition-opacity"
            >
              {loading ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
                  >
                    <Zap size={16} />
                  </motion.div>
                  Signing in…
                </>
              ) : (
                <>
                  <Zap size={16} />
                  Sign In
                </>
              )}
            </motion.button>
          </form>

          {/* Demo hint */}
          <div className="mt-6 pt-5 border-t border-slate-800">
            <p className="text-xs text-gray-600 text-center mb-3">Demo credentials</p>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => { setUsername('Panendra'); setPassword('16062005'); setError('') }}
                className="flex-1 py-2 rounded-lg border border-slate-700 bg-slate-900/50 text-xs text-gray-400 hover:text-white hover:border-purple-500/40 transition-colors"
              >
                <span className="font-mono">Panendra / 16062005</span>
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginForm />
    </Suspense>
  )
}
