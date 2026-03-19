'use client'

import { motion } from 'framer-motion'

export const PulseLoader = () => (
  <div className="flex gap-1 items-center justify-center">
    {[0, 1, 2].map((i) => (
      <motion.div
        key={i}
        animate={{ scale: [1, 1.5, 1], opacity: [0.5, 1, 0.5] }}
        transition={{ delay: i * 0.1, duration: 1, repeat: Infinity }}
        className="w-2 h-2 bg-purple-400 rounded-full"
      />
    ))}
  </div>
)

export const ShimmerLoader = ({ count = 3 }: { count?: number }) => (
  <div className="space-y-3">
    {Array.from({ length: count }).map((_, i) => (
      <motion.div
        key={i}
        className="h-4 bg-gradient-to-r from-gray-800 via-gray-700 to-gray-800 rounded shimmer"
        style={{ backgroundSize: '200% 100%' }}
      />
    ))}
  </div>
)

export const SkeletonCard = () => (
  <motion.div
    animate={{ opacity: [0.5, 1, 0.5] }}
    transition={{ duration: 2, repeat: Infinity }}
    className="bg-gray-800/50 rounded-lg p-4 space-y-3"
  >
    <div className="h-4 bg-gray-700 rounded w-3/4" />
    <div className="h-24 bg-gray-700 rounded" />
    <div className="h-4 bg-gray-700 rounded" />
  </motion.div>
)

export const PipelineLoadingAnimation = () => (
  <div className="flex flex-col items-center justify-center py-12">
    <motion.div
      animate={{ rotate: 360 }}
      transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
      className="w-24 h-24 border-4 border-purple-500/20 border-t-purple-500 rounded-full mb-6 glow"
    />
    <motion.div
      animate={{ opacity: [0.5, 1, 0.5] }}
      transition={{ duration: 1.5, repeat: Infinity }}
      className="text-center"
    >
      <p className="text-lg font-semibold text-transparent bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text">
        Pipeline Running...
      </p>
      <div className="flex gap-1 justify-center mt-2">
        <motion.span
          animate={{ y: [0, -10, 0] }}
          transition={{ delay: 0, duration: 0.5, repeat: Infinity }}
        >
          .
        </motion.span>
        <motion.span
          animate={{ y: [0, -10, 0] }}
          transition={{ delay: 0.1, duration: 0.5, repeat: Infinity }}
        >
          .
        </motion.span>
        <motion.span
          animate={{ y: [0, -10, 0] }}
          transition={{ delay: 0.2, duration: 0.5, repeat: Infinity }}
        >
          .
        </motion.span>
      </div>
    </motion.div>
  </div>
)

export const UploadProgressLoader = ({ progress }: { progress: number }) => (
  <div className="space-y-3">
    <div className="flex justify-between items-center">
      <p className="text-sm font-medium text-gray-300">Uploading...</p>
      <p className="text-sm text-purple-400 font-semibold">{progress}%</p>
    </div>
    <motion.div
      className="h-2 bg-gray-800 rounded-full overflow-hidden"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <motion.div
        className="h-full bg-gradient-to-r from-purple-500 to-cyan-500 glow"
        initial={{ width: 0 }}
        animate={{ width: `${progress}%` }}
        transition={{ duration: 0.3 }}
      />
    </motion.div>
  </div>
)
