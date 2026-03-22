'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useRouter, useSearchParams } from 'next/navigation'
import { Upload as UploadIcon, CheckCircle, AlertCircle } from 'lucide-react'
import { useUploadDataset, useRunPipeline } from '@/hooks/useApi'
import { setStoredDataset } from '@/hooks/useStoredDataset'
import { useQueryClient } from '@tanstack/react-query'
import { containerVariants, itemVariants } from '@/animations/variants'

export default function UploadPage() {
  const router = useRouter()
  const queryClient = useQueryClient()
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const uploadMutation = useUploadDataset()
  const runMutation = useRunPipeline()

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files.length > 0) {
      const file = files[0]
      if (file.name.endsWith('.csv')) {
        setSelectedFile(file)
      }
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      setSelectedFile(files[0])
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    uploadMutation.mutate(
      {
        file: selectedFile,
        onProgress: setUploadProgress,
      },
      {
        onSuccess: async (data) => {
          queryClient.removeQueries({ queryKey: ['pipelineStatus'] })
          setStoredDataset({ ...data, status: 'running', started_at: new Date().toISOString() })
          await runMutation.mutateAsync({
            datasetId: data.dataset_id,
            targetColumn: data.target_column,
          })
          router.push('/pipeline')
        },
      }
    )
  }

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={containerVariants}
      className="min-h-screen bg-black p-8 flex items-center justify-center"
    >
      <div className="max-w-2xl w-full">
        {/* Header */}
        <motion.div variants={itemVariants} className="mb-12">
          <h1 className="text-4xl font-bold text-gradient mb-2">Upload Your Dataset</h1>
          <p className="text-gray-400">Choose a CSV file to start the AutoML pipeline</p>
        </motion.div>

        {/* Upload Area */}
        <motion.div
          variants={itemVariants}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          whileHover={{ scale: 1.01 }}
          className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer ${
            isDragging
              ? 'border-purple-500 bg-purple-500/10'
              : 'border-purple-500/30 hover:border-purple-500/60 bg-white/2'
          }`}
        >
          <input
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            className="hidden"
            id="file-input"
          />

          <label htmlFor="file-input" className="cursor-pointer block">
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="flex justify-center mb-4"
            >
              <div className="w-16 h-16 bg-gradient-to-br from-purple-500/20 to-cyan-500/20 rounded-full flex items-center justify-center">
                <UploadIcon size={32} className="text-purple-400" />
              </div>
            </motion.div>

            <h3 className="text-xl font-semibold text-white mb-2">
              Drag and drop your CSV file here
            </h3>
            <p className="text-gray-400 mb-6">or click to browse your computer</p>

            {selectedFile ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center justify-center gap-2 text-green-400"
              >
                <CheckCircle size={20} />
                <span>{selectedFile.name}</span>
              </motion.div>
            ) : (
              <p className="text-sm text-gray-500">CSV files up to 100MB</p>
            )}
          </label>
        </motion.div>

        {/* Upload Progress */}
        {(uploadMutation.isPending || runMutation.isPending) && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 space-y-3"
          >
            <div className="flex justify-between items-center">
              <p className="text-sm text-purple-400 font-medium">
                {runMutation.isPending ? 'Starting pipeline...' : 'Uploading...'}
              </p>
              {uploadMutation.isPending && (
                <p className="text-sm text-purple-400 font-semibold">{uploadProgress}%</p>
              )}
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              {runMutation.isPending ? (
                // Indeterminate shimmer for pipeline start
                <motion.div
                  className="h-full w-1/3 bg-gradient-to-r from-purple-500 via-cyan-400 to-purple-500 rounded-full"
                  animate={{ x: ['0%', '300%'] }}
                  transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
                />
              ) : (
                <motion.div
                  className="h-full bg-gradient-to-r from-purple-500 to-cyan-500"
                  initial={{ width: '0%' }}
                  animate={{ width: uploadProgress === 0 ? '5%' : `${uploadProgress}%` }}
                  transition={{ duration: 0.4, ease: 'easeOut' }}
                />
              )}
            </div>
          </motion.div>
        )}

        {/* Upload Button */}
        {selectedFile && !uploadMutation.isPending && !runMutation.isPending && (
          <motion.button
            variants={itemVariants}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleUpload}
            className="w-full mt-8 px-6 py-4 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-lg font-semibold text-white glow hover:glow-lg transition-all"
          >
            Start Upload
          </motion.button>
        )}

        {/* Error Message */}
        {(uploadMutation.isError || runMutation.isError) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-6 flex items-center gap-2 text-red-400"
          >
            <AlertCircle size={20} />
            <span>{uploadMutation.isError ? 'Upload failed. Please try again.' : 'Failed to start pipeline. Please try again.'}</span>
          </motion.div>
        )}

        {/* Dataset Preview */}
        {selectedFile && (
          <motion.div variants={itemVariants} className="mt-12">
            <h2 className="text-xl font-semibold text-white mb-6">File Information</h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white/5 rounded-lg p-4 border border-purple-500/20">
                <p className="text-gray-400 text-sm mb-1">File Name</p>
                <p className="text-white font-semibold truncate">{selectedFile.name}</p>
              </div>
              <div className="bg-white/5 rounded-lg p-4 border border-purple-500/20">
                <p className="text-gray-400 text-sm mb-1">File Size</p>
                <p className="text-white font-semibold">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  )
}
