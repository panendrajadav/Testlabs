export function cn(...classes: (string | undefined | null | boolean)[]): string {
  return classes
    .filter(Boolean)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function formatNumber(value: number, decimals: number = 2): string {
  return value.toFixed(decimals)
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(2)}%`
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
}

export function getStatusColor(status: string): string {
  const statusColors: Record<string, string> = {
    pending: 'text-gray-400',
    running: 'text-blue-400',
    completed: 'text-green-400',
    failed: 'text-red-400',
    success: 'text-green-400',
    error: 'text-red-400',
  }
  return statusColors[status] || 'text-gray-400'
}

export function getStatusBgColor(status: string): string {
  const statusColors: Record<string, string> = {
    pending: 'bg-gray-900/50 border border-gray-700',
    running: 'bg-blue-900/20 border border-blue-500/50',
    completed: 'bg-green-900/20 border border-green-500/50',
    failed: 'bg-red-900/20 border border-red-500/50',
  }
  return statusColors[status] || 'bg-gray-900/50 border border-gray-700'
}

export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export function generateId(): string {
  return Math.random().toString(36).substr(2, 9)
}

export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function getInitials(name: string): string {
  return name
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
}

export function truncate(str: string, length: number): string {
  return str.length > length ? str.substring(0, length) + '...' : str
}
