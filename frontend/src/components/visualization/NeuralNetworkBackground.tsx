'use client'

import { useEffect, useRef } from 'react'

interface Particle {
  x: number
  y: number
  ox: number  // original/home x
  oy: number  // original/home y
  vx: number
  vy: number
  size: number
  opacity: number
}

const REPEL_RADIUS = 160
const REPEL_STRENGTH = 6
const RETURN_EASE = 0.04

export default function NeuralNetworkBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const particlesRef = useRef<Particle[]>([])
  const mouseRef = useRef({ x: -9999, y: -9999 })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    const onMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY }
    }
    const onMouseLeave = () => {
      mouseRef.current = { x: -9999, y: -9999 }
    }
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseleave', onMouseLeave)

    const particleCount = 60
    particlesRef.current = Array.from({ length: particleCount }).map(() => {
      const x = Math.random() * canvas.width
      const y = Math.random() * canvas.height
      return {
        x, y, ox: x, oy: y,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        size: Math.random() * 2.5 + 1.5,
        opacity: Math.random() * 0.4 + 0.5,
      }
    })

    let animFrameId: number
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const { x: mx, y: my } = mouseRef.current

      particlesRef.current.forEach((p) => {
        // Drift home position slowly
        p.ox += p.vx
        p.oy += p.vy
        if (p.ox < 0 || p.ox > canvas.width) p.vx *= -1
        if (p.oy < 0 || p.oy > canvas.height) p.vy *= -1

        // Repel from cursor
        const dx = p.x - mx
        const dy = p.y - my
        const dist = Math.hypot(dx, dy)
        if (dist < REPEL_RADIUS && dist > 0) {
          const force = (1 - dist / REPEL_RADIUS) * REPEL_STRENGTH
          p.x += (dx / dist) * force
          p.y += (dy / dist) * force
        }

        // Ease back to drifting home position
        p.x += (p.ox - p.x) * RETURN_EASE
        p.y += (p.oy - p.y) * RETURN_EASE

        // Draw particle — brighter when near cursor
        const proximity = dist < REPEL_RADIUS ? 1 - dist / REPEL_RADIUS : 0
        const alpha = Math.min(1, p.opacity + proximity * 0.5)
        const radius = p.size + proximity * 2
        // Glow
        ctx.shadowBlur = 8 + proximity * 12
        ctx.shadowColor = 'rgba(167, 139, 250, 0.9)'
        ctx.fillStyle = proximity > 0.2 ? `rgba(200, 180, 255, ${alpha})` : `rgba(167, 139, 250, ${alpha})`
        ctx.beginPath()
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2)
        ctx.fill()
        ctx.shadowBlur = 0
      })

      // Draw connections
      for (let i = 0; i < particlesRef.current.length; i++) {
        for (let j = i + 1; j < particlesRef.current.length; j++) {
          const p1 = particlesRef.current[i]
          const p2 = particlesRef.current[j]
          const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y)
          if (dist < 160) {
            const midX = (p1.x + p2.x) / 2
            const midY = (p1.y + p2.y) / 2
            const cursorDist = Math.hypot(midX - mx, midY - my)
            const boost = cursorDist < REPEL_RADIUS ? (1 - cursorDist / REPEL_RADIUS) : 0
            const baseAlpha = (1 - dist / 160) * 0.35
            ctx.globalAlpha = Math.min(1, baseAlpha + boost * 0.7)
            ctx.lineWidth = boost > 0.3 ? 1.5 : 1
            ctx.shadowBlur = boost > 0.2 ? 6 : 0
            ctx.shadowColor = 'rgba(167, 139, 250, 0.8)'
            ctx.strokeStyle = boost > 0.2 ? 'rgba(200, 180, 255, 1)' : 'rgba(167, 139, 250, 1)'
            ctx.beginPath()
            ctx.moveTo(p1.x, p1.y)
            ctx.lineTo(p2.x, p2.y)
            ctx.stroke()
            ctx.shadowBlur = 0
            ctx.globalAlpha = 1
          }
        }
      }

      animFrameId = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      window.removeEventListener('resize', resizeCanvas)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseleave', onMouseLeave)
      cancelAnimationFrame(animFrameId)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
    />
  )
}
