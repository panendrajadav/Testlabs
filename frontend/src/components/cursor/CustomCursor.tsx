'use client'

import { useEffect, useRef, useState } from 'react'
import { motion, useMotionValue, useSpring } from 'framer-motion'

export default function CustomCursor() {
  const [visible, setVisible]   = useState(false)
  const [clicking, setClicking] = useState(false)
  const [hovering, setHovering] = useState(false)

  const rawX = useMotionValue(-200)
  const rawY = useMotionValue(-200)

  // dot — snappy
  const dotX = useSpring(rawX, { stiffness: 900, damping: 40 })
  const dotY = useSpring(rawY, { stiffness: 900, damping: 40 })

  // ring — medium lag
  const ringX = useSpring(rawX, { stiffness: 130, damping: 22 })
  const ringY = useSpring(rawY, { stiffness: 130, damping: 22 })

  // aura — very laggy
  const auraX = useSpring(rawX, { stiffness: 50, damping: 16 })
  const auraY = useSpring(rawY, { stiffness: 50, damping: 16 })

  const orbsRef = useRef<{ el: HTMLElement; ox: number; oy: number }[]>([])

  useEffect(() => {
    const collectOrbs = () => {
      orbsRef.current = Array.from(
        document.querySelectorAll<HTMLElement>('[data-orb]')
      ).map((el) => {
        const r = el.getBoundingClientRect()
        return { el, ox: r.left + r.width / 2, oy: r.top + r.height / 2 }
      })
    }

    collectOrbs()

    const onMove = (e: MouseEvent) => {
      rawX.set(e.clientX)
      rawY.set(e.clientY)
      setVisible(true)

      // repel orbs
      orbsRef.current.forEach(({ el, ox, oy }) => {
        const dx = e.clientX - ox
        const dy = e.clientY - oy
        const dist = Math.hypot(dx, dy)
        const RADIUS = 180
        if (dist < RADIUS) {
          const force = (1 - dist / RADIUS) * 100
          const angle = Math.atan2(dy, dx)
          el.style.transform = `translate(${-Math.cos(angle) * force}px, ${-Math.sin(angle) * force}px)`
          el.style.transition = 'transform 0.12s ease-out'
        } else {
          el.style.transform = 'translate(0,0)'
          el.style.transition = 'transform 0.7s ease-out'
        }
      })

      const t = e.target as HTMLElement
      setHovering(!!t.closest('button,a,input,[role="button"],[data-hover]'))
    }

    const onLeave = () => setVisible(false)
    const onDown  = () => setClicking(true)
    const onUp    = () => setClicking(false)

    window.addEventListener('mousemove', onMove)
    document.addEventListener('mouseleave', onLeave)
    window.addEventListener('mousedown', onDown)
    window.addEventListener('mouseup', onUp)
    window.addEventListener('scroll', collectOrbs, { passive: true })

    return () => {
      window.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseleave', onLeave)
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('mouseup', onUp)
      window.removeEventListener('scroll', collectOrbs)
    }
  }, [rawX, rawY])

  return (
    <>
      {/* aura blob */}
      <motion.div
        className="fixed top-0 left-0 pointer-events-none z-[9999] rounded-full"
        style={{
          x: auraX, y: auraY,
          translateX: '-50%', translateY: '-50%',
          width:  hovering ? 80 : clicking ? 52 : 52,
          height: hovering ? 80 : clicking ? 52 : 52,
          background: hovering
            ? 'radial-gradient(circle, rgba(139,92,246,0.22) 0%, transparent 70%)'
            : 'radial-gradient(circle, rgba(139,92,246,0.10) 0%, transparent 70%)',
          opacity: visible ? 1 : 0,
          transition: 'width 0.25s, height 0.25s, opacity 0.3s',
        }}
      />

      {/* ring */}
      <motion.div
        className="fixed top-0 left-0 pointer-events-none z-[9999] rounded-full border"
        style={{
          x: ringX, y: ringY,
          translateX: '-50%', translateY: '-50%',
          width:  hovering ? 40 : clicking ? 18 : 28,
          height: hovering ? 40 : clicking ? 18 : 28,
          borderColor: hovering ? 'rgba(139,92,246,1)' : 'rgba(139,92,246,0.55)',
          boxShadow: hovering ? '0 0 14px rgba(139,92,246,0.6)' : 'none',
          opacity: visible ? 1 : 0,
          transition: 'width 0.2s, height 0.2s, border-color 0.2s, opacity 0.3s',
        }}
      />

      {/* dot */}
      <motion.div
        className="fixed top-0 left-0 pointer-events-none z-[9999] rounded-full"
        style={{
          x: dotX, y: dotY,
          translateX: '-50%', translateY: '-50%',
          width:  clicking ? 3 : 5,
          height: clicking ? 3 : 5,
          background: '#a78bfa',
          boxShadow: '0 0 8px rgba(167,139,250,0.9)',
          opacity: visible ? 1 : 0,
          transition: 'width 0.1s, height 0.1s, opacity 0.3s',
        }}
      />
    </>
  )
}
