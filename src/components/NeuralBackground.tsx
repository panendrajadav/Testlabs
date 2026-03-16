"use client";
import { useEffect, useRef } from "react";

export default function NeuralBackground() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d")!;
    let raf: number;
    const N = 55;
    type P = { x:number;y:number;vx:number;vy:number;r:number };
    const pts: P[] = [];
    const resize = () => { c.width = window.innerWidth; c.height = window.innerHeight; };
    resize();
    window.addEventListener("resize", resize);
    for (let i=0;i<N;i++) pts.push({ x:Math.random()*c.width, y:Math.random()*c.height, vx:(Math.random()-.5)*.35, vy:(Math.random()-.5)*.35, r:Math.random()*1.8+.8 });
    const draw = () => {
      ctx.clearRect(0,0,c.width,c.height);
      pts.forEach(p => { p.x+=p.vx; p.y+=p.vy; if(p.x<0||p.x>c.width)p.vx*=-1; if(p.y<0||p.y>c.height)p.vy*=-1; });
      for (let i=0;i<N;i++) for (let j=i+1;j<N;j++) {
        const dx=pts[i].x-pts[j].x, dy=pts[i].y-pts[j].y, d=Math.sqrt(dx*dx+dy*dy);
        if (d<140) { ctx.beginPath(); ctx.strokeStyle=`rgba(59,130,246,${(1-d/140)*.12})`; ctx.lineWidth=.7; ctx.moveTo(pts[i].x,pts[i].y); ctx.lineTo(pts[j].x,pts[j].y); ctx.stroke(); }
      }
      pts.forEach(p => { ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fillStyle="rgba(99,179,237,.45)"; ctx.fill(); });
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize",resize); };
  }, []);
  return <canvas ref={ref} className="fixed inset-0 pointer-events-none z-0" style={{opacity:.55}} />;
}
