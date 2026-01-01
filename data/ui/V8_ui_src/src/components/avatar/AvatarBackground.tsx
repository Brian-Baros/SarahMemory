import { useEffect, useRef } from "react";

interface AvatarBackgroundProps {
  className?: string;
}

/**
 * Tron-style animated background.
 * Later: you can overlay an image/video background provided by backend “scene spec”
 * inside Avatar3D (SceneBackground) without changing this file.
 */
export function AvatarBackground({ className }: AvatarBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId = 0;

    type Particle = {
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      alpha: number;
      color: string;
    };

    type CircuitLine = {
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      progress: number;
      speed: number;
      color: string;
      glowIntensity: number;
    };

    let particles: Particle[] = [];
    let circuitLines: CircuitLine[] = [];

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    const createParticles = () => {
      particles = [];
      const count = Math.floor((canvas.width * canvas.height) / 8000);
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.3,
          vy: (Math.random() - 0.5) * 0.3,
          size: Math.random() * 2 + 0.5,
          alpha: Math.random() * 0.5 + 0.2,
          color: Math.random() > 0.5 ? "#00d4ff" : "#00ffcc",
        });
      }
    };

    const createCircuitLines = () => {
      circuitLines = [];
      const count = 8;
      for (let i = 0; i < count; i++) {
        const isHorizontal = Math.random() > 0.5;
        const startX = isHorizontal ? 0 : Math.random() * canvas.width;
        const startY = isHorizontal ? Math.random() * canvas.height : 0;
        const endX = isHorizontal ? canvas.width : startX;
        const endY = isHorizontal ? startY : canvas.height;

        circuitLines.push({
          x1: startX,
          y1: startY,
          x2: endX,
          y2: endY,
          progress: Math.random(),
          speed: 0.002 + Math.random() * 0.003,
          color: Math.random() > 0.5 ? "rgba(0, 212, 255, 0.6)" : "rgba(0, 255, 204, 0.6)",
          glowIntensity: 0.3 + Math.random() * 0.4,
        });
      }
    };

    const drawGrid = () => {
      ctx.strokeStyle = "rgba(0, 212, 255, 0.05)";
      ctx.lineWidth = 0.5;
      const gridSize = 40;

      for (let x = 0; x < canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }

      for (let y = 0; y < canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }
    };

    const drawParticle = (p: Particle) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.alpha;
      ctx.fill();
      ctx.globalAlpha = 1;
    };

    const drawCircuitLine = (line: CircuitLine) => {
      const x = line.x1 + (line.x2 - line.x1) * line.progress;
      const y = line.y1 + (line.y2 - line.y1) * line.progress;

      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 30);
      gradient.addColorStop(0, line.color);
      gradient.addColorStop(1, "transparent");

      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.globalAlpha = line.glowIntensity;
      ctx.fill();
      ctx.globalAlpha = 1;

      ctx.beginPath();
      ctx.arc(x, y, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawGrid();

      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        drawParticle(p);
      });

      circuitLines.forEach((line) => {
        line.progress += line.speed;
        if (line.progress > 1) {
          line.progress = 0;
          line.glowIntensity = 0.3 + Math.random() * 0.4;
        }
        drawCircuitLine(line);
      });

      animationId = requestAnimationFrame(animate);
    };

    const onResize = () => {
      resize();
      createParticles();
      createCircuitLines();
    };

    onResize();
    animate();
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 w-full h-full ${className || ""}`}
      style={{ background: "linear-gradient(135deg, hsl(220 20% 7%) 0%, hsl(220 25% 4%) 100%)" }}
    />
  );
}
