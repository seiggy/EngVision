import { useRef, useEffect, useState, useCallback } from 'react';
import type { BubbleResult, DimensionMatch } from '../types';

interface Props {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  bubbles: BubbleResult[];
  dimensionMap: Record<string, DimensionMatch>;
  selectedBubble: number | null;
  onSelectBubble: (bubbleNumber: number | null) => void;
  showOverlay: boolean;
  thresholds: { error: number; warning: number };
}

export default function PipelineCanvas({
  imageUrl, imageWidth, imageHeight,
  bubbles, dimensionMap, selectedBubble, onSelectBubble, showOverlay, thresholds,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [pan, setPan] = useState(false);
  const [panStart, setPanStart] = useState<{ x: number; y: number; ox: number; oy: number } | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.onload = () => { imgRef.current = img; setImgLoaded(true); };
    img.src = imageUrl;
    setImgLoaded(false);
  }, [imageUrl]);

  // Fit to container
  useEffect(() => {
    if (!imgLoaded || !canvasRef.current) return;
    const container = canvasRef.current.parentElement!;
    const sw = container.clientWidth / imageWidth;
    const sh = container.clientHeight / imageHeight;
    setScale(Math.min(sw, sh, 1));
    setOffset({ x: 0, y: 0 });
  }, [imgLoaded, imageWidth, imageHeight]);

  const toImageCoords = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left - offset.x) / scale,
      y: (e.clientY - rect.top - offset.y) / scale,
    };
  }, [scale, offset]);

  // Hit test: find bubble near click
  const findBubbleAt = useCallback((px: number, py: number): number | null => {
    for (const b of bubbles) {
      const dx = px - b.cx, dy = py - b.cy;
      if (Math.sqrt(dx * dx + dy * dy) < b.radius + 15) return b.bubbleNumber;
    }
    return null;
  }, [bubbles]);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      setPan(true);
      setPanStart({ x: e.clientX, y: e.clientY, ox: offset.x, oy: offset.y });
      return;
    }
    const pos = toImageCoords(e);
    const hit = findBubbleAt(pos.x, pos.y);
    onSelectBubble(hit);
  }, [toImageCoords, findBubbleAt, onSelectBubble, offset]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (pan && panStart) {
      setOffset({ x: panStart.ox + (e.clientX - panStart.x), y: panStart.oy + (e.clientY - panStart.y) });
      return;
    }
    const pos = toImageCoords(e);
    setHovered(findBubbleAt(pos.x, pos.y));
  }, [pan, panStart, toImageCoords, findBubbleAt]);

  const onMouseUp = useCallback(() => { setPan(false); setPanStart(null); }, []);

  // Wheel zoom/pan
  const wheelHandler = useCallback((e: WheelEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    if (e.ctrlKey || e.metaKey) {
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const factor = e.deltaY > 0 ? 0.95 : 1.05;
      const newScale = Math.max(0.1, Math.min(5, scale * factor));
      setOffset({ x: mx - (mx - offset.x) * (newScale / scale), y: my - (my - offset.y) * (newScale / scale) });
      setScale(newScale);
    } else {
      setOffset({ x: offset.x - e.deltaX, y: offset.y - e.deltaY });
    }
  }, [scale, offset]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener('wheel', wheelHandler, { passive: false });
    return () => canvas.removeEventListener('wheel', wheelHandler);
  }, [wheelHandler]);

  // Render
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !imgLoaded) return;

    const container = canvas.parentElement!;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    const ctx = canvas.getContext('2d')!;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(offset.x, offset.y);
    ctx.scale(scale, scale);

    // Draw page image
    ctx.drawImage(img, 0, 0, imageWidth, imageHeight);

    if (showOverlay) {
      // Draw each bubble with color-coding
      for (const bubble of bubbles) {
        const match = dimensionMap?.[String(bubble.bubbleNumber)];
        const conf = match?.confidence ?? 0;
        const isSelected = bubble.bubbleNumber === selectedBubble;
        const isHovered = bubble.bubbleNumber === hovered;

        // Color based on confidence thresholds
        let color: string;
        let bgColor: string;
        if (conf < thresholds.error) {
          color = '#ef4444'; bgColor = '#ef444430'; // red/error
        } else if (conf < thresholds.warning) {
          color = '#f59e0b'; bgColor = '#f59e0b30'; // amber/warning
        } else {
          color = '#22c55e'; bgColor = '#22c55e30'; // green/success
        }

        const lineWidth = (isSelected || isHovered ? 4 : 2) / scale;
        const radius = bubble.radius + 4;

        // Glow effect for selected
        if (isSelected) {
          ctx.beginPath();
          ctx.arc(bubble.cx, bubble.cy, radius + 6, 0, Math.PI * 2);
          ctx.fillStyle = bgColor;
          ctx.fill();
        }

        // Circle
        ctx.beginPath();
        ctx.arc(bubble.cx, bubble.cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.stroke();

        // Semi-transparent fill on hover/select
        if (isSelected || isHovered) {
          ctx.fillStyle = bgColor;
          ctx.fill();
        }

        // Bubble number label (top)
        ctx.font = `bold ${Math.max(13, 13 / scale)}px sans-serif`;
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        const label = `#${bubble.bubbleNumber}`;
        ctx.fillText(label, bubble.cx, bubble.cy - radius - 6 / scale);

        // Dimension text label (right side)
        if (match?.dimension) {
          const dimText = match.dimension.length > 25
            ? match.dimension.slice(0, 25) + '…'
            : match.dimension;

          // Background pill for readability
          ctx.font = `${Math.max(11, 11 / scale)}px sans-serif`;
          ctx.textAlign = 'left';
          const textW = ctx.measureText(dimText).width;
          const pillX = bubble.cx + radius + 8 / scale;
          const pillY = bubble.cy - 7 / scale;
          const pillH = 16 / scale;

          ctx.fillStyle = '#000000aa';
          ctx.beginPath();
          ctx.roundRect(pillX - 3 / scale, pillY - pillH + 2 / scale, textW + 6 / scale, pillH + 2 / scale, 3 / scale);
          ctx.fill();

          ctx.fillStyle = color;
          ctx.fillText(dimText, pillX, pillY);
        }

        // Source indicator dot (bottom-right of bubble)
        if (match) {
          const dotR = 4 / scale;
          const dotX = bubble.cx + radius * 0.7;
          const dotY = bubble.cy + radius * 0.7;
          ctx.beginPath();
          ctx.arc(dotX, dotY, dotR, 0, Math.PI * 2);
          ctx.fillStyle = match.source === 'Both' ? '#3b82f6'
            : match.source === 'LLM' ? '#a855f7'
            : match.source === 'Tesseract' ? '#06b6d4'
            : '#6b7280';
          ctx.fill();
        }
      }

      ctx.textAlign = 'start'; // reset
    }

    ctx.restore();
  }, [imgLoaded, scale, offset, bubbles, dimensionMap, selectedBubble, hovered, showOverlay, thresholds, imageWidth, imageHeight]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        cursor: pan ? 'grabbing' : (hovered ? 'pointer' : 'default'),
        display: 'block',
        width: '100%',
        height: '100%',
      }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    />
  );
}
