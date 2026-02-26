import { useRef, useEffect, useState, useCallback } from 'react';
import type { Annotation, DetectedRegion, BoundingBox, Point } from '../types';

type AnnotationMode = 'select' | 'annotate';

interface Props {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  manualAnnotations: Annotation[];
  autoDetections: DetectedRegion[];
  selectedId: string | null;
  showAuto: boolean;
  mode: AnnotationMode;
  onAnnotationCreated: (bubbleCenter: Point, bbox: BoundingBox) => void;
  onAnnotationMoved: (id: string, bbox: BoundingBox) => void;
  onSelect: (id: string | null) => void;
}

type DragMode = 'none' | 'draw' | 'move' | 'resize';
type ResizeHandle = 'nw' | 'ne' | 'sw' | 'se' | null;

export default function AnnotationCanvas({
  imageUrl, imageWidth, imageHeight,
  manualAnnotations, autoDetections, selectedId, showAuto, mode,
  onAnnotationCreated, onAnnotationMoved, onSelect,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [dragMode, setDragMode] = useState<DragMode>('none');
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawEnd, setDrawEnd] = useState<{ x: number; y: number } | null>(null);
  const [moveStart, setMoveStart] = useState<{ x: number; y: number; bbox: BoundingBox } | null>(null);
  const [resizeHandle, setResizeHandle] = useState<ResizeHandle>(null);
  const [pan, setPan] = useState(false);
  const [panStart, setPanStart] = useState<{ x: number; y: number; ox: number; oy: number } | null>(null);

  // Two-step annotation state: step 1 = click bubble, step 2 = draw data box
  const [pendingBubbleCenter, setPendingBubbleCenter] = useState<Point | null>(null);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.onload = () => { imgRef.current = img; setImgLoaded(true); };
    img.src = imageUrl;
    setImgLoaded(false);
  }, [imageUrl]);

  // Fit image to container on load
  useEffect(() => {
    if (!imgLoaded || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const container = canvas.parentElement!;
    const sw = container.clientWidth / imageWidth;
    const sh = container.clientHeight / imageHeight;
    const s = Math.min(sw, sh, 1);
    setScale(s);
    setOffset({ x: 0, y: 0 });
  }, [imgLoaded, imageWidth, imageHeight]);

  // Cancel pending bubble center when mode changes
  useEffect(() => {
    if (mode !== 'annotate') setPendingBubbleCenter(null);
  }, [mode]);

  // Convert mouse position to image coordinates
  const toImageCoords = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left - offset.x) / scale;
    const y = (e.clientY - rect.top - offset.y) / scale;
    return { x: Math.round(x), y: Math.round(y) };
  }, [scale, offset]);

  // Hit test for annotations
  const hitTest = useCallback((px: number, py: number): { id: string; handle: ResizeHandle } | null => {
    const handleSize = 8 / scale;
    for (const a of manualAnnotations) {
      const b = a.boundingBox;
      const corners: [number, number, ResizeHandle][] = [
        [b.x, b.y, 'nw'], [b.x + b.width, b.y, 'ne'],
        [b.x, b.y + b.height, 'sw'], [b.x + b.width, b.y + b.height, 'se'],
      ];
      for (const [cx, cy, h] of corners) {
        if (Math.abs(px - cx) < handleSize && Math.abs(py - cy) < handleSize) {
          return { id: a.id, handle: h };
        }
      }
      if (px >= b.x && px <= b.x + b.width && py >= b.y && py <= b.y + b.height) {
        return { id: a.id, handle: null };
      }
    }
    return null;
  }, [manualAnnotations, scale]);

  // Mouse handlers
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      setPan(true);
      setPanStart({ x: e.clientX, y: e.clientY, ox: offset.x, oy: offset.y });
      return;
    }

    const pos = toImageCoords(e);

    // In annotate mode: two-step workflow
    if (mode === 'annotate') {
      if (!pendingBubbleCenter) {
        // Step 1: click on the bubble to mark its center
        setPendingBubbleCenter(pos);
        return;
      }
      // Step 2: draw the data bounding box
      setDragMode('draw');
      setDrawStart(pos);
      setDrawEnd(pos);
      return;
    }

    // Select mode: select/move/resize existing annotations
    const hit = hitTest(pos.x, pos.y);
    if (hit) {
      onSelect(hit.id);
      const ann = manualAnnotations.find(a => a.id === hit.id)!;
      if (hit.handle) {
        setDragMode('resize');
        setResizeHandle(hit.handle);
        setMoveStart({ x: pos.x, y: pos.y, bbox: { ...ann.boundingBox } });
      } else {
        setDragMode('move');
        setMoveStart({ x: pos.x, y: pos.y, bbox: { ...ann.boundingBox } });
      }
    } else {
      onSelect(null);
    }
  }, [toImageCoords, hitTest, manualAnnotations, onSelect, offset, mode, pendingBubbleCenter]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (pan && panStart) {
      setOffset({
        x: panStart.ox + (e.clientX - panStart.x),
        y: panStart.oy + (e.clientY - panStart.y),
      });
      return;
    }
    const pos = toImageCoords(e);
    if (dragMode === 'draw' && drawStart) {
      setDrawEnd(pos);
    } else if (dragMode === 'move' && moveStart && selectedId) {
      const dx = pos.x - moveStart.x;
      const dy = pos.y - moveStart.y;
      onAnnotationMoved(selectedId, {
        x: moveStart.bbox.x + dx,
        y: moveStart.bbox.y + dy,
        width: moveStart.bbox.width,
        height: moveStart.bbox.height,
      });
    } else if (dragMode === 'resize' && moveStart && selectedId && resizeHandle) {
      const b = moveStart.bbox;
      let { x, y, width, height } = b;
      const dx = pos.x - moveStart.x;
      const dy = pos.y - moveStart.y;
      if (resizeHandle.includes('w')) { x = b.x + dx; width = b.width - dx; }
      if (resizeHandle.includes('e')) { width = b.width + dx; }
      if (resizeHandle.includes('n')) { y = b.y + dy; height = b.height - dy; }
      if (resizeHandle.includes('s')) { height = b.height + dy; }
      if (width > 5 && height > 5) {
        onAnnotationMoved(selectedId, { x, y, width, height });
      }
    }
  }, [pan, panStart, dragMode, drawStart, moveStart, selectedId, resizeHandle, toImageCoords, onAnnotationMoved]);

  const onMouseUp = useCallback(() => {
    if (pan) { setPan(false); setPanStart(null); return; }
    if (dragMode === 'draw' && drawStart && drawEnd && pendingBubbleCenter) {
      const x = Math.min(drawStart.x, drawEnd.x);
      const y = Math.min(drawStart.y, drawEnd.y);
      const w = Math.abs(drawEnd.x - drawStart.x);
      const h = Math.abs(drawEnd.y - drawStart.y);
      if (w > 10 && h > 10) {
        onAnnotationCreated(pendingBubbleCenter, { x, y, width: w, height: h });
        setPendingBubbleCenter(null);
      }
    }
    setDragMode('none');
    setDrawStart(null);
    setDrawEnd(null);
    setMoveStart(null);
    setResizeHandle(null);
  }, [pan, dragMode, drawStart, drawEnd, pendingBubbleCenter, onAnnotationCreated]);

  // Right-click to cancel pending bubble center
  const onContextMenu = useCallback((e: React.MouseEvent) => {
    if (pendingBubbleCenter) {
      e.preventDefault();
      setPendingBubbleCenter(null);
    }
  }, [pendingBubbleCenter]);

  // Use native wheel listener with {passive: false} so preventDefault blocks browser zoom
  const wheelHandler = useCallback((e: WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();

    if (e.ctrlKey || e.metaKey) {
      // Pinch-to-zoom or Ctrl+scroll → zoom in canvas
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY > 0 ? 0.95 : 1.05;
      const newScale = Math.max(0.1, Math.min(5, scale * factor));
      setOffset({
        x: mx - (mx - offset.x) * (newScale / scale),
        y: my - (my - offset.y) * (newScale / scale),
      });
      setScale(newScale);
    } else {
      // Two-finger drag / regular scroll → pan
      setOffset({
        x: offset.x - e.deltaX,
        y: offset.y - e.deltaY,
      });
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

    // Draw auto-detections
    if (showAuto) {
      ctx.lineWidth = 2 / scale;
      for (const d of autoDetections) {
        const b = d.boundingBox;
        // Bubble type (0) = draw as circle; others = draw as rectangle
        if (d.type === 0) {
          const cx = b.x + b.width / 2;
          const cy = b.y + b.height / 2;
          const r = b.width / 2;
          // Circle outline
          ctx.beginPath();
          ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
          ctx.strokeStyle = '#3b82f6';
          ctx.lineWidth = 2 / scale;
          ctx.setLineDash([]);
          ctx.stroke();
          // Semi-transparent fill
          ctx.fillStyle = '#3b82f618';
          ctx.fill();
          // Label above
          ctx.font = `bold ${11 / scale}px sans-serif`;
          ctx.fillStyle = '#3b82f6cc';
          const label = d.label || `#${d.bubbleNumber ?? d.id}`;
          ctx.fillText(label, cx - 8 / scale, cy - r - 6 / scale);
        } else {
          ctx.strokeStyle = '#3b82f6';
          ctx.setLineDash([6 / scale, 4 / scale]);
          ctx.strokeRect(b.x, b.y, b.width, b.height);
          ctx.font = `${12 / scale}px sans-serif`;
          ctx.fillStyle = '#3b82f680';
          ctx.fillText(d.label || `#${d.id}`, b.x, b.y - 4 / scale);
        }
      }
      ctx.setLineDash([]);
    }

    // Draw manual annotations (green, solid)
    for (const a of manualAnnotations) {
      const b = a.boundingBox;
      const isSelected = a.id === selectedId;
      ctx.strokeStyle = isSelected ? '#f59e0b' : '#22c55e';
      ctx.lineWidth = (isSelected ? 3 : 2) / scale;
      ctx.strokeRect(b.x, b.y, b.width, b.height);

      // Semi-transparent fill
      ctx.fillStyle = isSelected ? '#f59e0b18' : '#22c55e10';
      ctx.fillRect(b.x, b.y, b.width, b.height);

      // Draw bubble center marker if present
      if (a.bubbleCenter) {
        const bc = a.bubbleCenter;
        const markerR = 8 / scale;
        ctx.beginPath();
        ctx.arc(bc.x, bc.y, markerR, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? '#f59e0bcc' : '#ef4444cc';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5 / scale;
        ctx.stroke();
        // Crosshair
        ctx.beginPath();
        ctx.moveTo(bc.x - markerR * 1.5, bc.y);
        ctx.lineTo(bc.x + markerR * 1.5, bc.y);
        ctx.moveTo(bc.x, bc.y - markerR * 1.5);
        ctx.lineTo(bc.x, bc.y + markerR * 1.5);
        ctx.strokeStyle = '#ffffffaa';
        ctx.lineWidth = 1 / scale;
        ctx.stroke();
        // Line from bubble center to data box
        ctx.beginPath();
        ctx.setLineDash([4 / scale, 3 / scale]);
        ctx.moveTo(bc.x, bc.y);
        const boxCx = b.x + b.width / 2, boxCy = b.y + b.height / 2;
        ctx.lineTo(boxCx, boxCy);
        ctx.strokeStyle = isSelected ? '#f59e0b88' : '#22c55e88';
        ctx.lineWidth = 1.5 / scale;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Label
      ctx.font = `bold ${13 / scale}px sans-serif`;
      ctx.fillStyle = isSelected ? '#f59e0b' : '#22c55e';
      const label = a.bubbleNumber ? `#${a.bubbleNumber}` : (a.label || a.id.slice(0, 6));
      ctx.fillText(label, b.x + 4 / scale, b.y - 5 / scale);

      // Resize handles on selected
      if (isSelected) {
        const hs = 6 / scale;
        ctx.fillStyle = '#f59e0b';
        for (const [cx, cy] of [
          [b.x, b.y], [b.x + b.width, b.y],
          [b.x, b.y + b.height], [b.x + b.width, b.y + b.height],
        ]) {
          ctx.fillRect(cx - hs / 2, cy - hs / 2, hs, hs);
        }
      }
    }

    // Draw pending bubble center (step 1 complete, waiting for box draw)
    if (pendingBubbleCenter) {
      const bc = pendingBubbleCenter;
      const r = 12 / scale;
      // Pulsing red circle
      ctx.beginPath();
      ctx.arc(bc.x, bc.y, r, 0, Math.PI * 2);
      ctx.fillStyle = '#ef444488';
      ctx.fill();
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2 / scale;
      ctx.stroke();
      // Crosshair
      ctx.beginPath();
      ctx.moveTo(bc.x - r * 2, bc.y);
      ctx.lineTo(bc.x + r * 2, bc.y);
      ctx.moveTo(bc.x, bc.y - r * 2);
      ctx.lineTo(bc.x, bc.y + r * 2);
      ctx.strokeStyle = '#ef4444aa';
      ctx.lineWidth = 1 / scale;
      ctx.stroke();
      // Instruction text
      ctx.font = `bold ${14 / scale}px sans-serif`;
      ctx.fillStyle = '#ef4444';
      ctx.fillText('Now draw a box around the data ↓', bc.x + r * 2, bc.y - r);
    }

    // Draw current drawing rect
    if (dragMode === 'draw' && drawStart && drawEnd) {
      const x = Math.min(drawStart.x, drawEnd.x);
      const y = Math.min(drawStart.y, drawEnd.y);
      const w = Math.abs(drawEnd.x - drawStart.x);
      const h = Math.abs(drawEnd.y - drawStart.y);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2 / scale;
      ctx.setLineDash([4 / scale, 3 / scale]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }

    ctx.restore();
  }, [
    imgLoaded, scale, offset, manualAnnotations, autoDetections,
    selectedId, showAuto, dragMode, drawStart, drawEnd, imageWidth, imageHeight,
    pendingBubbleCenter,
  ]);

  const cursor = pan ? 'grabbing'
    : mode === 'annotate' ? (pendingBubbleCenter ? 'crosshair' : 'cell')
    : dragMode === 'draw' ? 'crosshair' : 'default';

  return (
    <canvas
      ref={canvasRef}
      style={{ cursor, display: 'block', width: '100%', height: '100%' }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onContextMenu={onContextMenu}
    />
  );
}
