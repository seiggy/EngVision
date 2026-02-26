import { useState, useEffect, useCallback } from 'react';
import AnnotationCanvas from './components/AnnotationCanvas';
import AnnotationPanel from './components/AnnotationPanel';
import Toolbar from './components/Toolbar';
import PipelineView from './components/PipelineView';
import * as api from './api';
import type { Annotation, BoundingBox, Point, AccuracyResult, DetectedRegion } from './types';

type AppView = 'pipeline' | 'annotate';
type AnnotationMode = 'select' | 'annotate';

export default function App() {
  const [view, setView] = useState<AppView>('pipeline');
  const [pdfs, setPdfs] = useState<string[]>([]);
  const [selectedPdf, setSelectedPdf] = useState('');
  const [pageNum, setPageNum] = useState(1);
  const [pageCount, setPageCount] = useState(0);
  const [imageWidth, setImageWidth] = useState(0);
  const [imageHeight, setImageHeight] = useState(0);
  const [manualAnnotations, setManualAnnotations] = useState<Annotation[]>([]);
  const [autoDetections, setAutoDetections] = useState<DetectedRegion[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showAuto, setShowAuto] = useState(true);
  const [detecting, setDetecting] = useState(false);
  const [accuracy, setAccuracy] = useState<AccuracyResult | null>(null);
  const [mode, setMode] = useState<AnnotationMode>('select');

  // Load PDF list
  useEffect(() => {
    api.listPdfs().then(files => {
      setPdfs(files);
      if (files.length > 0) setSelectedPdf(files[0]);
    });
  }, []);

  // Load PDF info when selection changes
  useEffect(() => {
    if (!selectedPdf) return;
    api.getPdfInfo(selectedPdf).then(info => {
      setPageCount(info.pageCount);
      setImageWidth(info.width);
      setImageHeight(info.height);
      setPageNum(1);
    });
  }, [selectedPdf]);

  // Load annotations when page changes
  useEffect(() => {
    if (!selectedPdf || !pageNum) return;
    api.getAnnotations(selectedPdf, pageNum).then(({ manual, auto }) => {
      setManualAnnotations(manual);
      setAutoDetections(auto);
      setSelectedId(null);
    });
  }, [selectedPdf, pageNum]);

  // Refresh accuracy when manual annotations change
  useEffect(() => {
    if (!selectedPdf || !pageNum || manualAnnotations.length === 0) {
      setAccuracy(null);
      return;
    }
    api.getAccuracy(selectedPdf, pageNum).then(setAccuracy).catch(() => setAccuracy(null));
  }, [selectedPdf, pageNum, manualAnnotations]);

  const handleAnnotationCreated = useCallback(async (bubbleCenter: Point, bbox: BoundingBox) => {
    if (!selectedPdf) return;
    const ann = await api.saveAnnotation(selectedPdf, pageNum, { bubbleCenter, boundingBox: bbox });
    setManualAnnotations(prev => [...prev, ann]);
    setSelectedId(ann.id);
    // Stay in annotate mode for quick successive annotations
  }, [selectedPdf, pageNum]);

  const handleAnnotationMoved = useCallback((_id: string, bbox: BoundingBox) => {
    setManualAnnotations(prev =>
      prev.map(a => a.id === _id ? { ...a, boundingBox: bbox } : a)
    );
  }, []);

  const handleSelect = useCallback(async (id: string | null) => {
    if (selectedId && selectedId !== id) {
      const ann = manualAnnotations.find(a => a.id === selectedId);
      if (ann && selectedPdf) {
        await api.updateAnnotation(selectedPdf, pageNum, selectedId, ann);
      }
    }
    setSelectedId(id);
  }, [selectedId, manualAnnotations, selectedPdf, pageNum]);

  const handleDelete = useCallback(async (id: string) => {
    if (!selectedPdf) return;
    await api.deleteAnnotation(selectedPdf, pageNum, id);
    setManualAnnotations(prev => prev.filter(a => a.id !== id));
    setSelectedId(null);
  }, [selectedPdf, pageNum]);

  // Delete key shortcut
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) {
        // Don't delete if user is typing in an input
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;
        e.preventDefault();
        handleDelete(selectedId);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [selectedId, handleDelete]);

  const handleUpdate = useCallback(async (id: string, updates: Partial<Annotation>) => {
    setManualAnnotations(prev =>
      prev.map(a => a.id === id ? { ...a, ...updates } : a)
    );
    const ann = manualAnnotations.find(a => a.id === id);
    if (ann && selectedPdf) {
      await api.updateAnnotation(selectedPdf, pageNum, id, { ...ann, ...updates });
    }
  }, [manualAnnotations, selectedPdf, pageNum]);

  const handleDetect = useCallback(async () => {
    if (!selectedPdf) return;
    setDetecting(true);
    try {
      const regions = await api.detectRegions(selectedPdf, pageNum);
      setAutoDetections(regions);
    } finally {
      setDetecting(false);
    }
  }, [selectedPdf, pageNum]);

  const handleExport = useCallback(async () => {
    if (!selectedPdf) return;
    const data = await api.exportAnnotations(selectedPdf);
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedPdf.replace('.pdf', '')}_ground_truth.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [selectedPdf]);

  const handleClearAll = useCallback(async () => {
    if (!selectedPdf) return;
    await api.clearAnnotations(selectedPdf, pageNum);
    setManualAnnotations([]);
    setSelectedId(null);
  }, [selectedPdf, pageNum]);

  const imageUrl = selectedPdf && pageNum ? api.getPageImageUrl(selectedPdf, pageNum) : '';

  // Pipeline view
  if (view === 'pipeline') {
    return (
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        <ViewSwitcher view={view} onViewChange={setView} />
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <PipelineView samplePdfs={pdfs} />
        </div>
      </div>
    );
  }

  // Annotation view
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#0d1117', color: '#eee' }}>
      <ViewSwitcher view={view} onViewChange={setView} />
      <Toolbar
        pdfs={pdfs}
        selectedPdf={selectedPdf}
        pageNum={pageNum}
        pageCount={pageCount}
        showAuto={showAuto}
        detecting={detecting}
        mode={mode}
        onPdfChange={setSelectedPdf}
        onPageChange={setPageNum}
        onDetect={handleDetect}
        onExport={handleExport}
        onClearAll={handleClearAll}
        onToggleAuto={() => setShowAuto(p => !p)}
        onModeChange={setMode}
      />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <div style={{ flex: 1, position: 'relative' }}>
          {imageUrl && imageWidth > 0 && (
            <AnnotationCanvas
              imageUrl={imageUrl}
              imageWidth={imageWidth}
              imageHeight={imageHeight}
              manualAnnotations={manualAnnotations}
              autoDetections={autoDetections}
              selectedId={selectedId}
              showAuto={showAuto}
              mode={mode}
              onAnnotationCreated={handleAnnotationCreated}
              onAnnotationMoved={handleAnnotationMoved}
              onSelect={handleSelect}
            />
          )}
        </div>
        <AnnotationPanel
          annotations={manualAnnotations}
          selectedId={selectedId}
          accuracy={accuracy}
          onSelect={handleSelect}
          onDelete={handleDelete}
          onUpdate={handleUpdate}
        />
      </div>
    </div>
  );
}

function ViewSwitcher({ view, onViewChange }: { view: AppView; onViewChange: (v: AppView) => void }) {
  return (
    <div style={{
      display: 'flex', gap: 0, background: '#010409', borderBottom: '1px solid #30363d',
    }}>
      {(['pipeline', 'annotate'] as AppView[]).map(v => (
        <button
          key={v}
          onClick={() => onViewChange(v)}
          style={{
            background: v === view ? '#161b22' : 'transparent',
            color: v === view ? '#58a6ff' : '#8b949e',
            border: 'none',
            borderBottom: v === view ? '2px solid #58a6ff' : '2px solid transparent',
            padding: '8px 20px',
            fontSize: 13,
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          {v === 'pipeline' ? '🔬 Pipeline' : '✏️ Annotate'}
        </button>
      ))}
    </div>
  );
}
