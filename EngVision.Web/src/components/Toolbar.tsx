type AnnotationMode = 'select' | 'annotate';

interface Props {
  pdfs: string[];
  selectedPdf: string;
  pageNum: number;
  pageCount: number;
  showAuto: boolean;
  detecting: boolean;
  mode: AnnotationMode;
  onPdfChange: (pdf: string) => void;
  onPageChange: (page: number) => void;
  onDetect: () => void;
  onExport: () => void;
  onClearAll: () => void;
  onToggleAuto: () => void;
  onModeChange: (mode: AnnotationMode) => void;
}

export default function Toolbar({
  pdfs, selectedPdf, pageNum, pageCount, showAuto, detecting, mode,
  onPdfChange, onPageChange, onDetect, onExport, onClearAll, onToggleAuto, onModeChange,
}: Props) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '8px 16px', background: '#16213e',
      borderBottom: '1px solid #333', flexWrap: 'wrap',
    }}>
      <span style={{ fontWeight: 700, color: '#22c55e', fontSize: 15 }}>
        EngVision Annotator
      </span>

      <select
        value={selectedPdf}
        onChange={e => onPdfChange(e.target.value)}
        style={selectStyle}
      >
        {pdfs.map(p => <option key={p} value={p}>{p}</option>)}
      </select>

      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <button
          onClick={() => onPageChange(Math.max(1, pageNum - 1))}
          disabled={pageNum <= 1}
          style={btnStyle}
        >◀</button>
        <span style={{ fontSize: 13, minWidth: 80, textAlign: 'center' }}>
          Page {pageNum} / {pageCount}
        </span>
        <button
          onClick={() => onPageChange(Math.min(pageCount, pageNum + 1))}
          disabled={pageNum >= pageCount}
          style={btnStyle}
        >▶</button>
      </div>

      {/* Mode toggle */}
      <div style={{ display: 'flex', borderRadius: 4, overflow: 'hidden', border: '1px solid #444' }}>
        <button
          onClick={() => onModeChange('select')}
          style={{
            ...btnStyle,
            border: 'none', borderRadius: 0,
            background: mode === 'select' ? '#1e3a5f' : '#1a1a2e',
            color: mode === 'select' ? '#fff' : '#888',
          }}
        >🖱️ Select</button>
        <button
          onClick={() => onModeChange('annotate')}
          style={{
            ...btnStyle,
            border: 'none', borderRadius: 0,
            background: mode === 'annotate' ? '#14532d' : '#1a1a2e',
            color: mode === 'annotate' ? '#4ade80' : '#888',
          }}
        >🎯 Annotate</button>
      </div>

      <button onClick={onDetect} disabled={detecting} style={{ ...btnStyle, background: '#1e3a5f' }}>
        {detecting ? '⏳ Detecting...' : '🔍 Auto-Detect'}
      </button>

      <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, cursor: 'pointer' }}>
        <input type="checkbox" checked={showAuto} onChange={onToggleAuto} />
        Show CV detections
      </label>

      <div style={{ flex: 1 }} />

      <button onClick={onExport} style={{ ...btnStyle, background: '#14532d' }}>
        📥 Export Ground Truth
      </button>

      <button onClick={() => { if (confirm('Clear all annotations on this page?')) onClearAll(); }}
        style={{ ...btnStyle, background: '#7f1d1d', color: '#fca5a5' }}>
        🗑️ Clear All
      </button>

      <span style={{ fontSize: 11, color: '#666' }}>
        {mode === 'annotate'
          ? '1) Click bubble center → 2) Draw box around data | Right-click: cancel | Pan: two-finger scroll | Zoom: pinch/Ctrl+scroll'
          : 'Click to select | Pan: two-finger scroll or Alt+drag | Zoom: pinch/Ctrl+scroll'}
      </span>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: '5px 12px',
  background: '#1a1a2e',
  border: '1px solid #444',
  borderRadius: 4,
  color: '#ccc',
  fontSize: 12,
  cursor: 'pointer',
};

const selectStyle: React.CSSProperties = {
  padding: '5px 8px',
  background: '#0d1b2a',
  border: '1px solid #444',
  borderRadius: 4,
  color: '#eee',
  fontSize: 13,
};
