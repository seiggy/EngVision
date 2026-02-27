import { useState, useCallback, useRef, Component, type ReactNode, type ErrorInfo } from 'react';
import PipelineCanvas from './PipelineCanvas';
import PipelineDetailPanel from './PipelineDetailPanel';
import PipelineProgressView from './PipelineProgressView';
import { usePipelineStream } from '../hooks/usePipelineStream';
import * as api from '../api';
import type { PipelineResult } from '../types';

// Error boundary to prevent white-screen crashes
class ErrorBoundary extends Component<
  { children: ReactNode; onReset: () => void },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('PipelineView crashed:', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          height: '100%', color: '#ef4444', padding: 32, textAlign: 'center',
        }}>
          <p style={{ fontSize: 48, margin: 0 }}>💥</p>
          <p style={{ fontSize: 16, margin: '16px 0 8px', color: '#c9d1d9' }}>
            Something went wrong while rendering results
          </p>
          <p style={{ fontSize: 13, color: '#8b949e', maxWidth: 500, wordBreak: 'break-word' }}>
            {this.state.error.message}
          </p>
          <button
            onClick={() => { this.setState({ error: null }); this.props.onReset(); }}
            style={{
              marginTop: 16, background: '#238636', color: '#fff', border: 'none',
              borderRadius: 6, padding: '8px 20px', fontSize: 14, cursor: 'pointer',
            }}
          >
            Reset &amp; Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

interface Props {
  samplePdfs: string[];
}

export default function PipelineView({ samplePdfs }: Props) {
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [selectedBubble, setSelectedBubble] = useState<number | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [pageNum, setPageNum] = useState(1);
  const [dragOver, setDragOver] = useState(false);
  const [thresholds, setThresholds] = useState({ error: 0.3, warning: 0.8 });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const stream = usePipelineStream();

  // When stream completes, promote its result
  const prevStreamResult = useRef<PipelineResult | null>(null);
  if (stream.result && stream.result !== prevStreamResult.current) {
    prevStreamResult.current = stream.result;
    // Schedule state update outside render
    queueMicrotask(() => {
      setResult(stream.result);
      setPageNum(1);
    });
  }

  const runOnFile = useCallback(async (file: File) => {
    setResult(null);
    setSelectedBubble(null);
    stream.runStream('upload', file);
  }, [stream.runStream]);

  const runOnSample = useCallback(async (filename: string) => {
    setResult(null);
    setSelectedBubble(null);
    stream.runStream('sample', filename);
  }, [stream.runStream]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file?.type === 'application/pdf') runOnFile(file);
  }, [runOnFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) runOnFile(file);
  }, [runOnFile]);

  const handleReset = useCallback(() => {
    setResult(null);
    stream.stop();
    setSelectedBubble(null);
  }, [stream.stop]);

  const imageUrl = result
    ? api.getPipelinePageImageUrl(result.runId, pageNum)
    : '';

  return (
    <ErrorBoundary onReset={handleReset}>
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#0d1117', color: '#eee' }}>
      {/* Top bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '8px 16px', background: '#161b22', borderBottom: '1px solid #30363d',
        flexWrap: 'wrap',
      }}>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: '#58a6ff' }}>
          EngVision Pipeline
        </h2>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={stream.running}
            style={btnStyle}
          >
            📁 Upload PDF
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />

          {samplePdfs.length > 0 && (
            <select
              disabled={stream.running}
              onChange={e => { if (e.target.value) runOnSample(e.target.value); }}
              style={{ ...selectStyle, minWidth: 200 }}
              defaultValue=""
            >
              <option value="" disabled>Run on sample PDF...</option>
              {samplePdfs.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          )}
        </div>

        {stream.running && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#f59e0b' }}>
            <Spinner />
            <span style={{ fontSize: 13 }}>
              {stream.steps.find(s => s.status === 'running')?.message || 'Processing...'}
            </span>
          </div>
        )}

        {result && (
          <>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
              <label style={{ fontSize: 12, color: '#8b949e', display: 'flex', alignItems: 'center', gap: 4 }}>
                <input
                  type="checkbox"
                  checked={showOverlay}
                  onChange={() => setShowOverlay(p => !p)}
                />
                Show annotations
              </label>

              {result.pageCount > 1 && (
                <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                  <button
                    onClick={() => setPageNum(p => Math.max(1, p - 1))}
                    disabled={pageNum <= 1}
                    style={smallBtnStyle}
                  >◀</button>
                  <span style={{ fontSize: 12, minWidth: 60, textAlign: 'center' }}>
                    Page {pageNum}/{result.pageCount}
                  </span>
                  <button
                    onClick={() => setPageNum(p => Math.min(result.pageCount, p + 1))}
                    disabled={pageNum >= result.pageCount}
                    style={smallBtnStyle}
                  >▶</button>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Main content */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Canvas area */}
        <div
          style={{
            flex: 1,
            position: 'relative',
            border: dragOver ? '2px dashed #58a6ff' : '2px solid transparent',
          }}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          {result && result.status === 'complete' ? (
            <PipelineCanvas
              imageUrl={imageUrl}
              imageWidth={result.imageWidth}
              imageHeight={result.imageHeight}
              bubbles={pageNum === 1 ? result.bubbles : []}
              dimensionMap={result.dimensionMap}
              selectedBubble={selectedBubble}
              onSelectBubble={setSelectedBubble}
              showOverlay={showOverlay && pageNum === 1}
              thresholds={thresholds}
            />
          ) : (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              height: '100%', color: '#8b949e',
            }}>
              {stream.running ? (
                <PipelineProgressView
                  steps={stream.steps}
                  bubbleStatuses={stream.bubbleStatuses}
                  currentBubble={stream.currentBubble}
                  elapsedMs={stream.elapsedMs}
                  llmCalls={stream.llmCalls}
                  totalBubbles={
                    stream.steps.find(s => s.name === 'Detect bubbles' && s.status === 'complete')
                      ?.detail?.bubbleCount as number ?? 0
                  }
                  error={stream.error}
                />
              ) : result?.status === 'error' ? (
                <div style={{ textAlign: 'center', color: '#ef4444' }}>
                  <p style={{ fontSize: 48, margin: 0 }}>⚠</p>
                  <p>{result.error}</p>
                </div>
              ) : (
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: 48, margin: 0, opacity: 0.5 }}>📄</p>
                  <p style={{ fontSize: 16, margin: '16px 0 8px' }}>
                    Drop a PDF here or use the buttons above
                  </p>
                  <p style={{ fontSize: 13 }}>
                    The pipeline will detect bubbles, OCR numbers, extract table data, and validate dimensions.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Detail panel */}
        {result && result.status === 'complete' && (
          <PipelineDetailPanel
            result={result}
            selectedBubble={selectedBubble}
            onSelectBubble={setSelectedBubble}
            thresholds={thresholds}
            onThresholdsChange={setThresholds}
          />
        )}
      </div>
    </div>
    </ErrorBoundary>
  );
}

function Spinner({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" style={{ animation: 'spin 1s linear infinite' }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="3" strokeDasharray="31.4 31.4" strokeLinecap="round" />
    </svg>
  );
}

const btnStyle: React.CSSProperties = {
  background: '#238636', color: '#fff', border: 'none', borderRadius: 6,
  padding: '6px 14px', fontSize: 13, fontWeight: 600, cursor: 'pointer',
};

const smallBtnStyle: React.CSSProperties = {
  background: '#21262d', color: '#c9d1d9', border: '1px solid #30363d', borderRadius: 4,
  padding: '2px 8px', fontSize: 12, cursor: 'pointer',
};

const selectStyle: React.CSSProperties = {
  background: '#21262d', color: '#c9d1d9', border: '1px solid #30363d', borderRadius: 6,
  padding: '6px 8px', fontSize: 13,
};
