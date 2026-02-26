import { useState } from 'react';
import type { BubbleResult, DimensionMatch, PipelineResult } from '../types';

interface Props {
  result: PipelineResult;
  selectedBubble: number | null;
  onSelectBubble: (bubbleNumber: number | null) => void;
  thresholds: { error: number; warning: number };
  onThresholdsChange: (t: { error: number; warning: number }) => void;
}

export default function PipelineDetailPanel({ result, selectedBubble, onSelectBubble, thresholds, onThresholdsChange }: Props) {
  const selectedMatch = selectedBubble != null
    ? result.dimensionMap[String(selectedBubble)]
    : null;
  const selectedBubbleData = selectedBubble != null
    ? result.bubbles.find(b => b.bubbleNumber === selectedBubble)
    : null;

  // Compute dynamic counts based on thresholds
  const dims = Object.values(result.dimensionMap ?? {});
  const errorCount = dims.filter(d => d.confidence < thresholds.error).length;
  const warningCount = dims.filter(d => d.confidence >= thresholds.error && d.confidence < thresholds.warning).length;
  const successCount = dims.filter(d => d.confidence >= thresholds.warning).length;

  const [showMetrics, setShowMetrics] = useState(false);

  return (
    <div style={{
      width: 360, background: '#161b22', borderLeft: '1px solid #30363d',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      {/* Summary stats */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid #30363d' }}>
        <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#c9d1d9' }}>Pipeline Results</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 6 }}>
          <StatCard label="Total" value={result.totalBubbles} color="#c9d1d9" />
          <StatCard label="Success" value={successCount} color="#22c55e" />
          <StatCard label="Warnings" value={warningCount} color="#f59e0b" />
          <StatCard label="Errors" value={errorCount} color="#ef4444" />
        </div>
        <div style={{ marginTop: 8, fontSize: 12, color: '#8b949e' }}>
          {result.pdfFilename} · {result.pageCount} pages · {result.imageWidth}×{result.imageHeight}
        </div>
      </div>

      {/* Threshold controls */}
      <div style={{ padding: '8px 16px', borderBottom: '1px solid #30363d', background: '#0d1117' }}>
        <div style={{ fontSize: 12, color: '#8b949e', fontWeight: 600, marginBottom: 6 }}>
          CONFIDENCE THRESHOLDS
        </div>
        <ThresholdSlider
          label="Error below"
          value={thresholds.error}
          color="#ef4444"
          onChange={v => onThresholdsChange({ ...thresholds, error: v })}
        />
        <ThresholdSlider
          label="Warning below"
          value={thresholds.warning}
          color="#f59e0b"
          onChange={v => onThresholdsChange({ ...thresholds, warning: v })}
          min={thresholds.error}
        />
      </div>

      {/* Metrics section */}
      <div style={{ padding: '4px 16px', borderBottom: '1px solid #30363d' }}>
        <button
          onClick={() => setShowMetrics(p => !p)}
          style={{
            background: 'none', border: 'none', color: '#58a6ff', cursor: 'pointer',
            padding: '4px 0', fontSize: 12, fontWeight: 600,
          }}
        >
          {showMetrics ? '▾' : '▸'} Processing Metrics
        </button>
        {showMetrics && (
          <div style={{ paddingBottom: 8, fontSize: 12 }}>
            {result.metrics && (
              <div style={{ marginBottom: 8 }}>
                <MetricRow label="Total time" value={formatMs(result.metrics.totalDurationMs)} />
                <MetricRow label="  Render" value={formatMs(result.metrics.renderDurationMs)} />
                <MetricRow label="  Detect" value={formatMs(result.metrics.detectDurationMs)} />
                <MetricRow label="  OCR" value={formatMs(result.metrics.ocrDurationMs)} />
                <MetricRow label="  LLM" value={formatMs(result.metrics.llmDurationMs)} />
                <MetricRow label="  Merge" value={formatMs(result.metrics.mergeDurationMs)} />
                <MetricRow label="Peak memory" value={`${result.metrics.peakMemoryMb} MB`} />
              </div>
            )}
            {result.tokenUsage && (
              <div>
                <div style={{ color: '#8b949e', fontWeight: 600, marginBottom: 2 }}>LLM Token Usage</div>
                <MetricRow label="Input tokens" value={result.tokenUsage.inputTokens.toLocaleString()} />
                <MetricRow label="Output tokens" value={result.tokenUsage.outputTokens.toLocaleString()} />
                <MetricRow label="Total tokens" value={result.tokenUsage.totalTokens.toLocaleString()} />
                <MetricRow label="LLM calls" value={String(result.tokenUsage.llmCalls)} />
              </div>
            )}
            {!result.metrics && !result.tokenUsage && (
              <div style={{ color: '#8b949e' }}>No metrics available</div>
            )}
          </div>
        )}
      </div>

      {/* Selected bubble detail */}
      {selectedBubbleData && selectedMatch && (
        <div style={{ padding: '12px 16px', borderBottom: '1px solid #30363d', background: '#0d1117' }}>
          <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#c9d1d9' }}>
            Balloon #{selectedBubble}
          </h4>
          <DetailRow label="Position" value={`(${selectedBubbleData.cx}, ${selectedBubbleData.cy})`} />
          <DetailRow label="Radius" value={`${selectedBubbleData.radius}px`} />
          <DetailRow
            label="Dimension"
            value={selectedMatch.dimension ?? '—'}
            color={selectedMatch.dimension ? '#22c55e' : '#ef4444'}
          />
          <DetailRow label="Source" value={selectedMatch.source}>
            <SourceBadge source={selectedMatch.source} />
          </DetailRow>
          <DetailRow label="Confidence">
            <ConfidenceBar value={selectedMatch.confidence} thresholds={thresholds} />
          </DetailRow>
          {/* Always show source values when available */}
          {selectedMatch.source !== 'None' && (
            <div style={{
              marginTop: 8, padding: 8, borderRadius: 4, fontSize: 12,
              background: selectedMatch.confidence >= 1.0 ? '#21262d'
                : selectedMatch.confidence < thresholds.error ? '#ef444415'
                : selectedMatch.confidence < thresholds.warning ? '#f59e0b15' : '#22c55e10',
              border: `1px solid ${selectedMatch.confidence >= 1.0 ? '#30363d'
                : selectedMatch.confidence < thresholds.error ? '#ef444440'
                : selectedMatch.confidence < thresholds.warning ? '#f59e0b40' : '#22c55e30'}`,
            }}>
              {selectedMatch.confidence < 1.0 && (
                <div style={{
                  color: selectedMatch.confidence < thresholds.error ? '#ef4444' : selectedMatch.confidence < thresholds.warning ? '#f59e0b' : '#22c55e',
                  fontWeight: 600, marginBottom: 4,
                }}>
                  {selectedMatch.confidence < thresholds.error ? '✗ Error' : selectedMatch.confidence < thresholds.warning ? '⚠ Warning' : '≈ Close Match'}
                  {' '}({(selectedMatch.confidence * 100).toFixed(1)}%)
                </div>
              )}
              <div style={{ color: '#c9d1d9', marginBottom: 2 }}>
                <span style={{ color: '#06b6d4' }}>Tesseract:</span>{' '}
                <code>{selectedMatch.tesseractValue ?? '—'}</code>
              </div>
              <div style={{ color: '#c9d1d9' }}>
                <span style={{ color: '#a855f7' }}>LLM:</span>{' '}
                <code>{selectedMatch.llmValue ?? '—'}</code>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Bubble list */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
        <div style={{ padding: '4px 16px', fontSize: 12, color: '#8b949e', fontWeight: 600 }}>
          ALL BALLOONS
        </div>
        {result.bubbles.map(b => {
          const match = result.dimensionMap?.[String(b.bubbleNumber)];
          const conf = match?.confidence ?? 0;
          const isSelected = b.bubbleNumber === selectedBubble;
          const dotColor = conf < thresholds.error ? '#ef4444'
            : conf < thresholds.warning ? '#f59e0b'
            : '#22c55e';

          return (
            <div
              key={b.bubbleNumber}
              onClick={() => onSelectBubble(isSelected ? null : b.bubbleNumber)}
              style={{
                padding: '6px 16px',
                cursor: 'pointer',
                background: isSelected ? '#1f6feb30' : 'transparent',
                borderLeft: isSelected ? '3px solid #1f6feb' : '3px solid transparent',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                fontSize: 13,
              }}
              onMouseEnter={e => (e.currentTarget.style.background = isSelected ? '#1f6feb30' : '#21262d')}
              onMouseLeave={e => (e.currentTarget.style.background = isSelected ? '#1f6feb30' : 'transparent')}
            >
              <span style={{
                width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                background: dotColor,
              }} />
              <span style={{ color: '#c9d1d9', fontWeight: 600, minWidth: 36 }}>
                #{b.bubbleNumber}
              </span>
              <span style={{
                color: match?.dimension ? '#8b949e' : '#ef444488',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
                fontSize: 12,
              }}>
                {match?.dimension ?? 'No match'}
              </span>
              <span style={{ fontSize: 10, color: dotColor, minWidth: 32, textAlign: 'right' }}>
                {(conf * 100).toFixed(0)}%
              </span>
              <SourceBadge source={match?.source ?? 'None'} small />
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div style={{ padding: '8px 16px', borderTop: '1px solid #30363d', fontSize: 11, color: '#8b949e' }}>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <LegendItem color="#22c55e" label="Success" />
          <LegendItem color="#f59e0b" label="Warning" />
          <LegendItem color="#ef4444" label="Error" />
        </div>
        <div style={{ display: 'flex', gap: 12, marginTop: 4, flexWrap: 'wrap' }}>
          <LegendItem color="#3b82f6" label="Both" dot />
          <LegendItem color="#06b6d4" label="OCR" dot />
          <LegendItem color="#a855f7" label="LLM" dot />
        </div>
      </div>
    </div>
  );
}

function ThresholdSlider({ label, value, color, onChange, min = 0 }: {
  label: string; value: number; color: string; onChange: (v: number) => void; min?: number;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
      <span style={{ color, fontSize: 11, minWidth: 80 }}>{label}</span>
      <input
        type="range"
        min={Math.round(min * 100)}
        max={100}
        value={Math.round(value * 100)}
        onChange={e => onChange(Number(e.target.value) / 100)}
        style={{ flex: 1, accentColor: color, height: 4 }}
      />
      <span style={{ color: '#c9d1d9', fontSize: 11, minWidth: 32, textAlign: 'right' }}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function ConfidenceBar({ value, thresholds }: { value: number; thresholds: { error: number; warning: number } }) {
  const color = value < thresholds.error ? '#ef4444'
    : value < thresholds.warning ? '#f59e0b' : '#22c55e';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ flex: 1, height: 6, background: '#21262d', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${value * 100}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
      <span style={{ color, fontSize: 11, fontWeight: 600 }}>{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '1px 0' }}>
      <span style={{ color: '#8b949e' }}>{label}</span>
      <span style={{ color: '#c9d1d9', fontFamily: 'monospace' }}>{value}</span>
    </div>
  );
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ background: '#0d1117', borderRadius: 6, padding: '8px 10px', textAlign: 'center' }}>
      <div style={{ fontSize: 20, fontWeight: 700, color }}>{value}</div>
      <div style={{ fontSize: 11, color: '#8b949e' }}>{label}</div>
    </div>
  );
}

function DetailRow({ label, value, color, children }: {
  label: string; value?: string; color?: string; children?: React.ReactNode;
}) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 12 }}>
      <span style={{ color: '#8b949e' }}>{label}</span>
      {children ?? <span style={{ color: color ?? '#c9d1d9' }}>{value}</span>}
    </div>
  );
}

function SourceBadge({ source, small }: { source: string; small?: boolean }) {
  const colors: Record<string, string> = {
    Both: '#3b82f6', Tesseract: '#06b6d4', LLM: '#a855f7', None: '#6b7280',
  };
  return (
    <span style={{
      background: (colors[source] ?? '#6b7280') + '30',
      color: colors[source] ?? '#6b7280',
      padding: small ? '1px 4px' : '2px 6px',
      borderRadius: 3,
      fontSize: small ? 10 : 11,
      fontWeight: 600,
    }}>
      {source}
    </span>
  );
}

function LegendItem({ color, label, dot }: { color: string; label: string; dot?: boolean }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      <span style={{
        width: dot ? 8 : 10, height: dot ? 8 : 10,
        borderRadius: dot ? '50%' : 2,
        background: color,
      }} />
      {label}
    </span>
  );
}
