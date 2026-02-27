import { useMemo } from 'react';
import type { PipelineStep, BubbleStatus } from '../types';

interface Props {
  steps: PipelineStep[];
  bubbleStatuses: Map<number, BubbleStatus>;
  currentBubble: number | null;
  elapsedMs: number;
  llmCalls: number;
  totalBubbles: number;
  error: string | null;
}

export default function PipelineProgressView({
  steps,
  bubbleStatuses,
  currentBubble,
  elapsedMs,
  llmCalls,
  totalBubbles,
  error,
}: Props) {
  const bubbleArray = useMemo(() => {
    if (totalBubbles === 0) return [];
    return Array.from({ length: totalBubbles }, (_, i) => {
      const num = i + 1;
      return bubbleStatuses.get(num) ?? {
        bubbleNumber: num,
        captureSize: '',
        status: 'pending' as const,
      };
    });
  }, [totalBubbles, bubbleStatuses]);

  const matchCount = useMemo(
    () => Array.from(bubbleStatuses.values()).filter(b => b.status === 'match').length,
    [bubbleStatuses]
  );

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', height: '100%', padding: 32, gap: 24,
    }}>
      {/* Header */}
      <div style={{ textAlign: 'center' }}>
        <h3 style={{ margin: 0, color: '#58a6ff', fontSize: 18 }}>Pipeline Running</h3>
        <p style={{ margin: '4px 0 0', color: '#8b949e', fontSize: 13 }}>
          Elapsed: {formatMs(elapsedMs)}
          {llmCalls > 0 && <> · LLM calls: {llmCalls}</>}
          {matchCount > 0 && <> · Matched: {matchCount}/{totalBubbles}</>}
        </p>
      </div>

      {/* Steps */}
      <div style={{
        width: '100%', maxWidth: 520, display: 'flex', flexDirection: 'column', gap: 6,
      }}>
        {steps.map(s => (
          <StepRow key={s.step} step={s} />
        ))}
      </div>

      {/* Bubble grid */}
      {totalBubbles > 0 && (
        <div style={{ width: '100%', maxWidth: 520 }}>
          <p style={{ margin: '0 0 8px', fontSize: 12, color: '#8b949e' }}>
            Bubble validation ({bubbleStatuses.size}/{totalBubbles})
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
            {bubbleArray.map(b => (
              <BubbleCell
                key={b.bubbleNumber}
                bubble={b}
                isCurrent={b.bubbleNumber === currentBubble}
              />
            ))}
          </div>
          <div style={{ display: 'flex', gap: 12, marginTop: 8, fontSize: 11, color: '#8b949e' }}>
            <Legend color={COLORS.match} label="Match" />
            <Legend color={COLORS.discovered} label="Discovered" />
            <Legend color={COLORS.bestGuess} label="Best guess" />
            <Legend color={COLORS.noMatch} label="No match" />
            <Legend color={COLORS.expanding} label="Expanding" />
            <Legend color={COLORS.checking} label="Checking" />
            <Legend color={COLORS.pending} label="Pending" />
          </div>
        </div>
      )}

      {/* Current bubble detail */}
      {currentBubble && bubbleStatuses.has(currentBubble) && (
        <CurrentBubbleDetail bubble={bubbleStatuses.get(currentBubble)!} />
      )}

      {/* Error */}
      {error && (
        <div style={{ color: '#ef4444', textAlign: 'center', fontSize: 14 }}>
          ⚠ {error}
        </div>
      )}
    </div>
  );
}

// ── Step row ─────────────────────────────────────────────────────────────────────

function StepRow({ step }: { step: PipelineStep }) {
  const icon = step.status === 'complete' ? '✓'
    : step.status === 'running' ? '◉'
    : '○';
  const iconColor = step.status === 'complete' ? '#3fb950'
    : step.status === 'running' ? '#f59e0b'
    : '#484f58';

  const detailText = step.detail
    ? Object.entries(step.detail).map(([k, v]) => `${k}: ${v}`).join(', ')
    : '';

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10,
      padding: '5px 8px', borderRadius: 6,
      background: step.status === 'running' ? '#1c2128' : 'transparent',
    }}>
      <span style={{ color: iconColor, fontSize: 14, fontWeight: 700, width: 16, textAlign: 'center' }}>
        {step.status === 'running' ? <Spinner /> : icon}
      </span>
      <span style={{
        flex: 1, fontSize: 13,
        color: step.status === 'pending' ? '#484f58' : '#c9d1d9',
      }}>
        {step.name}
      </span>
      {step.durationMs != null && (
        <span style={{ fontSize: 12, color: '#8b949e', minWidth: 50, textAlign: 'right' }}>
          {formatMs(step.durationMs)}
        </span>
      )}
      {detailText && (
        <span style={{ fontSize: 11, color: '#6e7681' }}>
          ({detailText})
        </span>
      )}
    </div>
  );
}

// ── Bubble grid cell ─────────────────────────────────────────────────────────────

const COLORS = {
  pending: '#21262d',
  checking: '#58a6ff',
  match: '#3fb950',
  expanding: '#f59e0b',
  bestGuess: '#d29922',
  noMatch: '#f85149',
  discovered: '#a371f7',
};

function BubbleCell({ bubble, isCurrent }: { bubble: BubbleStatus; isCurrent: boolean }) {
  const bg = COLORS[bubble.status] || COLORS.pending;
  return (
    <div
      title={`#${bubble.bubbleNumber} ${bubble.status}${bubble.observed ? ` → ${bubble.observed}` : ''}`}
      style={{
        width: 18, height: 18, borderRadius: 3,
        background: bg,
        border: isCurrent ? '2px solid #fff' : '1px solid #30363d',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 8, color: '#fff', fontWeight: 600,
        animation: isCurrent ? 'pulse 1s infinite' : undefined,
      }}
    >
      {bubble.bubbleNumber}
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
      <span style={{
        width: 10, height: 10, borderRadius: 2, background: color,
        display: 'inline-block', border: '1px solid #30363d',
      }} />
      {label}
    </span>
  );
}

// ── Current bubble detail ────────────────────────────────────────────────────────

function CurrentBubbleDetail({ bubble }: { bubble: BubbleStatus }) {
  return (
    <div style={{
      background: '#161b22', border: '1px solid #30363d', borderRadius: 8,
      padding: '10px 16px', width: '100%', maxWidth: 520, fontSize: 13,
    }}>
      <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
        <span style={{ color: '#58a6ff', fontWeight: 700 }}>
          Bubble #{bubble.bubbleNumber}
        </span>
        <span style={{ color: '#8b949e' }}>
          Capture: {bubble.captureSize || '—'}
        </span>
        <span style={{
          marginLeft: 'auto',
          color: bubble.status === 'match' ? '#3fb950'
            : bubble.status === 'noMatch' ? '#f85149'
            : '#f59e0b',
          fontWeight: 600,
        }}>
          {bubble.status}
        </span>
      </div>
      {(bubble.tableDim || bubble.observed) && (
        <div style={{ marginTop: 6, display: 'flex', gap: 16, color: '#c9d1d9' }}>
          {bubble.tableDim && <span>Table: <code>{bubble.tableDim}</code></span>}
          {bubble.observed && <span>Observed: <code>{bubble.observed}</code></span>}
          {bubble.confidence != null && (
            <span style={{ color: '#8b949e' }}>
              Confidence: {(bubble.confidence * 100).toFixed(0)}%
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────────

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function Spinner() {
  return (
    <svg width={14} height={14} viewBox="0 0 24 24" style={{ animation: 'spin 1s linear infinite' }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
      `}</style>
      <circle cx="12" cy="12" r="10" fill="none" stroke="#f59e0b" strokeWidth="3"
        strokeDasharray="31.4 31.4" strokeLinecap="round" />
    </svg>
  );
}
