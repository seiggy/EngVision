import type { Annotation, AccuracyResult } from '../types';

interface Props {
  annotations: Annotation[];
  selectedId: string | null;
  accuracy: AccuracyResult | null;
  onSelect: (id: string | null) => void;
  onDelete: (id: string) => void;
  onUpdate: (id: string, updates: Partial<Annotation>) => void;
}

export default function AnnotationPanel({
  annotations, selectedId, accuracy, onSelect, onDelete, onUpdate,
}: Props) {
  const selected = annotations.find(a => a.id === selectedId);

  return (
    <div style={{ width: 320, borderLeft: '1px solid #333', background: '#1a1a2e', color: '#eee', overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '12px 16px', borderBottom: '1px solid #333' }}>
        <h3 style={{ margin: 0, fontSize: 14, color: '#22c55e' }}>
          Manual Annotations ({annotations.length})
        </h3>
      </div>

      {/* Selected annotation editor */}
      {selected && (
        <div style={{ padding: 16, borderBottom: '1px solid #333', background: '#16213e' }}>
          <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>
            ID: {selected.id}
          </div>
          <label style={{ fontSize: 12, color: '#aaa', display: 'block', marginBottom: 4 }}>
            Bubble Number
          </label>
          <input
            type="number"
            value={selected.bubbleNumber ?? ''}
            onChange={e => onUpdate(selected.id, {
              bubbleNumber: e.target.value ? parseInt(e.target.value) : undefined,
            })}
            style={inputStyle}
            placeholder="e.g. 1, 2, 3..."
          />
          <label style={{ fontSize: 12, color: '#aaa', display: 'block', marginBottom: 4, marginTop: 8 }}>
            Label
          </label>
          <input
            type="text"
            value={selected.label ?? ''}
            onChange={e => onUpdate(selected.id, { label: e.target.value || undefined })}
            style={inputStyle}
            placeholder="Optional label"
          />
          <label style={{ fontSize: 12, color: '#aaa', display: 'block', marginBottom: 4, marginTop: 8 }}>
            Notes
          </label>
          <textarea
            value={selected.notes ?? ''}
            onChange={e => onUpdate(selected.id, { notes: e.target.value || undefined })}
            style={{ ...inputStyle, height: 60, resize: 'vertical' }}
            placeholder="Optional notes"
          />
          <div style={{ fontSize: 11, color: '#666', marginTop: 8 }}>
            Box: ({selected.boundingBox.x}, {selected.boundingBox.y}) {selected.boundingBox.width}×{selected.boundingBox.height}
          </div>
          {selected.bubbleCenter && (
            <div style={{ fontSize: 11, color: '#ef4444', marginTop: 2 }}>
              🎯 Bubble center: ({selected.bubbleCenter.x}, {selected.bubbleCenter.y})
            </div>
          )}
          <button onClick={() => onDelete(selected.id)} style={deleteButtonStyle}>
            Delete Annotation
          </button>
        </div>
      )}

      {/* Annotation list */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {annotations.map(a => (
          <div
            key={a.id}
            onClick={() => onSelect(a.id)}
            style={{
              padding: '8px 16px',
              cursor: 'pointer',
              borderBottom: '1px solid #222',
              background: a.id === selectedId ? '#1e3a5f' : 'transparent',
            }}
          >
            <div style={{ fontSize: 13, fontWeight: a.bubbleNumber ? 600 : 400 }}>
              {a.bubbleNumber ? `Bubble #${a.bubbleNumber}` : (a.label || `Annotation ${a.id.slice(0, 6)}`)}
            </div>
            <div style={{ fontSize: 11, color: '#666' }}>
              ({a.boundingBox.x}, {a.boundingBox.y}) {a.boundingBox.width}×{a.boundingBox.height}
            </div>
          </div>
        ))}
        {annotations.length === 0 && (
          <div style={{ padding: 16, color: '#666', fontSize: 13, textAlign: 'center' }}>
            Draw bounding boxes on the image to create annotations.
          </div>
        )}
      </div>

      {/* Accuracy metrics */}
      {accuracy && (
        <div style={{ padding: 16, borderTop: '1px solid #333', background: '#0f3460' }}>
          <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#f59e0b' }}>Detection Accuracy</h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 12 }}>
            <span style={{ color: '#aaa' }}>Ground Truth:</span><span>{accuracy.groundTruth}</span>
            <span style={{ color: '#aaa' }}>Detected:</span><span>{accuracy.detected}</span>
            <span style={{ color: '#aaa' }}>Matched:</span><span style={{ color: '#22c55e' }}>{accuracy.matched}</span>
            <span style={{ color: '#aaa' }}>Missed:</span><span style={{ color: '#ef4444' }}>{accuracy.missed}</span>
            <span style={{ color: '#aaa' }}>False Pos:</span><span style={{ color: '#ef4444' }}>{accuracy.falsePositives}</span>
            <span style={{ color: '#aaa' }}>Precision:</span><span>{(accuracy.precision * 100).toFixed(1)}%</span>
            <span style={{ color: '#aaa' }}>Recall:</span><span>{(accuracy.recall * 100).toFixed(1)}%</span>
            <span style={{ color: '#aaa' }}>F1:</span><span style={{ fontWeight: 600, color: '#f59e0b' }}>{(accuracy.f1 * 100).toFixed(1)}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '6px 8px',
  background: '#0d1b2a',
  border: '1px solid #444',
  borderRadius: 4,
  color: '#eee',
  fontSize: 13,
  boxSizing: 'border-box',
};

const deleteButtonStyle: React.CSSProperties = {
  marginTop: 12,
  width: '100%',
  padding: '6px 0',
  background: '#7f1d1d',
  border: '1px solid #991b1b',
  borderRadius: 4,
  color: '#fca5a5',
  fontSize: 12,
  cursor: 'pointer',
};
