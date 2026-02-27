export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface DetectedRegion {
  id: number;
  pageNumber: number;
  type: number; // 0=Bubble, 1=BubbleWithFigure, 2=TableRegion, 3=FullPage
  boundingBox: BoundingBox;
  bubbleNumber?: number;
  label?: string;
  croppedImagePath?: string;
}

export interface Annotation {
  id: string;
  bubbleNumber?: number;
  bubbleCenter?: Point;
  boundingBox: BoundingBox;
  label?: string;
  notes?: string;
}

export interface PdfInfo {
  pageCount: number;
  width: number;
  height: number;
}

export interface AccuracyResult {
  groundTruth: number;
  detected: number;
  matched: number;
  missed: number;
  falsePositives: number;
  precision: number;
  recall: number;
  f1: number;
}

// Pipeline types
export interface BubbleResult {
  bubbleNumber: number;
  cx: number;
  cy: number;
  radius: number;
  boundingBox: BoundingBox;
}

export interface DimensionMatch {
  balloonNo: number;
  dimension: string | null;
  source: string; // "Table+Validated" | "TableOnly" | "None"
  tesseractValue: string | null;
  llmObservedValue: string | null;
  llmMatches: boolean | null;
  llmConfidence: number;
  llmNotes: string | null;
  hasConflict: boolean;
  confidence: number; // 0-1, 1.0 = exact match
  captureSize: string | null; // e.g. "128x128", "256x128"
}

export interface ProcessingMetrics {
  totalDurationMs: number;
  renderDurationMs: number;
  detectDurationMs: number;
  traceDurationMs: number;
  ocrDurationMs: number;
  llmDurationMs: number;
  mergeDurationMs: number;
  peakMemoryMb: number;
}

export interface LlmTokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  llmCalls: number;
}

export interface PipelineResult {
  runId: string;
  pdfFilename: string;
  pageCount: number;
  imageWidth: number;
  imageHeight: number;
  bubbles: BubbleResult[];
  dimensionMap: Record<string, DimensionMatch>;
  totalBubbles: number;
  matchedBubbles: number;
  unmatchedBubbles: number;
  warnings: number;
  metrics?: ProcessingMetrics;
  tokenUsage?: LlmTokenUsage;
  status: string;
  error?: string;
}

// ── SSE Pipeline Event Types ────────────────────────────────────────────────────

export interface StepEvent {
  type: 'step';
  step: number;
  totalSteps: number;
  name: string;
  message: string;
}

export interface StepCompleteEvent {
  type: 'stepComplete';
  step: number;
  name: string;
  durationMs: number;
  detail?: Record<string, unknown>;
}

export interface BubbleEvent {
  type: 'bubble';
  bubbleNumber: number;
  captureSize: string;
  status: 'match' | 'expanding' | 'bestGuess' | 'noMatch' | 'discovered';
  tableDim: string;
  observed: string;
  confidence: number;
}

export interface CompleteEvent {
  type: 'complete';
  result: PipelineResult;
}

export interface ErrorEvent {
  type: 'error';
  message: string;
}

export type PipelineSSEEvent =
  | StepEvent
  | StepCompleteEvent
  | BubbleEvent
  | CompleteEvent
  | ErrorEvent;

export interface PipelineStep {
  step: number;
  name: string;
  message: string;
  status: 'pending' | 'running' | 'complete';
  durationMs?: number;
  detail?: Record<string, unknown>;
}

export interface BubbleStatus {
  bubbleNumber: number;
  captureSize: string;
  status: 'pending' | 'checking' | 'match' | 'expanding' | 'bestGuess' | 'noMatch' | 'discovered';
  tableDim?: string;
  observed?: string;
  confidence?: number;
}
