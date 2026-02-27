import { useState, useCallback, useRef } from 'react';
import type {
  PipelineResult,
  PipelineSSEEvent,
  PipelineStep,
  BubbleStatus,
} from '../types';

const BASE = '/api';

const STEP_NAMES: Record<number, string> = {
  1: 'Render pages',
  2: 'Detect bubbles',
  3: 'OCR table data',
  4: 'Trace leader lines',
  5: 'Validate dimensions',
  6: 'Merge results',
  7: 'Generate overlay',
};

export interface PipelineStreamState {
  /** Pipeline is actively running */
  running: boolean;
  /** Ordered step progress */
  steps: PipelineStep[];
  /** Per-bubble validation status */
  bubbleStatuses: Map<number, BubbleStatus>;
  /** The bubble currently being validated */
  currentBubble: number | null;
  /** Total elapsed time in ms (updated periodically) */
  elapsedMs: number;
  /** Final result once complete */
  result: PipelineResult | null;
  /** Error message if failed */
  error: string | null;
  /** Number of LLM calls made so far */
  llmCalls: number;
}

export function usePipelineStream() {
  const [state, setState] = useState<PipelineStreamState>({
    running: false,
    steps: [],
    bubbleStatuses: new Map(),
    currentBubble: null,
    elapsedMs: 0,
    result: null,
    error: null,
    llmCalls: 0,
  });

  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  const runStream = useCallback(
    async (mode: 'upload' | 'sample', fileOrName: File | string) => {
      // Reset state
      setState({
        running: true,
        steps: Array.from({ length: 7 }, (_, i) => ({
          step: i + 1,
          name: STEP_NAMES[i + 1],
          message: '',
          status: 'pending' as const,
        })),
        bubbleStatuses: new Map(),
        currentBubble: null,
        elapsedMs: 0,
        result: null,
        error: null,
        llmCalls: 0,
      });

      const controller = new AbortController();
      abortRef.current = controller;

      // Elapsed timer
      startTimeRef.current = Date.now();
      timerRef.current = setInterval(() => {
        setState(prev => ({
          ...prev,
          elapsedMs: Date.now() - startTimeRef.current,
        }));
      }, 250);

      try {
        let response: Response;

        if (mode === 'upload') {
          const formData = new FormData();
          formData.append('pdf', fileOrName as File);
          response = await fetch(`${BASE}/pipeline/run-stream`, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
          });
        } else {
          response = await fetch(
            `${BASE}/pipeline/run-stream-sample/${encodeURIComponent(fileOrName as string)}`,
            { method: 'POST', signal: controller.signal }
          );
        }

        if (!response.ok) {
          throw new Error(await response.text());
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error('No response body');

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          let currentEventType = '';
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              currentEventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              const data = line.slice(6);
              try {
                const event = JSON.parse(data) as PipelineSSEEvent;
                processEvent(event, setState);
              } catch {
                // skip malformed JSON
              }
              currentEventType = '';
            }
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        const msg = err instanceof Error ? err.message : 'Unknown error';
        setState(prev => ({ ...prev, error: msg, running: false }));
      } finally {
        if (timerRef.current) clearInterval(timerRef.current);
        setState(prev => ({
          ...prev,
          running: prev.result ? false : prev.running,
          elapsedMs: Date.now() - startTimeRef.current,
        }));
      }
    },
    []
  );

  return { ...state, runStream, stop };
}

function processEvent(
  event: PipelineSSEEvent,
  setState: React.Dispatch<React.SetStateAction<PipelineStreamState>>
) {
  switch (event.type) {
    case 'step':
      setState(prev => ({
        ...prev,
        steps: prev.steps.map(s =>
          s.step === event.step
            ? { ...s, message: event.message, status: 'running' as const }
            : s
        ),
      }));
      break;

    case 'stepComplete':
      setState(prev => ({
        ...prev,
        steps: prev.steps.map(s =>
          s.step === event.step
            ? {
                ...s,
                status: 'complete' as const,
                durationMs: event.durationMs,
                detail: event.detail,
              }
            : s
        ),
      }));
      break;

    case 'bubble':
      setState(prev => {
        const next = new Map(prev.bubbleStatuses);
        next.set(event.bubbleNumber, {
          bubbleNumber: event.bubbleNumber,
          captureSize: event.captureSize,
          status: event.status === 'match' ? 'match'
            : event.status === 'expanding' ? 'expanding'
            : event.status === 'bestGuess' ? 'bestGuess'
            : event.status === 'discovered' ? 'discovered'
            : 'noMatch',
          tableDim: event.tableDim,
          observed: event.observed,
          confidence: event.confidence,
        });
        return {
          ...prev,
          bubbleStatuses: next,
          currentBubble: event.bubbleNumber,
          llmCalls: prev.llmCalls + 1,
        };
      });
      break;

    case 'complete':
      setState(prev => ({
        ...prev,
        result: event.result,
        running: false,
      }));
      break;

    case 'error':
      setState(prev => ({
        ...prev,
        error: event.message,
        running: false,
      }));
      break;
  }
}
