import type { Annotation, DetectedRegion, PdfInfo, AccuracyResult, BoundingBox } from './types';

const BASE = '/api';

export async function listPdfs(): Promise<string[]> {
  const res = await fetch(`${BASE}/pdfs`);
  return res.json();
}

export async function getPdfInfo(filename: string): Promise<PdfInfo> {
  const res = await fetch(`${BASE}/pdfs/${encodeURIComponent(filename)}/info`);
  return res.json();
}

export function getPageImageUrl(filename: string, pageNum: number): string {
  return `${BASE}/pdfs/${encodeURIComponent(filename)}/pages/${pageNum}/image`;
}

export async function detectRegions(filename: string, pageNum: number): Promise<DetectedRegion[]> {
  const res = await fetch(`${BASE}/pdfs/${encodeURIComponent(filename)}/pages/${pageNum}/detect`, {
    method: 'POST',
  });
  return res.json();
}

export async function getAnnotations(filename: string, pageNum: number): Promise<{
  manual: Annotation[];
  auto: DetectedRegion[];
}> {
  const res = await fetch(`${BASE}/pdfs/${encodeURIComponent(filename)}/pages/${pageNum}/annotations`);
  return res.json();
}

export async function saveAnnotation(filename: string, pageNum: number, annotation: {
  bubbleNumber?: number;
  bubbleCenter?: { x: number; y: number };
  boundingBox: BoundingBox;
  label?: string;
  notes?: string;
}): Promise<Annotation> {
  const res = await fetch(`${BASE}/pdfs/${encodeURIComponent(filename)}/pages/${pageNum}/annotations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(annotation),
  });
  return res.json();
}

export async function updateAnnotation(
  filename: string, pageNum: number, id: string, annotation: Annotation
): Promise<Annotation> {
  const res = await fetch(
    `${BASE}/pdfs/${encodeURIComponent(filename)}/pages/${pageNum}/annotations/${id}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(annotation),
    }
  );
  return res.json();
}

export async function clearAnnotations(filename: string, pageNum: number): Promise<void> {
  await fetch(`${BASE}/pdfs/${encodeURIComponent(filename)}/pages/${pageNum}/annotations`, {
    method: 'DELETE',
  });
}

export async function deleteAnnotation(filename: string, pageNum: number, id: string): Promise<void> {
  await fetch(`${BASE}/pdfs/${encodeURIComponent(filename)}/pages/${pageNum}/annotations/${id}`, {
    method: 'DELETE',
  });
}

export async function exportAnnotations(filename: string): Promise<Record<string, Annotation[]>> {
  const res = await fetch(`${BASE}/pdfs/${encodeURIComponent(filename)}/annotations/export`);
  return res.json();
}

export async function getAccuracy(filename: string, pageNum: number): Promise<AccuracyResult> {
  const res = await fetch(`${BASE}/pdfs/${encodeURIComponent(filename)}/pages/${pageNum}/accuracy`);
  return res.json();
}

// ── Pipeline API ──────────────────────────────────────────────────────────────

import type { PipelineResult } from './types';

export async function runPipeline(file: File): Promise<PipelineResult> {
  const formData = new FormData();
  formData.append('pdf', file);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300_000); // 5 min timeout
  try {
    const res = await fetch(`${BASE}/pipeline/run`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError')
      throw new Error('Pipeline timed out after 5 minutes. The server may still be processing.');
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export async function runPipelineSample(filename: string): Promise<PipelineResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300_000); // 5 min timeout
  try {
    const res = await fetch(`${BASE}/pipeline/run-sample/${encodeURIComponent(filename)}`, {
      method: 'POST',
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError')
      throw new Error('Pipeline timed out after 5 minutes. The server may still be processing.');
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export async function getPipelineResults(runId: string): Promise<PipelineResult> {
  const res = await fetch(`${BASE}/pipeline/${runId}/results`);
  return res.json();
}

export function getPipelinePageImageUrl(runId: string, pageNum: number): string {
  return `${BASE}/pipeline/${runId}/pages/${pageNum}/image`;
}

export function getPipelineOverlayUrl(runId: string, pageNum: number): string {
  return `${BASE}/pipeline/${runId}/pages/${pageNum}/overlay`;
}
