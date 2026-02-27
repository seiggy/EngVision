# API Reference

EngVision exposes a REST API (FastAPI / Python) for PDF rendering, region detection, annotation management, accuracy evaluation, and pipeline orchestration.

> **See also:** [Architecture](architecture.md) · [Pipeline Orchestration](pipeline.md)

---

## PDF Management

### List available PDFs

```
GET /api/pdfs
```

Returns the filenames of all PDFs in the `sample_docs/` directory.

**Response** `200 OK`

```json
["file1.pdf", "file2.pdf"]
```

---

### Get PDF info

```
GET /api/pdfs/{filename}/info
```

Returns page count and pixel dimensions (at the configured render DPI) of the first page.

| Parameter  | In   | Type   | Description          |
|-----------|------|--------|----------------------|
| `filename` | path | string | PDF filename in `sample_docs/` |

**Response** `200 OK`

```json
{
  "pageCount": 4,
  "width": 2550,
  "height": 3300
}
```

---

### Render page as image

```
GET /api/pdfs/{filename}/pages/{pageNum}/image
```

Renders the requested page as a PNG at the configured DPI. The image is cached in `Output/<docKey>/page_{pageNum}.png`.

| Parameter  | In   | Type    | Description       |
|-----------|------|---------|-------------------|
| `filename` | path | string  | PDF filename      |
| `pageNum`  | path | integer | 1-based page number |

**Response** `200 OK` — binary `image/png`

---

## Detection

### Auto-detect regions

```
POST /api/pdfs/{filename}/pages/{pageNum}/detect
```

Runs automatic region detection on the given page. Page 1 uses bubble detection; pages 2+ use table detection (falls back to a full-page region if no tables are found). Results are stored as auto-detections in the annotation store.

| Parameter  | In   | Type    | Description       |
|-----------|------|---------|-------------------|
| `filename` | path | string  | PDF filename      |
| `pageNum`  | path | integer | 1-based page number |

**Response** `200 OK` — `DetectedRegion[]`

```json
[
  {
    "id": 1,
    "pageNumber": 1,
    "type": 0,
    "boundingBox": { "x": 120, "y": 340, "width": 80, "height": 80 },
    "bubbleNumber": 1,
    "label": null,
    "croppedImagePath": null
  },
  {
    "id": 2,
    "pageNumber": 2,
    "type": 2,
    "boundingBox": { "x": 50, "y": 100, "width": 2400, "height": 1200 },
    "bubbleNumber": null,
    "label": null,
    "croppedImagePath": null
  }
]
```

---

## Annotations

Annotations come in two flavours: **manual** (user-drawn, persisted to disk) and **auto** (detection results, in-memory only).

### Get annotations for a page

```
GET /api/pdfs/{filename}/pages/{pageNum}/annotations
```

Returns both manual and auto-detected annotations for the page.

**Response** `200 OK`

```json
{
  "manual": [
    {
      "id": "a1b2c3d4",
      "bubbleNumber": 1,
      "bubbleCenter": { "x": 160, "y": 380 },
      "boundingBox": { "x": 120, "y": 340, "width": 80, "height": 80 },
      "label": "Bubble 1",
      "notes": null
    }
  ],
  "auto": [
    {
      "id": 1,
      "pageNumber": 1,
      "type": 0,
      "boundingBox": { "x": 118, "y": 338, "width": 82, "height": 82 },
      "bubbleNumber": 1,
      "label": null,
      "croppedImagePath": null
    }
  ]
}
```

---

### Create annotation

```
POST /api/pdfs/{filename}/pages/{pageNum}/annotations
```

Creates a new manual annotation. If `id` is omitted or empty, a random 8-character hex ID is generated.

**Request body** — `Annotation` JSON (`boundingBox` is required)

```json
{
  "boundingBox": { "x": 120, "y": 340, "width": 80, "height": 80 },
  "bubbleNumber": 1,
  "label": "Bubble 1"
}
```

**Response** `200 OK` — the created `Annotation` (with `id` populated)

```json
{
  "id": "a1b2c3d4",
  "bubbleNumber": 1,
  "bubbleCenter": null,
  "boundingBox": { "x": 120, "y": 340, "width": 80, "height": 80 },
  "label": "Bubble 1",
  "notes": null
}
```

---

### Update annotation

```
PUT /api/pdfs/{filename}/pages/{pageNum}/annotations/{id}
```

Replaces an existing manual annotation with the provided data.

| Parameter | In   | Type   | Description     |
|----------|------|--------|-----------------|
| `id`      | path | string | Annotation ID   |

**Request body** — full `Annotation` JSON

**Response** `200 OK` — the updated `Annotation`

---

### Clear all manual annotations for a page

```
DELETE /api/pdfs/{filename}/pages/{pageNum}/annotations
```

Removes every manual annotation on the given page.

**Response** `204 No Content`

---

### Delete single annotation

```
DELETE /api/pdfs/{filename}/pages/{pageNum}/annotations/{id}
```

Removes the manual annotation with the given ID.

| Parameter | In   | Type   | Description     |
|----------|------|--------|-----------------|
| `id`      | path | string | Annotation ID   |

**Response** `204 No Content`

---

### Export annotations as ground truth

```
GET /api/pdfs/{filename}/annotations/export
```

Exports all manual annotations for the document as a JSON dictionary keyed by `"docKey:pageNum"`. The file is also written to `Output/<docKey>/ground_truth.json`.

**Response** `200 OK`

```json
{
  "my_drawing:1": [
    {
      "id": "a1b2c3d4",
      "boundingBox": { "x": 120, "y": 340, "width": 80, "height": 80 },
      "bubbleNumber": 1,
      "label": "Bubble 1"
    }
  ],
  "my_drawing:2": []
}
```

---

## Accuracy

### Compute detection accuracy

```
GET /api/pdfs/{filename}/pages/{pageNum}/accuracy
```

Compares auto-detected regions against manual (ground-truth) annotations using [IoU](https://en.wikipedia.org/wiki/Jaccard_index) (Intersection over Union — the ratio of overlapping area to total area of two bounding boxes) ≥ 0.3 matching. Returns precision, recall, and F1.

**Response** `200 OK` — `AccuracyResult`

```json
{
  "groundTruth": 12,
  "detected": 14,
  "matched": 11,
  "missed": 1,
  "falsePositives": 3,
  "precision": 0.786,
  "recall": 0.917,
  "f1": 0.846
}
```

If no manual annotations exist:

```json
{
  "message": "No manual annotations to compare against"
}
```

---

## Pipeline

### Upload PDF and run pipeline

```
POST /api/pipeline/run
```

Accepts a PDF via multipart form upload and runs the full extraction pipeline.

| Parameter | In       | Type | Description                  |
|----------|----------|------|------------------------------|
| `pdf`     | formData | file | PDF file (multipart field name `pdf`) |

**Response** `200 OK` — `PipelineResult` (see [Data Types](#data-types))

---

### Run pipeline on a sample PDF

```
POST /api/pipeline/run-sample/{filename}
```

Runs the pipeline on a PDF already present in `sample_docs/`.

| Parameter  | In   | Type   | Description          |
|-----------|------|--------|----------------------|
| `filename` | path | string | Filename in `sample_docs/` |

**Response** `200 OK` — `PipelineResult`

---

### Get pipeline run results

```
GET /api/pipeline/{runId}/results
```

Returns the results for a pipeline run. If the run is still in progress, returns a status object instead.

| Parameter | In   | Type   | Description |
|----------|------|--------|-------------|
| `runId`   | path | string | Run ID      |

**Response (complete)** `200 OK` — `PipelineResult`

**Response (in progress)** `200 OK`

```json
{
  "status": "running",
  "progress": "Detecting bubbles on page 1..."
}
```

**Response (not found)** `404 Not Found`

---

### Get rendered page image from a run

```
GET /api/pipeline/{runId}/pages/{pageNum}/image
```

Returns the rendered page PNG from a pipeline run's output directory.

**Response** `200 OK` — binary `image/png`

---

### Get annotated overlay image

```
GET /api/pipeline/{runId}/pages/{pageNum}/overlay
```

Returns the overlay PNG (page image with detection bounding boxes drawn) from a pipeline run.

**Response** `200 OK` — binary `image/png`

---

### List all pipeline runs

```
GET /api/pipeline/runs
```

Returns a summary list of all completed pipeline runs (most recent first).

**Response** `200 OK`

```json
[
  {
    "runId": "abc12345",
    "pdfFilename": "drawing.pdf",
    "totalBubbles": 12,
    "matchedBubbles": 10,
    "unmatchedBubbles": 2,
    "status": "complete"
  }
]
```

---

### Upload PDF and run pipeline with SSE streaming

```
POST /api/pipeline/run-stream
```

Accepts a PDF via multipart form upload and runs the pipeline, streaming progress events via **Server-Sent Events (SSE)**. Same as `/api/pipeline/run` but returns `text/event-stream` instead of JSON.

| Parameter | In       | Type | Description                  |
|----------|----------|------|------------------------------|
| `pdf`     | formData | file | PDF file (multipart field name `pdf`) |

**Response** `200 OK` — `text/event-stream`

Each SSE event has the format `event: <type>\ndata: <json>\n\n`.

Event types: `step`, `stepComplete`, `bubble`, `complete`, `error`.

```
event: step
data: {"step":1,"totalSteps":7,"name":"render","message":"Rendering PDF pages..."}

event: bubble
data: {"bubbleNumber":1,"captureSize":"128x128","status":"match","tableDim":".81","observed":"0.81","confidence":0.95}

event: complete
data: {"runId":"abc123","pdfFilename":"drawing.pdf",...}
```

---

### Run pipeline with SSE streaming on a sample PDF

```
POST /api/pipeline/run-stream-sample/{filename}
```

Streams pipeline progress for a PDF already present in `sample_docs/`.

| Parameter  | In   | Type   | Description          |
|-----------|------|--------|----------------------|
| `filename` | path | string | Filename in `sample_docs/` |

**Response** `200 OK` — `text/event-stream` (same event format as above)

---

## Data Types

### BoundingBox

Pixel-coordinate rectangle.

| Field    | Type | Description            |
|---------|------|------------------------|
| `x`      | int  | Left edge (px)         |
| `y`      | int  | Top edge (px)          |
| `width`  | int  | Width (px)             |
| `height` | int  | Height (px)            |

---

### RegionType (enum)

| Value | Name               | Description                                |
|-------|--------------------|--------------------------------------------|
| `0`   | Bubble             | Circular balloon callout on page 1         |
| `1`   | BubbleWithFigure   | Bubble that includes an adjacent figure     |
| `2`   | TableRegion        | Tabular data region on pages 2+            |
| `3`   | FullPage           | Entire page (fallback when no tables found) |

---

### DetectedRegion

Result of automatic region detection.

| Field             | Type        | Description                         |
|------------------|-------------|-------------------------------------|
| `id`              | int         | Sequential region ID                |
| `pageNumber`      | int         | 1-based page number                 |
| `type`            | RegionType  | Region classification               |
| `boundingBox`     | BoundingBox | Pixel bounding box                  |
| `bubbleNumber`    | int \| null | Assigned bubble number (page 1 only)|
| `label`           | string \| null | Optional label                   |
| `croppedImagePath`| string \| null | Path to cropped image on disk    |

---

### Annotation

User-created (manual) annotation.

| Field          | Type           | Description                        |
|---------------|----------------|------------------------------------|
| `id`           | string         | Unique ID (auto-generated if omitted) |
| `bubbleNumber` | int \| null    | Associated bubble number           |
| `bubbleCenter` | Point \| null  | Center point `{x, y}`             |
| `boundingBox`  | BoundingBox    | **Required.** Pixel bounding box   |
| `label`        | string \| null | Display label                      |
| `notes`        | string \| null | Free-form notes                    |

---

### AccuracyResult

IoU-based accuracy metrics comparing manual annotations to auto-detections.

| Field            | Type  | Description                              |
|-----------------|-------|------------------------------------------|
| `groundTruth`    | int   | Number of manual (ground truth) annotations |
| `detected`       | int   | Number of auto-detected regions          |
| `matched`        | int   | Detections matched to ground truth (IoU ≥ 0.3) |
| `missed`         | int   | Ground truth with no matching detection  |
| `falsePositives` | int   | Detections with no matching ground truth |
| `precision`      | float | `matched / detected`                     |
| `recall`         | float | `matched / groundTruth`                  |
| `f1`             | float | Harmonic mean of precision and recall    |

---

### PipelineResult

Full output of a pipeline run.

| Field             | Type                          | Description                         |
|------------------|-------------------------------|-------------------------------------|
| `runId`           | string                        | Unique run identifier               |
| `pdfFilename`     | string                        | Source PDF filename                  |
| `pageCount`       | int                           | Number of pages in the PDF          |
| `imageWidth`      | int                           | Rendered image width (px)           |
| `imageHeight`     | int                           | Rendered image height (px)          |
| `bubbles`         | BubbleResult[]                | Detected bubbles on page 1          |
| `dimensionMap`    | Record\<string, DimensionMatch\> | Bubble-to-dimension mapping (keyed by bubble number) |
| `totalBubbles`    | int                           | Total bubbles detected              |
| `matchedBubbles`  | int                           | Bubbles matched to table dimensions |
| `unmatchedBubbles`| int                           | Bubbles without a dimension match   |
| `warnings`        | int                           | Number of processing warnings       |
| `metrics`         | ProcessingMetrics \| null     | Timing and memory stats             |
| `tokenUsage`      | LlmTokenUsage \| null        | LLM token consumption               |
| `status`          | string                        | `"complete"` or `"error"`           |
| `error`           | string \| null                | Error message if `status = "error"` |

---

### BubbleResult

A single detected bubble on page 1.

| Field          | Type        | Description                |
|---------------|-------------|----------------------------|
| `bubbleNumber` | int         | Assigned bubble number     |
| `cx`           | int         | Center X coordinate (px)  |
| `cy`           | int         | Center Y coordinate (px)  |
| `radius`       | int         | Approximate radius (px)   |
| `boundingBox`  | BoundingBox | Enclosing bounding box    |

---

### DimensionMatch

Maps a balloon number to its extracted dimension value.

| Field              | Type           | Description                                  |
|-------------------|----------------|----------------------------------------------|
| `balloonNo`        | int            | Balloon / bubble number                      |
| `dimension`        | string \| null | Dimension value (from table, or LLM-discovered) |
| `source`           | string         | `"Table+Validated"`, `"TableOnly"`, `"LLMOnly"`, or `"None"` |
| `tesseractValue`   | string \| null | Value from Tesseract table OCR               |
| `llmObservedValue` | string \| null | What the LLM saw on the drawing crop         |
| `llmMatches`       | bool \| null   | Whether LLM says drawing matches table       |
| `llmConfidence`    | float          | LLM's self-reported confidence (0.0–1.0)     |
| `llmNotes`         | string \| null | LLM explanation                              |
| `hasConflict`      | bool           | `true` when LLM says drawing doesn't match table |
| `confidence`       | float          | LLM confidence (on match) or fuzzy match score (on conflict), 0.0–1.0 |
| `captureSize`      | string \| null | Final capture box size (e.g. `"128x128"`, `"256x128"`) |

---

### ProcessingMetrics

Timing breakdown for a pipeline run.

| Field                | Type  | Description                        |
|---------------------|-------|------------------------------------|
| `totalDurationMs`    | int   | Total wall-clock time (ms)         |
| `renderDurationMs`   | int   | PDF rendering time (ms)            |
| `detectDurationMs`   | int   | Region detection time (ms)         |
| `ocrDurationMs`      | int   | Tesseract OCR time (ms)            |
| `traceDurationMs`    | int   | Leader line tracing time (ms)      |
| `llmDurationMs`      | int   | LLM inference time (ms)            |
| `mergeDurationMs`    | int   | Result merging time (ms)           |
| `overlayDurationMs`  | int   | Overlay image generation time (ms) |
| `peakMemoryMb`       | float | Peak memory usage (MB)             |

---

### LlmTokenUsage

Token consumption across all LLM calls in a run.

| Field          | Type | Description              |
|---------------|------|--------------------------|
| `inputTokens`  | int  | Prompt tokens consumed   |
| `outputTokens` | int  | Completion tokens generated |
| `totalTokens`  | int  | `inputTokens + outputTokens` |
| `llmCalls`     | int  | Number of LLM API calls  |

---

## Annotation Store

The `AnnotationStore` manages two categories of annotations:

- **Manual annotations** — user-created via the UI. Persisted to `Output/annotations/manual_annotations.json` as a JSON dictionary keyed by `"docKey:pageNum"`. Loaded on startup; saved after every mutation.
- **Auto-detections** — produced by the `/detect` endpoint. Stored in-memory only and lost on server restart.

Pipeline runs are also held in-memory and are not persisted across restarts.

The `docKey` is derived from the PDF filename by stripping the `.pdf` extension and replacing spaces and dots with underscores (e.g. `My Drawing.pdf` → `My_Drawing`).

---

> **See also:** [Architecture](architecture.md) · [Pipeline Orchestration](pipeline.md)
