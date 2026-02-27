# Analysis Pipeline

![Pipeline Flow](diagrams/pipeline-flow.excalidraw.json)

## Overview

The analysis pipeline is the end-to-end orchestration layer that takes an engineering-drawing PDF and produces a structured analysis result. It renders pages, detects inspection bubbles, reads dimension tables via OCR, traces leader lines from each bubble to locate drawing regions, validates those regions against table dimensions using a Vision LLM, merges the results with confidence scoring, and generates an annotated overlay image.

The pipeline is exposed through two HTTP endpoints:

| Endpoint | Description |
|---|---|
| `POST /api/pipeline/run` | Upload a PDF for analysis |
| `POST /api/pipeline/run-sample/{filename}` | Run against a pre-loaded sample PDF |

The entry point in Python is `PipelineService.run_async()` in [`services/pipeline.py`](../engvision-py/src/engvision/services/pipeline.py).

---

## Pipeline Steps

Each step is individually timed and reported in [`ProcessingMetrics`](#processingmetrics).

### Step 1 — PDF Rendering

Renders every page of the PDF to a NumPy image array at 300 DPI and writes PNGs to the `pages/` output directory.

```python
renderer = PdfRendererService(self._config.pdf_render_dpi)
page_images = renderer.render_all_pages(pdf_path)

for i, img in enumerate(page_images):
    PdfRendererService.save_image(img, os.path.join(pages_dir, f"page_{i + 1}.png"))
```

**Metric:** `renderDurationMs`

### Step 2a — Bubble Detection

Runs circle-detection on **page 1 only** to locate inspection bubbles.

```python
bubble_detector = BubbleDetectionService(self._config)
bubbles = bubble_detector.detect_bubbles(page_images[0], page_number=1)
```

**Metric:** `detectDurationMs`
→ See [Bubble Detection](bubble-detection.md) for the Hough-transform algorithm and tuning parameters.

### Step 2b — Bubble Crops

For each detected bubble, a tight crop is extracted from page 1 using center ± radius + 2 px padding and saved to the `bubble_crops/` directory.

```python
pad = 2
x1 = max(0, cx - r - pad)
y1 = max(0, cy - r - pad)
x2 = min(img_w, cx + r + pad)
y2 = min(img_h, cy + r + pad)
crop = page_images[0][y1:y2, x1:x2]
cv2.imwrite(os.path.join(raw_crops_dir, f"bubble_{b['bubbleNumber']:03d}.png"), crop)
```

### Step 2c — Bubble Number OCR

Reads the numeric label inside each bubble crop using Tesseract.

```python
ocr_service = BubbleOcrService(self._tess_data_path)
ocr_results = ocr_service.extract_all(raw_crops_dir)
```

→ See [Table & OCR](table-ocr.md) for OCR configuration.

### Step 3 — Table OCR

Extracts a balloon-number → dimension mapping from tabular data on **pages 2+**. The OCR provider is selected by the `OCR_PROVIDER` environment variable:

- **Tesseract** (default): Morphological grid detection followed by per-cell Tesseract OCR, page by page.
- **Azure Document Intelligence**: Sends the full multi-page PDF in a single API call using the `prebuilt-layout` model.

```python
ocr_service, table_ocr = self._create_ocr_services()  # respects OCR_PROVIDER

# Azure Doc Intelligence: full-PDF mode (single call)
if hasattr(table_ocr, "extract_balloon_dimensions_from_pdf"):
    tesseract_dimensions = table_ocr.extract_balloon_dimensions_from_pdf(pdf_bytes)
else:
    # Tesseract: page-by-page
    for i in range(1, len(page_images)):
        page_dims = table_ocr.extract_balloon_dimensions(page_images[i])
```

The first occurrence of each balloon number wins (via `setdefault`).

**Metric:** `ocrDurationMs` (includes Steps 2c + 3)
→ See [Table & OCR](table-ocr.md).

### Step 4 — Leader Line Tracing & Capture Box Placement

Traces the triangle pointer direction from each bubble on page 1 and places progressive capture boxes along the ray. The `LeaderLineTracerService` uses [flood-fill](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga366aae45a6c1289b341d140c2c547f2f) (a paint-bucket-like spread that isolates connected pixels) + [Harris corner detection](https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html) (detects sharp corners by measuring how pixel intensity changes in all directions) to handle overlapping/adjacent bubbles.

```python
tracer = LeaderLineTracerService()
expanded_bubbles = tracer.trace_and_expand(bubbles, page_images[0])
```

Each expanded bubble receives:
- `leaderDirection` — a unit vector `{dx, dy}` from the bubble center toward the triangle pointer
- `captureBox` — the initial 128×128 capture box placed along the ray

Debug: capture box crops and an annotated overview are saved to the `debug/` output directory.

**Metric:** `traceDurationMs`
→ See [Bubble Detection — Leader Line Tracing](bubble-detection.md#leader-line-tracing--triangle-pointer-detection) for the full algorithm.

### Step 5 — Vision LLM Validation with Progressive Capture Expansion

If Azure OpenAI credentials are configured (`AZURE_ENDPOINT`, `AZURE_KEY`, `AZURE_DEPLOYMENT_NAME`), the pipeline validates each bubble's table dimension against the actual drawing using progressively larger capture boxes.

#### Normal validation (table dimension exists)

For each bubble that has a table dimension from Step 3, the pipeline:

1. Crops a **128×128** region from page 1 along the leader direction
2. Sends the crop + table dimension text to `vision_service.validate_dimension()`
3. If the LLM confirms a match → stop, record as "match"
4. If not → expand to **256×128**, try again
5. Continue through **512×256** and **1024×512**
6. If no match at maximum size → use the LLM's best guess

```python
for cap_w, cap_h in CAPTURE_STEPS:  # [(128,128), (256,128), (512,256), (1024,512)]
    cap = tracer.place_capture_box(bcx, bcy, b_radius, dx, dy, cap_w, cap_h, img_w, img_h)
    crop = page_images[0][y1:y2, x1:x2]
    _, crop_bytes = cv2.imencode(".png", crop)
    validation = await vision_service.validate_dimension(crop_bytes.tobytes(), number, table_dim)
    if validation.matches:
        break  # Stop expanding — match found
```

All capture boxes share the same center point (fixed-anchor placement) so larger boxes expand outward rather than shifting position.

#### Discovery mode (table dimension missing)

When the table OCR fails to extract a dimension for a balloon number, the pipeline uses **discovery mode** instead of skipping the bubble entirely. It captures the initial 128×128 crop and calls `vision_service.discover_dimension()` to read whatever annotation is visible:

```python
discovery = await vision_service.discover_dimension(crop_bytes.tobytes(), number)
```

This ensures all bubbles receive crops and LLM analysis, even when table OCR fails. Discovered dimensions are tagged with source `"LLMOnly"` and status `"discovered"`.

### Step 6 — Merge OCR + LLM Validation

Combines bubble OCR results with Tesseract table output and LLM validation results into a unified `dimensionMap`. See [Merge Algorithm](#the-merge-algorithm) below.

**Metric:** `mergeDurationMs`

### Step 7 — Overlay Generation

Produces an annotated copy of page 1 with color-coded circles and dimension labels. See [Overlay Generation](#overlay-generation) below.

---

## The Merge Algorithm

For each bubble whose OCR successfully read a number, the merge step looks up the dimension value from the Tesseract table and any LLM validation result, then scores and reconciles them.

### Source Classification

| Tesseract value | LLM validation | `source` |
|---|---|---|
| present | validated | `"Table+Validated"` |
| present | not validated | `"TableOnly"` |
| missing | LLM discovered a value | `"LLMOnly"` |
| missing | no LLM result | `"None"` |

### Confidence Scoring

When the LLM validation confirms a match (`llmMatches` is `true`), the LLM's own confidence score is used directly — this avoids false downgrades from string-formatting differences (e.g., `"0.06"` vs `".06"` scoring 75% via Levenshtein but 99% from the LLM).

When the LLM reports a conflict (values don't match), fuzzy string matching quantifies the degree of disagreement. When only a table value exists with no LLM validation, confidence is `0.0`.

```python
if validation and llm_matches:
    conf = llm_confidence          # LLM confirmed — trust its score
elif tess_val and llm_observed:
    conf = confidence_score(tess_val, llm_observed)  # Conflict — measure disagreement
elif validation:
    conf = llm_confidence          # Discovery mode or other LLM-only result
else:
    conf = 0.0                     # No validation at all
```

The scoring algorithm:

1. Normalize whitespace and uppercase both strings.
2. If the normalized strings are equal, return `1.0`.
3. Compute the Levenshtein edit distance between the two.
4. Return `1.0 - (distance / max_length)`, rounded to 4 decimal places.

```python
def confidence_score(a: str | None, b: str | None) -> float:
    if a is None or b is None:
        return 0.0
    na = _normalize_whitespace(a).upper()
    nb = _normalize_whitespace(b).upper()
    if na == nb:
        return 1.0
    max_len = max(len(na), len(nb))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(na, nb)
    return round(1.0 - distance / max_len, 4)
```

### Conflict Detection

A conflict is flagged when the LLM validation says the drawing does not match the table dimension:

```python
has_conflict = llm_result is not None and not llm_result.matches
```

### Final Dimension Selection

The dimension value comes from the Tesseract table when available. When table OCR missed the entry (discovery mode), the LLM's observed value is used instead:

```python
dimension = tess_val or llm_observed
```

---

## Dimension Matching Details

[`dimension_matcher.py`](../engvision-py/src/engvision/services/dimension_matcher.py) provides fuzzy comparison utilities designed for OCR-produced dimension strings.

### Levenshtein Distance

A standard dynamic-programming implementation of edit distance:

```python
def _levenshtein_distance(s: str, t: str) -> int:
    n, m = len(s), len(t)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m]
```

### OCR Character Normalization

Common OCR misreads are corrected contextually:

| OCR char | Replacement | Condition |
|---|---|---|
| `O`, `o` | `0` | Always (in numeric tokens) or adjacent to digit/`.` (in mixed tokens) |
| `I`, `l` | `1` | Always in numeric context |
| `S` | `5` | Surrounded by digits or `.` |
| `B` | `8` | Surrounded by digits or `.` |
| `Ø`, `⌀` | `0` | Always |
| `°`, `º` | `°` | Normalized to standard degree sign |

### Token Type Detection

A token is classified as **numeric** when more than 60% of its characters are digit-like (digits or `./-°OoØ⌀`):

```python
def _is_numeric_token(s: str) -> bool:
    digit_like = sum(1 for c in s if c.isdigit() or c in "./-°OoØ⌀")
    return digit_like / len(s) > 0.6
```

### Mixed Token Handling

For tokens that mix alphabetic and numeric characters (e.g., `"RO.06"`), only `O` adjacent to a digit or `.` is replaced with `0`, preserving leading/trailing letters:

```python
def _normalize_mixed(s: str) -> str:
    result = s.upper()
    chars = list(result)
    for i, c in enumerate(chars):
        if c == "O":
            adj_digit = (
                (i > 0 and (chars[i - 1].isdigit() or chars[i - 1] == "."))
                or (i < len(chars) - 1 and (chars[i + 1].isdigit() or chars[i + 1] == "."))
            )
            if adj_digit:
                chars[i] = "0"
    return "".join(chars)
```

Example: `"RO.06"` → `"R0.06"`.

---

## Overlay Generation

The `_generate_overlay()` function creates an annotated copy of page 1.

### Color Coding

| Condition | Color (BGR) | Meaning |
|---|---|---|
| Dimension matched, no conflict | Green `(0, 200, 0)` | Bubble successfully matched |
| Dimension matched, has conflict | Amber `(0, 200, 255)` | LLM says drawing doesn't match table |
| No dimension found | Red `(0, 0, 255)` | Bubble detected but unmatched |

### Annotations

For each bubble:

1. A **circle** is drawn at `(cx, cy)` with `radius + 3` px and line width 3.
2. A **`#number` label** is placed above the circle (offset −12 px horizontal, −radius − 8 px vertical).
3. If a dimension was found, the **dimension text** is placed to the right of the circle (offset +radius + 8 px horizontal), truncated to 20 characters with an ellipsis.

```python
overlay = page_image.copy()
for bubble in bubbles:
    # ... determine color ...
    cv2.circle(overlay, (bubble["cx"], bubble["cy"]), bubble["radius"] + 3, color, 3)
    cv2.putText(overlay, f"#{num}",
        (bubble["cx"] - 12, bubble["cy"] - bubble["radius"] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if match and match.get("dimension"):
        dim_label = match["dimension"][:20] + "…" if len(match["dimension"]) > 20 else match["dimension"]
        cv2.putText(overlay, dim_label,
            (bubble["cx"] + bubble["radius"] + 8, bubble["cy"] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
```

---

## Output Structure

The pipeline returns a [`PipelineResult`](../engvision-py/src/engvision/models.py) with the following shape:

### PipelineResult

| Field | Type | Description |
|---|---|---|
| `runId` | `str` | Unique identifier for this run |
| `pdfFilename` | `str` | Original PDF filename |
| `pageCount` | `int` | Number of pages rendered |
| `imageWidth` | `int` | Width of page 1 in pixels |
| `imageHeight` | `int` | Height of page 1 in pixels |
| `bubbles` | `list[BubbleResult]` | Detected bubbles, sorted by number |
| `dimensionMap` | `dict[str, DimensionMatch]` | Balloon number (string key) → dimension match |
| `totalBubbles` | `int` | Count of successfully OCR'd bubbles |
| `matchedBubbles` | `int` | Bubbles with at least one dimension source |
| `unmatchedBubbles` | `int` | Bubbles with no dimension found |
| `warnings` | `int` | Count of matches with `0 < confidence < 0.8` |
| `metrics` | `ProcessingMetrics` | Per-step timing breakdown |
| `tokenUsage` | `LlmTokenUsage \| null` | LLM token usage (null if LLM not used) |
| `status` | `str` | `"complete"` or `"error"` |
| `error` | `str \| null` | Error message if `status` is `"error"` |

### BubbleResult

| Field | Type | Description |
|---|---|---|
| `bubbleNumber` | `int` | OCR'd bubble label |
| `cx` | `int` | Center X coordinate |
| `cy` | `int` | Center Y coordinate |
| `radius` | `int` | Bubble radius in pixels |
| `boundingBox` | `BoundingBox` | `{x, y, width, height}` |

### DimensionMatch

| Field | Type | Description |
|---|---|---|
| `balloonNo` | `int` | Balloon/bubble number |
| `dimension` | `str \| null` | Dimension value (from table, or LLM-discovered if table OCR missed) |
| `source` | `str` | `"Table+Validated"`, `"TableOnly"`, `"LLMOnly"`, or `"None"` |
| `tesseractValue` | `str \| null` | Value from table OCR |
| `llmObservedValue` | `str \| null` | What LLM saw on the drawing |
| `llmMatches` | `bool \| null` | Whether LLM says drawing matches table |
| `llmConfidence` | `float` | LLM confidence in its assessment |
| `llmNotes` | `str \| null` | LLM explanation |
| `hasConflict` | `bool` | `true` when LLM says drawing doesn't match table |
| `confidence` | `float` | LLM confidence (on match) or fuzzy match score (on conflict), 0.0–1.0 |
| `captureSize` | `str \| null` | Final capture box size used (e.g. `"128x128"`, `"256x128"`) |

### ProcessingMetrics

| Field | Type | Description |
|---|---|---|
| `totalDurationMs` | `int` | Wall-clock time for entire pipeline |
| `renderDurationMs` | `int` | PDF → image rendering |
| `detectDurationMs` | `int` | Bubble detection |
| `ocrDurationMs` | `int` | Bubble OCR + table OCR |
| `traceDurationMs` | `int` | Leader line tracing |
| `llmDurationMs` | `int` | Vision LLM validation |
| `mergeDurationMs` | `int` | Result merging |
| `peakMemoryMb` | `float` | Peak memory usage (reserved) |

### LlmTokenUsage

| Field | Type | Description |
|---|---|---|
| `inputTokens` | `int` | Total input tokens across all pages |
| `outputTokens` | `int` | Total output tokens |
| `totalTokens` | `int` | Sum of input + output |
| `llmCalls` | `int` | Number of LLM API calls made |

---

## Error Handling

If any exception occurs during pipeline execution, the entire exception is caught and the pipeline returns a result with `status: "error"` and the exception message in the `error` field. All counters are zeroed and collections are empty:

```python
except Exception as ex:
    traceback.print_exc()
    return {
        "runId": run_id,
        "pdfFilename": filename,
        "pageCount": 0,
        "imageWidth": 0,
        "imageHeight": 0,
        "bubbles": [],
        "dimensionMap": {},
        "totalBubbles": 0,
        "matchedBubbles": 0,
        "unmatchedBubbles": 0,
        "warnings": 0,
        "metrics": None,
        "tokenUsage": None,
        "status": "error",
        "error": str(ex),
    }
```

The full stack trace is printed to stderr for server-side debugging. Callers should check the `status` field before consuming the rest of the response.

---

## Real-Time Pipeline Streaming (SSE)

The pipeline supports **Server-Sent Events (SSE)** for real-time progress reporting through dedicated streaming endpoints:

| Endpoint | Description |
|---|---|
| `POST /api/pipeline/run-stream` | Upload a PDF and stream progress |
| `POST /api/pipeline/run-stream-sample/{filename}` | Stream progress for a sample PDF |

### Why POST + ReadableStream (not EventSource)

Native `EventSource` only supports GET requests, but the pipeline needs POST for file upload. The frontend uses `fetch()` with a `ReadableStream` reader to parse SSE events manually.

### Event Types

The stream emits newline-delimited SSE events (`event: <type>\ndata: <json>\n\n`):

| Event | Fields | Emitted when |
|---|---|---|
| `step` | `step`, `totalSteps`, `name`, `message` | A pipeline step begins |
| `stepComplete` | `step`, `name`, `durationMs`, `detail` | A pipeline step finishes |
| `bubble` | `bubbleNumber`, `captureSize`, `status`, `tableDim`, `observed`, `confidence` | A bubble validation completes |
| `complete` | Full `PipelineResult` | Pipeline finishes successfully |
| `error` | `message` | Pipeline fails |

### Bubble Status Values

| Status | Meaning | UI Color |
|---|---|---|
| `match` | LLM confirmed dimension matches table | Green |
| `expanding` | Current capture size didn't match, trying larger | Yellow |
| `bestGuess` | Maximum capture size reached, using LLM's best guess | Orange |
| `discovered` | No table entry; LLM read the dimension directly | Purple |
| `noMatch` | LLM could not find any dimension | Red |

### Frontend Integration

The `usePipelineStream` React hook manages the SSE connection:

```typescript
const { state, startStream } = usePipelineStream();
// state.steps — 7 pipeline steps with timing
// state.bubbleStatuses — Map<number, BubbleStatus> per bubble
// state.elapsed — wall-clock time
// state.llmCalls — total LLM API calls
```

The `PipelineProgressView` component renders:
- 7-step progress bars with per-step timing
- 50-cell color-coded bubble grid showing validation status
- Current bubble detail panel
- Elapsed time counter

---

## Benchmark Logging

Both backends write a `benchmark.json` file to the run's output directory after pipeline completion:

```json
{
  "runId": "abc123",
  "backend": "python",
  "pdfFilename": "WFRD_Sample_Dimentional_Analysis.pdf",
  "totalDurationMs": 45230,
  "renderDurationMs": 1200,
  "detectDurationMs": 340,
  "ocrDurationMs": 2800,
  "traceDurationMs": 18,
  "llmDurationMs": 38500,
  "mergeDurationMs": 5,
  "overlayDurationMs": 120,
  "totalBubbles": 50,
  "matchedBubbles": 48,
  "llmCalls": 54,
  "timestamp": "2026-02-27T13:00:00Z"
}
```

The `backend` field (`"python"` or `"dotnet"`) enables direct comparison of processing times between the two implementations.

---

## See Also

- [Bubble Detection](bubble-detection.md) — Circle-detection algorithm and tuning
- [Table & OCR](table-ocr.md) — Tesseract configuration and table extraction
- [LLM Validation](llm-extraction.md) — Azure OpenAI Vision validation prompting strategy
