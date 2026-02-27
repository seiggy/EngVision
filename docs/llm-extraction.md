# Vision LLM Validation

## Purpose

The Vision LLM service **validates** that dimension values extracted from the
inspection table (via Tesseract OCR) match the dimension annotations visible on the
engineering drawing. It does **not** extract data from the table itself — Tesseract
handles that. For each bubble that has a table dimension, the pipeline crops
progressively larger regions along the leader line and asks an Azure OpenAI vision
model to confirm the annotation on the drawing matches the table value. When the LLM
reports a mismatch, a conflict is flagged so an operator can review it.

When a bubble has **no table dimension** (table OCR missed it), the service switches
to **discovery mode** — asking the LLM to read whatever annotation is visible without
a comparison value.

See also: [Table Detection & OCR](table-ocr.md) · [Pipeline Orchestration](pipeline.md) · [Architecture](architecture.md) · [Bubble Detection](bubble-detection.md)

---

## Configuration

The LLM service is **opt-in** — it runs only when `AZURE_KEY` and `AZURE_ENDPOINT`
are both set. Three environment variables control the connection:

| Variable | Description | Default |
|---|---|---|
| `AZURE_ENDPOINT` | Azure Cognitive Services endpoint URL | *(required)* |
| `AZURE_KEY` | API key for the Azure OpenAI resource | *(required)* |
| `AZURE_DEPLOYMENT_NAME` | Model deployment name | `gpt-5.3-codex` |

### Client initialisation

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=key,
    azure_endpoint=endpoint,
    api_version="2024-12-01-preview",
)
vision_service = VisionLlmService(client, model)
```

The pipeline reads these values from the environment at runtime
(`os.environ.get(…)`). If the key or endpoint is missing the LLM step is silently
skipped and only Tesseract results are used.

---

## What We Send

For **each bubble** on page 1 that has a table dimension from Tesseract OCR:

1. The `LeaderLineTracerService` uses flood-fill + Harris corner detection to
   determine the triangle pointer direction from the bubble.
2. A **128×128** capture box is placed along the leader direction and cropped as PNG.
3. The PNG bytes are **base64-encoded** and embedded in an `image_url` content
   block alongside the table dimension text for that balloon number.
4. If the LLM does not confirm a match, the capture box **expands** progressively
   through 256×128 → 512×256 → 1024×512 until a match is found or the maximum
   size is reached.

```python
for cap_w, cap_h in CAPTURE_STEPS:  # [(128,128), (256,128), (512,256), (1024,512)]
    cap = tracer.place_capture_box(bcx, bcy, b_radius, dx, dy, cap_w, cap_h, img_w, img_h)
    crop = page_images[0][y1:y2, x1:x2]
    _, crop_bytes = cv2.imencode(".png", crop)
    validation = await vision_service.validate_dimension(crop_bytes.tobytes(), number, table_dim)
    if validation.matches:
        break  # Match found — stop expanding
```

Each bubble is a **separate API call** (potentially multiple per bubble due to expansion). Results are accumulated per balloon number.

---

## The Prompt

### System prompt

The full system prompt sent with every validation call:

```text
You are an expert at reading engineering drawings and dimensional annotations.

You will be given:
1. A cropped region from an engineering drawing where a numbered balloon/bubble
   points via its leader line.
2. A dimension value extracted from the inspection table for that balloon number.

Your job is to:
- Examine the cropped drawing region and find the dimension annotation visible there.
- Compare it to the table dimension value provided.
- Determine if they match (accounting for formatting differences like leading zeros,
  degree symbols, diameter symbols, etc.).

Return a JSON object:
{
  "observedDimension": "<the dimension text you see on the drawing, or empty string if none visible>",
  "matches": <true if the drawing dimension matches the table value, false otherwise>,
  "confidence": <0.0 to 1.0 confidence in your assessment>,
  "notes": "<brief explanation, e.g. 'exact match', 'formatting difference only',
   'dimension not visible in crop', etc.>"
}

Rules:
- If you cannot see any dimension annotation in the crop, set observedDimension to ""
  and matches to false with a low confidence.
- Treat formatting variations as matches (e.g. '0.81' vs '.81', '18°' vs '18 DEG',
  'Ø.500' vs 'DIA .500').
- Return ONLY the JSON object, no other text.
```

### Why each instruction matters

| Instruction | Rationale |
|---|---|
| **Cropped drawing region** | The model sees only the area the leader line points to, keeping the visual context focused and reducing hallucination. |
| **Table dimension provided** | Gives the model a concrete value to compare against rather than asking it to extract from scratch. |
| **Formatting-tolerant matching** | Engineering drawings use inconsistent notation (`°` vs `DEG`, leading zeros, `Ø` vs `DIA`). The model must treat these as equivalent. |
| **Empty-string fallback** | Some crops may not contain a readable annotation (e.g. the leader line ends at a surface). The model should honestly report nothing visible. |
| **JSON-only response** | Eliminates prose or markdown that would complicate parsing. |

### User message

The user message accompanying the image:

```text
Balloon #<balloon_no>
Table dimension value: "<table_dimension>"

Examine the drawing region below and validate whether the dimension annotation
matches the table value:
```

---

## Discovery Mode

When the table OCR misses a balloon number's dimension, the pipeline uses **discovery mode** instead of skipping the bubble. This ensures every bubble receives LLM analysis and capture crops for debugging.

### Discovery Prompt

```text
You are an expert at reading engineering drawings and dimensional annotations.

You will be given a cropped region from an engineering drawing where a numbered
balloon/bubble points via its leader line.

Your job is to:
- Examine the cropped drawing region and find any dimension annotation visible there.
- Report what you see.

Return a JSON object:
{
  "observedDimension": "<the dimension text you see, or empty string if none>",
  "confidence": <0.0 to 1.0 confidence in your reading>,
  "notes": "<brief explanation>"
}

Rules:
- If you cannot see any dimension annotation in the crop, set observedDimension to ""
  with a low confidence.
- Return ONLY the JSON object, no other text.
```

### Discovery User Message

```text
Balloon #<balloon_no>

Examine the drawing region below and report any dimension annotation you can see:
```

### Discovery Response

The discovery response has no `matches` field (nothing to compare against). The pipeline treats finding any observed text as a success and tags the result with source `"LLMOnly"` and status `"discovered"`.

```python
discovery = await vision_service.discover_dimension(crop_bytes.tobytes(), number)
# discovery.observed_dimension = "0.81"  (what the LLM read)
# discovery.confidence = 0.92
```

---

## Response Parsing

```python
content = response.choices[0].message.content or ""
content = _strip_code_fences(content)
parsed = json.loads(content)

return LlmValidationResult(
    balloon_no=balloon_no,
    table_dimension=table_dimension,
    observed_dimension=parsed.get("observedDimension", ""),
    matches=bool(parsed.get("matches", False)),
    confidence=float(parsed.get("confidence", 0.0)),
    notes=parsed.get("notes", ""),
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    total_tokens=total_tokens,
)
```

Key details:

1. **Code-fence stripping** — The model occasionally returns `` ```json … ``` ``.
   `_strip_code_fences()` removes the opening and closing fence lines before
   parsing.
2. **JSON parse** — The response is expected to be a single JSON object with
   `observedDimension`, `matches`, `confidence`, and `notes` fields.
3. **Fallback on error** — If the API call or JSON parse fails, a default
   `LlmValidationResult` is returned with `matches=False` and zero confidence,
   so validation failures are non-fatal.
4. **Empty response** — If the model returns an empty string, a default result
   is returned (no observed dimension, no match).

The helper that strips fences:

```python
def _strip_code_fences(content: str) -> str:
    if content.startswith("```"):
        lines = content.split("\n")
        start_idx = 1 if lines[0].startswith("```") else 0
        end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        content = "\n".join(lines[start_idx:end_idx])
    return content.strip()
```

---

## Token Usage Tracking

Every API call returns token counts. The `LlmValidationResult` dataclass captures
them per-bubble:

```python
@dataclass
class LlmValidationResult:
    balloon_no: int = 0
    table_dimension: str = ""
    observed_dimension: str = ""
    matches: bool = False
    confidence: float = 0.0
    notes: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
```

The `LlmValidationBatchResult` dataclass aggregates across all bubbles:

```python
@dataclass
class LlmValidationBatchResult:
    validations: dict[int, LlmValidationResult] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
```

The pipeline **accumulates** tokens across all per-bubble calls:

```python
llm_input_tokens += validation.input_tokens
llm_output_tokens += validation.output_tokens
llm_total_tokens += validation.total_tokens
llm_calls += 1
```

These totals are surfaced in the final `PipelineResult.tokenUsage`:

```json
{
  "inputTokens": 24530,
  "outputTokens": 1842,
  "totalTokens": 26372,
  "llmCalls": 3
}
```

`tokenUsage` is `null` when no LLM calls were made (key/endpoint not configured).

---

## How LLM Results Merge with Tesseract

The dimension value comes from Tesseract OCR of the table when available. When table
OCR missed the entry, the LLM's discovered value is used instead. During the merge
step the pipeline checks validation status for each balloon number:

| Scenario | `source` | Behaviour |
|---|---|---|
| Table dimension exists and LLM validated it | `"Table+Validated"` | When LLM confirms match, its confidence is used directly. When LLM reports conflict, fuzzy `confidence_score(tess_val, llm_observed)` quantifies disagreement. `hasConflict` is `true` on mismatch. |
| Table dimension exists but no LLM validation | `"TableOnly"` | Used directly from Tesseract; confidence `0.0`. |
| No table dimension, LLM discovered a value | `"LLMOnly"` | Dimension set to LLM's observed value; confidence from LLM. Status `"discovered"`. |
| No table dimension, no LLM result | `"None"` | Bubble is unmatched; confidence `0.0`. |

Each entry in `dimensionMap` carries the validation details:

| Field | Description |
|---|---|
| `dimension` | Table value from Tesseract, or LLM-discovered value if table OCR missed. |
| `llmObservedValue` | What the LLM saw on the drawing crop (may differ from table). |
| `llmMatches` | `true` if the LLM determined drawing and table agree. |
| `llmConfidence` | LLM's self-reported confidence (0.0–1.0). |
| `llmNotes` | Brief explanation from the LLM (e.g. "exact match", "formatting difference only"). |
| `hasConflict` | `true` when the LLM says the drawing doesn't match the table. |
| `captureSize` | Final capture box size used (e.g. `"128x128"`, `"256x128"`). |

For the full merge logic, confidence scoring, and conflict resolution see
[Pipeline Orchestration](pipeline.md).
