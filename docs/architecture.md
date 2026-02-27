# Architecture Overview

EngVision is an automated analysis system for engineering CAD drawings. It ingests
multi-page PDF drawings, locates numbered measurement bubbles (balloons) on the
drawing page, reads dimension tables on subsequent pages via OCR, traces leader
lines to find where each bubble points on the drawing, then uses a vision LLM to
validate that the dimension on the drawing matches the table value.

The primary users are quality and manufacturing engineers who need to verify that
every dimensioned feature on a drawing has a corresponding entry in the inspection
table — a task that is tedious and error-prone when done manually.

---

## Component Diagram

![Architecture](diagrams/architecture.excalidraw.json)

The system is composed of four major components:

| Component | Directory | Role |
|---|---|---|
| **Aspire AppHost** | `EngVision.AppHost/` | Orchestrates all resources — API backend and frontend — via .NET Aspire. |
| **Python API** | `engvision-py/` | FastAPI service that owns the analysis pipeline: PDF rendering, bubble detection, leader line tracing, OCR, LLM validation, and result merging. |
| **.NET API** | `EngVision.Api/` | Alternative ASP.NET Core backend with the same route surface. Shares core logic from the `EngVision/` class library. |
| **Frontend** | `EngVision.Web/` | React + Vite SPA. Displays drawings, overlays detected bubbles, shows dimension tables, and drives the pipeline. |

The AppHost starts exactly one API backend (Python **or** .NET) and the frontend.
The frontend talks to whichever backend is active through its Vite dev-server proxy;
the browser never contacts the API directly.

---

## Aspire Orchestration

The entire application is launched with a single command:

```bash
aspire run
```

All resource wiring lives in
[`EngVision.AppHost/AppHost.cs`](../EngVision.AppHost/AppHost.cs):

```csharp
var usePython = string.Equals(
    builder.Configuration["ApiBackend"], "python", StringComparison.OrdinalIgnoreCase);

if (usePython)
{
    var api = builder.AddUvicornApp("api", "../engvision-py", "engvision.app:app")
        .WithUv();

    builder.AddNpmApp("frontend", "../EngVision.Web", "dev")
        .WithReference(api).WaitFor(api)
        .WithHttpEndpoint(env: "PORT")
        .WithExternalHttpEndpoints();
}
else
{
    var api = builder.AddProject<Projects.EngVision_Api>("api");
    // same frontend wiring …
}
```

### `ApiBackend` toggle

The `ApiBackend` setting selects which backend starts. It defaults to `"dotnet"` in
`appsettings.json` and can be overridden three ways (highest priority first):

1. **CLI argument** — `aspire run -- --ApiBackend python`
2. **Environment variable** — `set ApiBackend=python`
3. **Config file** — set `"ApiBackend": "python"` in
   `EngVision.AppHost/appsettings.json`

### `WithUv()` — Python dependency management

When the Python backend is selected, `AddUvicornApp` launches a Uvicorn process
inside the `engvision-py/` directory, targeting the ASGI application object
`engvision.app:app`. The chained `.WithUv()` call tells Aspire to use
[`uv`](https://docs.astral.sh/uv/) for dependency resolution and virtual-environment
creation, so `pip install` is never needed — Aspire handles it automatically from
`pyproject.toml`.

### `WithReference()` — frontend → API wiring

`.WithReference(api)` injects the API's base URL into the frontend process as the
environment variable `services__api__https__0` (or the `http` variant). The Vite
dev-server reads this at startup to configure its proxy target (see
[Frontend](#frontend) below). `.WaitFor(api)` ensures the frontend only starts
after the API is healthy.

---

## Tech Stack Comparison

Both backends expose the same REST API surface under `/api`. The table below
compares the underlying libraries:

| Concern | Python backend (`engvision-py/`) | .NET backend (`EngVision.Api/` + `EngVision/`) |
|---|---|---|
| **Web framework** | FastAPI 0.115 + Uvicorn | ASP.NET Core (minimal APIs) |
| **PDF rendering** | PyMuPDF (`pymupdf`) | PDFtoImage + SkiaSharp |
| **Computer vision** | OpenCV (`opencv-python-headless`) | OpenCvSharp4 |
| **OCR engine** | Tesseract via `pytesseract` | TesseractOCR (C# binding) |
| **LLM SDK** | `openai` Python SDK (Azure OpenAI) | Azure.AI.OpenAI + Microsoft.Extensions.AI |
| **Image processing** | NumPy + Pillow | SkiaSharp |
| **Target runtime** | Python ≥ 3.11 | .NET 10 |
| **Package manager** | uv (from `pyproject.toml`) | NuGet |

The Python backend is the primary development target. The .NET backend exists as
the original implementation and is maintained for parity, but new features land
in Python first.

---

## Frontend

The frontend is a React application scaffolded with Vite and lives in
`EngVision.Web/`. It is started by Aspire as an `NpmApp` resource running
`npm run dev`.

### Vite proxy

All API calls from the browser use the relative prefix `/api` (see
[`EngVision.Web/src/api.ts`](../EngVision.Web/src/api.ts)):

```typescript
const BASE = '/api';
// e.g. fetch(`${BASE}/pdfs`)
```

The Vite dev-server forwards these requests to whichever backend Aspire started.
The proxy configuration in
[`EngVision.Web/vite.config.ts`](../EngVision.Web/vite.config.ts) reads the
service URL that Aspire injected:

```typescript
proxy: {
  '/api': {
    target: process.env.services__api__https__0
         || process.env.services__api__http__0
         || 'http://localhost:5062',
    changeOrigin: true,
    secure: false,
  },
},
```

The fallback `http://localhost:5062` matches the default port used by the Python
backend's `uvicorn.run()` call in `engvision-py/src/engvision/app.py`, so the
frontend also works when run standalone outside Aspire.

---

## Data Flow Summary

A typical pipeline run follows these steps:

```
PDF upload
  │
  ▼
1. Render — PyMuPDF converts each page to a 300 DPI image
  │
  ▼
2. Detect — OpenCV Hough circle transform finds numbered bubbles on page 1
  │
  ▼
3. OCR — Tesseract reads bubble numbers from cropped circles;
         Tesseract also reads the dimension table on pages 2+
  │
  ▼
4. Trace — Flood-fill + Harris corner detection finds triangle
           pointer direction; capture boxes placed along ray
  │
  ▼
5. Validate — Vision LLM validates progressive capture crops
             against table dimensions; discovery mode for missing entries
  │
  ▼
6. Merge — Table OCR + LLM results joined; sources classified,
           conflicts flagged, confidence scored
  │
  ▼
7. Output — JSON result + annotated overlay + benchmark.json
```

Each step is implemented as a service class under
`engvision-py/src/engvision/services/`:

| Step | Service module | Key class / function |
|---|---|---|
| Render | `pdf_renderer.py` | `PdfRendererService` |
| Detect | `bubble_detection.py` | `BubbleDetectionService` |
| OCR (bubbles) | `bubble_ocr.py` / `azure_bubble_ocr.py` | `BubbleOcrService` or `AzureBubbleOcrService` |
| OCR (tables) | `table_ocr.py` / `azure_table_ocr.py` | `TableOcrService` or `AzureTableOcrService` |
| Leader line trace | `leader_line_tracer.py` | `LeaderLineTracerService` |
| LLM validation | `vision_llm.py` | `VisionLlmService` (`validate_dimension`, `discover_dimension`) |
| Merge | `dimension_matcher.py` | `are_similar`, `confidence_score` |
| Orchestration | `pipeline.py` | `PipelineService.run_async`, `PipelineService.run_stream` |

The merge step uses `dimension_matcher.confidence_score()` as a fallback when the
LLM reports a conflict. When the LLM confirms a match, its confidence is used
directly. Results are flagged with `hasConflict: true` when the LLM says the
drawing doesn't match the table.

For the full pipeline walkthrough, see [Pipeline Orchestration](pipeline.md).

---

## Further Reading

- [Bubble Detection Deep Dive](bubble-detection.md) — Hough circle parameters,
  context padding, and tuning guidance.
- [Table Detection & OCR](table-ocr.md) — How dimension tables are located and
  parsed with Tesseract.
- [LLM Validation](llm-extraction.md) — Prompt design, Azure OpenAI
  configuration, and token-usage tracking.
- [Pipeline Orchestration](pipeline.md) — End-to-end flow from PDF upload to
  merged output, including progress reporting and error handling.
- [API Reference](api-reference.md) — Complete route listing for the `/api`
  surface shared by both backends.
