# EngVision — CAD Dimensional Analysis Extraction

A toolkit that detects numbered measurement bubbles and table regions in engineering CAD drawings (PDF), segments them into cropped images, and feeds them through a **vision LLM** (GPT-5.3-codex) to extract and cross-check dimensional data.

Includes a **web-based annotation dashboard** for manually labeling ground truth bounding boxes, comparing against auto-detection, and computing precision/recall/F1 metrics to tune the detection pipeline.

The API backend is available in two implementations — **.NET** and **Python** — selectable via a single configuration flag.

## Architecture

```
eng-vision.sln
├── EngVision/              # Core library: OpenCV detection, PDF rendering, vision LLM (.NET)
├── EngVision.Api/          # .NET Minimal API serving detection + annotation endpoints
├── engvision-py/           # Python (FastAPI) port of the API — same endpoints, same behaviour
├── EngVision.Web/          # React + Vite annotation dashboard
├── EngVision.AppHost/      # Aspire orchestrator (runs API + frontend together)
├── EngVision.ServiceDefaults/  # Aspire service defaults
└── sample_docs/            # Sample CAD PDFs
```

## Quick Start

### Prerequisites

| Tool | Required for |
|------|-------------|
| [.NET 10 SDK](https://dotnet.microsoft.com/) | Building & running the .NET backend and Aspire host |
| [Node.js 20+](https://nodejs.org/) | React frontend |
| [Aspire CLI](https://learn.microsoft.com/dotnet/aspire) | Orchestrating the full stack |
| [uv](https://docs.astral.sh/uv/) | Python backend only — environment & dependency management |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Python backend only — system install needed for `pytesseract` |

### Run with Aspire (recommended)

By default the **.NET** backend is used:

```bash
aspire run
```

To run the **Python** backend instead, set the `ApiBackend` flag to `python` using any of these methods:

```bash
# Option 1 — CLI argument
aspire run -- --ApiBackend python

# Option 2 — Environment variable (PowerShell)
$env:ApiBackend = "python"; aspire run

# Option 3 — Environment variable (bash/zsh)
ApiBackend=python aspire run
```

Or change it persistently in `EngVision.AppHost/appsettings.json`:

```jsonc
{
  "ApiBackend": "python"   // "dotnet" (default) or "python"
}
```

Aspire starts the selected API backend, waits for it to be healthy, then launches the React frontend with the proxy automatically pointed at the API. Open the Aspire dashboard URL shown in the console to see all resource endpoints.

### Run standalone CLI (detection only — .NET)

```bash
cd EngVision
dotnet run -- ..\sample_docs\WFRD_Sample_Dimentional_Analysis.pdf
```

### Run the Python API standalone (without Aspire)

```bash
cd engvision-py
uv run engvision          # starts uvicorn on port 5062
```

### Run with vision LLM extraction

Create a `.env` file in the repo root (or set the variables directly):

```env
AZURE_ENDPOINT=https://your-openai-resource.cognitiveservices.azure.com
AZURE_KEY=your-key-here
AZURE_DEPLOYMENT_NAME=gpt-5.3-codex
```

Both backends read this file automatically.

## API Backend Comparison

| | .NET (`EngVision.Api`) | Python (`engvision-py`) |
|---|---|---|
| Framework | ASP.NET Minimal APIs | FastAPI + Uvicorn |
| PDF rendering | PDFtoImage + SkiaSharp | PyMuPDF (fitz) |
| Computer vision | OpenCvSharp4 | opencv-python |
| OCR | TesseractOCR NuGet | pytesseract (requires system Tesseract) |
| LLM | OpenAI .NET SDK | openai Python SDK |
| Package manager | NuGet (dotnet restore) | uv (uv sync) |
| Aspire integration | `AddProject<>()` | `AddUvicornApp()` + `WithUv()` |

Both backends expose **identical API endpoints** and return the same JSON shapes, so the React frontend works with either one without changes.

## Annotation Dashboard

The web dashboard (`EngVision.Web`) lets you:

- **View** rendered PDF pages with zoom/pan (scroll to zoom, Alt+drag to pan)
- **Draw** bounding boxes around bubbles by clicking and dragging
- **Edit** annotations: assign bubble numbers, labels, notes
- **Move/resize** existing boxes with drag handles
- **Auto-detect** bubbles using the OpenCV pipeline and see results overlaid
- **Compare** manual annotations vs auto-detections with IoU-based accuracy metrics
- **Export** ground truth JSON for training/tuning

### Controls

| Action | Input |
|--------|-------|
| Draw box | Click + drag |
| Pan | Alt + click + drag |
| Zoom | Mouse wheel |
| Select | Click on annotation |
| Delete | Select → Delete button in panel |

## Detection Pipeline

1. **PDF → Image**: Renders pages at 300 DPI
2. **Bubble Detection**: Multi-pass [HoughCircles](https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html) (detects circles via edge-pixel voting) + strict verification (white interior, dark text, visible outline)
3. **Leader Line Tracing**: [Flood-fill](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga366aae45a6c1289b341d140c2c547f2f) (isolates connected pixels) + [Harris corner detection](https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html) (finds sharp corners) isolates triangle pointers from overlapping bubbles, determines leader direction
4. **Capture Box Placement**: Progressive expansion (128×128 → 256×128 → 512×256 → 1024×512) along the leader direction with fixed-anchor centering
5. **Table Detection**: [Morphological](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) line detection (erosion/dilation to isolate table grid lines) for tabular data on pages 2-4; [Tesseract](https://github.com/tesseract-ocr/tesseract) OCR extracts dimension values
6. **Vision LLM Validation**: Sends cropped capture boxes to Azure OpenAI to confirm drawing annotations match table values; uses discovery mode when table OCR misses an entry
7. **Result Merge**: Combines Tesseract + LLM results with source classification (`Table+Validated`, `TableOnly`, `LLMOnly`, `None`) and confidence scoring

Real-time progress is streamed via **SSE** (Server-Sent Events) through the `/api/pipeline/run-stream` endpoints.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/pdfs` | List available PDFs |
| GET | `/api/pdfs/{file}/info` | Get page count + dimensions |
| GET | `/api/pdfs/{file}/pages/{n}/image` | Render page as PNG |
| POST | `/api/pdfs/{file}/pages/{n}/detect` | Run auto-detection |
| GET | `/api/pdfs/{file}/pages/{n}/annotations` | Get manual + auto annotations |
| POST | `/api/pdfs/{file}/pages/{n}/annotations` | Save manual annotation |
| PUT | `/api/pdfs/{file}/pages/{n}/annotations/{id}` | Update annotation |
| DELETE | `/api/pdfs/{file}/pages/{n}/annotations` | Clear page annotations |
| DELETE | `/api/pdfs/{file}/pages/{n}/annotations/{id}` | Delete annotation |
| GET | `/api/pdfs/{file}/annotations/export` | Export ground truth JSON |
| GET | `/api/pdfs/{file}/pages/{n}/accuracy` | Compute detection accuracy |
| POST | `/api/pipeline/run` | Upload PDF and run full pipeline |
| POST | `/api/pipeline/run-sample/{file}` | Run pipeline on a sample PDF |
| POST | `/api/pipeline/run-stream` | Upload PDF and stream progress (SSE) |
| POST | `/api/pipeline/run-stream-sample/{file}` | Stream pipeline progress for a sample (SSE) |
| GET | `/api/pipeline/{runId}/results` | Get pipeline run results |
| GET | `/api/pipeline/{runId}/pages/{n}/image` | Get rendered page from a run |
| GET | `/api/pipeline/{runId}/pages/{n}/overlay` | Get annotated overlay image |
| GET | `/api/pipeline/runs` | List all pipeline runs |
