# EngVision — CAD Dimensional Analysis Extraction

A .NET 10 + React toolkit that uses **OpenCvSharp4** to detect numbered measurement bubbles and table regions in engineering CAD drawings (PDF), segments them into cropped images, and feeds them through a **vision LLM** (GPT-4o / GPT-5.2) via **Microsoft.Extensions.AI** to extract and cross-check dimensional data.

Includes a **web-based annotation dashboard** for manually labeling ground truth bounding boxes, comparing against auto-detection, and computing precision/recall/F1 metrics to tune the detection pipeline.

## Architecture

```
eng-vision.sln
├── EngVision/              # Core library: OpenCV detection, PDF rendering, vision LLM
├── EngVision.Api/          # .NET Minimal API serving detection + annotation endpoints
├── EngVision.Web/          # React + Vite annotation dashboard
├── EngVision.AppHost/      # Aspire orchestrator (runs API + frontend together)
├── EngVision.ServiceDefaults/  # Aspire service defaults
└── sample_docs/            # Sample CAD PDFs
```

## Quick Start

### Run with Aspire (recommended)

```bash
cd EngVision.AppHost
dotnet run
```

This starts both the API and React frontend. Open the Aspire dashboard URL shown in the console to see endpoints.

### Run standalone CLI (detection only)

```bash
cd EngVision
dotnet run -- ..\sample_docs\WFRD_Sample_Dimentional_Analysis.pdf
```

### Run with vision LLM extraction

```bash
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_MODEL = "gpt-4o"
cd EngVision
dotnet run -- ..\sample_docs\WFRD_Sample_Dimentional_Analysis.pdf
```

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

1. **PDF → Image**: Renders pages at 300 DPI via PDFium/SkiaSharp
2. **Bubble Detection**: Multi-pass HoughCircles + strict verification (white interior, dark text, visible outline)
3. **Leader Line Tracing**: Follows leader lines from bubbles to find associated dimension text
4. **Table Detection**: Morphological line detection for tabular data on pages 2-4
5. **Vision LLM**: Feeds cropped segments to GPT-4o/5.2 for structured data extraction
6. **Comparison**: Matches bubble measurements against table values

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
| DELETE | `/api/pdfs/{file}/pages/{n}/annotations/{id}` | Delete annotation |
| GET | `/api/pdfs/{file}/annotations/export` | Export ground truth JSON |
| GET | `/api/pdfs/{file}/pages/{n}/accuracy` | Compute detection accuracy |

## NuGet / npm Packages

| Package | Purpose |
|---------|---------|
| `OpenCvSharp4` + `runtime.win` | Computer vision |
| `PDFtoImage` / `SkiaSharp` | PDF rendering |
| `Microsoft.Extensions.AI` | AI abstraction |
| `Microsoft.Extensions.AI.OpenAI` | OpenAI vision integration |
| `Aspire.Hosting.NodeJs` | Aspire Node.js hosting |
| `react` + `vite` | Annotation dashboard UI |
