# EngVision Documentation

> Technical documentation for the EngVision engineering drawing analysis system.
> All code references point to the **Python** (`engvision-py`) implementation.

## Architecture Diagrams

Open these in [Excalidraw](https://excalidraw.com) for interactive viewing (or use the VS Code Excalidraw addon):

| Diagram | Description |
|---------|-------------|
| [System Architecture](diagrams/architecture.excalidraw.json) | High-level component diagram — Aspire orchestrator, frontend, API backends, external services |
| [Pipeline Flow](diagrams/pipeline-flow.excalidraw.json) | Step-by-step data flow through the 8-stage analysis pipeline |
| [Bubble Detection](diagrams/bubble-detection.excalidraw.json) | Multi-pass HoughCircles detection with 4 verification gates |

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture.md) | System components, Aspire orchestration, backend toggle, tech stack comparison |
| [Bubble Detection](bubble-detection.md) | OpenCV circle detection — preprocessing, multi-pass HoughCircles, verification gates, flood-fill + Harris triangle pointer detection |
| [Table & OCR Processing](table-ocr.md) | Table region detection, grid-based cell OCR, bubble number OCR via Tesseract |
| [LLM Validation](llm-extraction.md) | Azure OpenAI vision integration — validates table dimensions against drawing annotations, discovery mode for missing entries |
| [Pipeline Orchestration](pipeline.md) | End-to-end pipeline — render → detect → OCR → trace → validate → merge → overlay, with SSE streaming and benchmark logging |
| [API Reference](api-reference.md) | REST endpoints including SSE streaming, with request/response examples and data type definitions |

## Reading Order

For a new engineer onboarding to this codebase:

1. **[Architecture Overview](architecture.md)** — understand the system components and how they connect
2. **[Pipeline Orchestration](pipeline.md)** — understand the end-to-end flow
3. **[Bubble Detection](bubble-detection.md)** — deep-dive into the core CV algorithm
4. **[Table & OCR Processing](table-ocr.md)** — understand text extraction
5. **[LLM Validation](llm-extraction.md)** — understand the AI-assisted validation
6. **[API Reference](api-reference.md)** — endpoint details for frontend integration

## Quick Links

- [Main README](../README.md) — setup instructions, backend toggle, running the app
- [Python API source](../engvision-py/src/engvision/) — FastAPI implementation
- [Frontend source](../EngVision.Web/) — React + Vite frontend
- [Aspire AppHost](../EngVision.AppHost/) — orchestrator configuration

## Computer Vision Glossary

New to CV? Here's a quick reference for the algorithms used throughout the docs. Each links to the OpenCV tutorial.

| Term | What it does | Where we use it |
|------|-------------|-----------------|
| [HSV Color Space](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html) | Separates color (Hue), intensity (Saturation), and brightness (Value) — makes it easy to isolate a color regardless of lighting | Blue bubble mask extraction |
| [Gaussian Blur](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html) | Smooths an image by averaging pixels with a bell-curve weighted kernel — reduces noise | Pre-processing before circle detection |
| [Morphological Operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) | Erosion (shrink white regions), dilation (expand them), open (erode→dilate), close (dilate→erode) — clean up binary masks | Table grid isolation, blue mask cleanup |
| [Adaptive Threshold](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) | Converts grayscale to black/white using a locally-computed cutoff per pixel — handles uneven lighting | Table detection, grayscale bubble detection |
| [Hough Circle Transform](https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html) | Detects circles by having edge pixels vote for possible centers — where votes accumulate, a circle is found | Primary bubble detection |
| [Contours](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html) | Traces the outlines of connected white regions in a binary image — returns point arrays representing shapes | Bubble and table region boundaries |
| [Flood Fill](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga366aae45a6c1289b341d140c2c547f2f) | Paint-bucket spread from a seed pixel to all connected same-color pixels | Isolating a single bubble's connected component |
| [Harris Corner Detection](https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html) | Measures how pixel intensity changes in all directions — high change = corner, one direction = edge, no change = flat | Finding triangle pointer vertices |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Open-source engine that converts images of text into machine-readable strings | Reading bubble numbers and table dimensions |
| [IoU (Intersection over Union)](https://en.wikipedia.org/wiki/Jaccard_index) | Ratio of overlapping area to total area of two bounding boxes — measures detection accuracy | Accuracy evaluation against ground truth |
