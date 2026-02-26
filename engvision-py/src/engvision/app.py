"""FastAPI application with all API routes, ported from .NET EngVision.Api."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .annotation_store import AnnotationStore
from .config import EngVisionConfig
from .services.bubble_detection import BubbleDetectionService
from .services.pdf_renderer import PdfRendererService
from .services.pipeline import PipelineService
from .services.table_detection import TableDetectionService

# ── Load .env from repo root ────────────────────────────────────────────────────
_base_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_base_dir, "..", "..", ".."))
_env_file = os.path.join(_repo_root, ".env")
if os.path.exists(_env_file):
    load_dotenv(_env_file)

# ── Configuration ────────────────────────────────────────────────────────────────
config = EngVisionConfig(
    pdf_render_dpi=300,
    output_directory=os.path.join(_base_dir, "..", "..", "Output"),
)
os.makedirs(config.output_directory, exist_ok=True)

# ── Services (singletons) ───────────────────────────────────────────────────────
renderer = PdfRendererService(config.pdf_render_dpi)
bubble_detector = BubbleDetectionService(config)
table_detector = TableDetectionService(config)
annotation_store = AnnotationStore(config.output_directory)

tess_data_path = os.path.join(_repo_root, "EngVision", "tessdata")
uploads_dir = os.path.join(_base_dir, "..", "..", "uploads")
pipeline_output_dir = os.path.join(_base_dir, "..", "..", "pipeline_runs")
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(pipeline_output_dir, exist_ok=True)

pipeline_service = PipelineService(config, tess_data_path)

# In-memory stores for pipeline runs
pipeline_runs: dict[str, dict[str, Any]] = {}
pipeline_progress: dict[str, str] = {}

sample_dir = os.path.join(_repo_root, "sample_docs")

# ── FastAPI app ──────────────────────────────────────────────────────────────────
app = FastAPI(title="EngVision API (Python)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sanitize_name(name: str) -> str:
    stem = Path(name).stem
    return stem.replace(" ", "_").replace(".", "_")


def _compute_iou(a: dict, b: dict) -> float:
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["width"], b["x"] + b["width"])
    y2 = min(a["y"] + a["height"], b["y"] + b["height"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union > 0 else 0.0


# ── PDF listing ──────────────────────────────────────────────────────────────────
@app.get("/api/pdfs")
def list_pdfs():
    if not os.path.isdir(sample_dir):
        return []
    return [f for f in os.listdir(sample_dir) if f.lower().endswith(".pdf")]


# ── Render page as image ────────────────────────────────────────────────────────
@app.get("/api/pdfs/{filename}/pages/{page_num}/image")
def get_page_image(filename: str, page_num: int):
    pdf_path = os.path.join(sample_dir, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "PDF not found")

    output_dir = os.path.join(config.output_directory, _sanitize_name(filename))
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f"page_{page_num}.png")

    if not os.path.exists(img_path):
        mat = renderer.render_page(pdf_path, page_num - 1)
        PdfRendererService.save_image(mat, img_path)

    return FileResponse(img_path, media_type="image/png")


# ── Page count ───────────────────────────────────────────────────────────────────
@app.get("/api/pdfs/{filename}/info")
def get_pdf_info(filename: str):
    pdf_path = os.path.join(sample_dir, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "PDF not found")

    page_count = renderer.get_page_count(pdf_path)
    mat = renderer.render_page(pdf_path, 0)
    return {"pageCount": page_count, "width": mat.shape[1], "height": mat.shape[0]}


# ── Auto-detect bubbles ─────────────────────────────────────────────────────────
@app.post("/api/pdfs/{filename}/pages/{page_num}/detect")
def detect_regions(filename: str, page_num: int):
    pdf_path = os.path.join(sample_dir, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "PDF not found")

    mat = renderer.render_page(pdf_path, page_num - 1)

    if page_num == 1:
        regions = bubble_detector.detect_bubbles(mat, page_num)
    else:
        regions = table_detector.detect_tables(mat, page_num)
        if not regions:
            regions = [table_detector.get_full_page_region(mat, page_num)]

    doc_key = _sanitize_name(filename)
    annotation_store.set_auto_detections(doc_key, page_num, regions)
    return regions


# ── Get annotations (manual + auto) ─────────────────────────────────────────────
@app.get("/api/pdfs/{filename}/pages/{page_num}/annotations")
def get_annotations(filename: str, page_num: int):
    doc_key = _sanitize_name(filename)
    manual = annotation_store.get_manual_annotations(doc_key, page_num)
    auto = annotation_store.get_auto_detections(doc_key, page_num)
    return {"manual": manual, "auto": auto}


# ── Save manual annotation ──────────────────────────────────────────────────────
@app.post("/api/pdfs/{filename}/pages/{page_num}/annotations")
async def save_annotation(filename: str, page_num: int, request: Request):
    doc_key = _sanitize_name(filename)
    annotation = await request.json()
    if "id" not in annotation or not annotation["id"]:
        annotation["id"] = uuid.uuid4().hex[:8]
    annotation_store.add_manual_annotation(doc_key, page_num, annotation)
    return annotation


# ── Update manual annotation ────────────────────────────────────────────────────
@app.put("/api/pdfs/{filename}/pages/{page_num}/annotations/{ann_id}")
async def update_annotation(filename: str, page_num: int, ann_id: str, request: Request):
    doc_key = _sanitize_name(filename)
    annotation = await request.json()
    annotation_store.update_manual_annotation(doc_key, page_num, ann_id, annotation)
    return annotation


# ── Clear all manual annotations for a page ─────────────────────────────────────
@app.delete("/api/pdfs/{filename}/pages/{page_num}/annotations")
def clear_annotations(filename: str, page_num: int):
    doc_key = _sanitize_name(filename)
    annotation_store.clear_manual_annotations(doc_key, page_num)
    return Response(status_code=204)


# ── Delete manual annotation ────────────────────────────────────────────────────
@app.delete("/api/pdfs/{filename}/pages/{page_num}/annotations/{ann_id}")
def delete_annotation(filename: str, page_num: int, ann_id: str):
    doc_key = _sanitize_name(filename)
    annotation_store.delete_manual_annotation(doc_key, page_num, ann_id)
    return Response(status_code=204)


# ── Export all annotations as ground truth JSON ─────────────────────────────────
@app.get("/api/pdfs/{filename}/annotations/export")
def export_annotations(filename: str):
    doc_key = _sanitize_name(filename)
    all_annotations = annotation_store.export_all(doc_key)
    output_dir = os.path.join(config.output_directory, doc_key)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ground_truth.json")
    with open(path, "w") as f:
        json.dump(all_annotations, f, indent=2)
    return all_annotations


# ── Compute detection accuracy vs ground truth ──────────────────────────────────
@app.get("/api/pdfs/{filename}/pages/{page_num}/accuracy")
def get_accuracy(filename: str, page_num: int):
    doc_key = _sanitize_name(filename)
    manual = annotation_store.get_manual_annotations(doc_key, page_num)
    auto = annotation_store.get_auto_detections(doc_key, page_num)

    if not manual:
        return {"message": "No manual annotations to compare against"}

    matched = 0
    missed = 0
    used_auto: set[int] = set()

    for gt in manual:
        gt_bb = gt.get("boundingBox", gt)
        best_iou = 0.0
        best_idx = -1
        for i, det in enumerate(auto):
            if i in used_auto:
                continue
            det_bb = det.get("boundingBox", det)
            iou = _compute_iou(gt_bb, det_bb)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= 0.3 and best_idx >= 0:
            matched += 1
            used_auto.add(best_idx)
        else:
            missed += 1

    false_positives = len(auto) - len(used_auto)
    precision = matched / len(auto) if auto else 0.0
    recall = matched / len(manual) if manual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "groundTruth": len(manual),
        "detected": len(auto),
        "matched": matched,
        "missed": missed,
        "falsePositives": false_positives,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


# ── Pipeline: Upload and run ────────────────────────────────────────────────────
@app.post("/api/pipeline/run")
async def run_pipeline(pdf: UploadFile = File(...)):
    if not pdf.filename:
        raise HTTPException(400, "No PDF file provided")

    run_id = uuid.uuid4().hex[:8]
    pdf_path = os.path.join(uploads_dir, f"{run_id}_{pdf.filename}")
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    run_output_dir = os.path.join(pipeline_output_dir, run_id)
    pipeline_progress[run_id] = "Starting..."

    result = await pipeline_service.run_async(
        pdf_path, run_id, run_output_dir,
        on_progress=lambda msg: pipeline_progress.__setitem__(run_id, msg),
    )

    pipeline_runs[run_id] = result
    pipeline_progress.pop(run_id, None)
    return result


# ── Pipeline: Run on existing sample PDF ────────────────────────────────────────
@app.post("/api/pipeline/run-sample/{filename}")
async def run_pipeline_sample(filename: str):
    pdf_path = os.path.join(sample_dir, filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(404, "PDF not found")

    run_id = uuid.uuid4().hex[:8]
    run_output_dir = os.path.join(pipeline_output_dir, run_id)
    pipeline_progress[run_id] = "Starting..."

    result = await pipeline_service.run_async(
        pdf_path, run_id, run_output_dir,
        on_progress=lambda msg: pipeline_progress.__setitem__(run_id, msg),
    )

    pipeline_runs[run_id] = result
    pipeline_progress.pop(run_id, None)
    return result


# ── Pipeline: Get run result ────────────────────────────────────────────────────
@app.get("/api/pipeline/{run_id}/results")
def get_pipeline_results(run_id: str):
    if run_id in pipeline_runs:
        return pipeline_runs[run_id]
    if run_id in pipeline_progress:
        return {"status": "running", "progress": pipeline_progress[run_id]}
    raise HTTPException(404)


# ── Pipeline: Get page image ────────────────────────────────────────────────────
@app.get("/api/pipeline/{run_id}/pages/{page_num}/image")
def get_pipeline_page_image(run_id: str, page_num: int):
    img_path = os.path.join(pipeline_output_dir, run_id, "pages", f"page_{page_num}.png")
    if not os.path.exists(img_path):
        raise HTTPException(404)
    return FileResponse(img_path, media_type="image/png")


# ── Pipeline: Get overlay image ─────────────────────────────────────────────────
@app.get("/api/pipeline/{run_id}/pages/{page_num}/overlay")
def get_pipeline_overlay(run_id: str, page_num: int):
    img_path = os.path.join(pipeline_output_dir, run_id, "overlays", f"page_{page_num}_overlay.png")
    if not os.path.exists(img_path):
        raise HTTPException(404)
    return FileResponse(img_path, media_type="image/png")


# ── Pipeline: List all runs ─────────────────────────────────────────────────────
@app.get("/api/pipeline/runs")
def list_pipeline_runs():
    runs = sorted(pipeline_runs.values(), key=lambda r: r.get("runId", ""), reverse=True)
    return [
        {
            "runId": r.get("runId"),
            "pdfFilename": r.get("pdfFilename"),
            "totalBubbles": r.get("totalBubbles"),
            "matchedBubbles": r.get("matchedBubbles"),
            "unmatchedBubbles": r.get("unmatchedBubbles"),
            "status": r.get("status"),
        }
        for r in runs
    ]


# ── Entrypoint ──────────────────────────────────────────────────────────────────
def main():
    port = int(os.environ.get("PORT", "5062"))
    uvicorn.run(
        "engvision.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
