"""Full analysis pipeline, ported from .NET PipelineService."""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Callable, Optional

import cv2
import numpy as np

from ..config import EngVisionConfig
from .bubble_detection import BubbleDetectionService
from .bubble_ocr import BubbleOcrService
from .dimension_matcher import are_similar, confidence_score
from .pdf_renderer import PdfRendererService
from .table_ocr import TableOcrService


class PipelineService:
    def __init__(self, config: EngVisionConfig, tess_data_path: str) -> None:
        self._config = config
        self._tess_data_path = tess_data_path

    async def run_async(
        self,
        pdf_path: str,
        run_id: str,
        output_dir: str,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        def progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        os.makedirs(output_dir, exist_ok=True)
        pages_dir = os.path.join(output_dir, "pages")
        overlay_dir = os.path.join(output_dir, "overlays")
        os.makedirs(pages_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        filename = os.path.basename(pdf_path)

        try:
            total_start = time.time()

            # Step 1: Render pages
            progress("Rendering PDF pages...")
            step_start = time.time()
            renderer = PdfRendererService(self._config.pdf_render_dpi)
            page_images = renderer.render_all_pages(pdf_path)
            render_ms = int((time.time() - step_start) * 1000)

            for i, img in enumerate(page_images):
                PdfRendererService.save_image(img, os.path.join(pages_dir, f"page_{i + 1}.png"))

            page_count = len(page_images)
            img_h, img_w = page_images[0].shape[:2]

            # Step 2: Detect bubbles on page 1
            progress("Detecting bubbles...")
            step_start = time.time()
            bubble_detector = BubbleDetectionService(self._config)
            bubbles = bubble_detector.detect_bubbles(page_images[0], page_number=1)
            detect_ms = int((time.time() - step_start) * 1000)

            # Step 2b: Save raw bubble crops for OCR
            raw_crops_dir = os.path.join(output_dir, "bubble_crops")
            os.makedirs(raw_crops_dir, exist_ok=True)
            for b in bubbles:
                bb = b["boundingBox"]
                cx = bb["x"] + bb["width"] // 2
                cy = bb["y"] + bb["height"] // 2
                r = bb["width"] // 2
                pad = 2
                x1 = max(0, cx - r - pad)
                y1 = max(0, cy - r - pad)
                x2 = min(img_w, cx + r + pad)
                y2 = min(img_h, cy + r + pad)
                crop = page_images[0][y1:y2, x1:x2]
                cv2.imwrite(os.path.join(raw_crops_dir, f"bubble_{b['bubbleNumber']:03d}.png"), crop)

            # Step 2c: OCR bubble numbers
            progress("OCR-ing bubble numbers...")
            step_start = time.time()
            ocr_service = BubbleOcrService(self._tess_data_path)
            ocr_results = ocr_service.extract_all(raw_crops_dir)

            # Step 3: Table OCR (pages 2+)
            progress("OCR-ing table data...")
            table_ocr = TableOcrService(self._tess_data_path)
            tesseract_dimensions: dict[int, str] = {}
            for i in range(1, len(page_images)):
                page_dims = table_ocr.extract_balloon_dimensions(page_images[i])
                for num, dim in page_dims.items():
                    tesseract_dimensions.setdefault(num, dim)
            ocr_ms = int((time.time() - step_start) * 1000)

            # Step 4: Vision LLM extraction (if configured)
            llm_dimensions: dict[int, str] = {}
            llm_input_tokens = 0
            llm_output_tokens = 0
            llm_total_tokens = 0
            llm_calls = 0

            step_start = time.time()
            endpoint = os.environ.get("AZURE_ENDPOINT", "")
            key = os.environ.get("AZURE_KEY", "")
            model = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-5.3-codex")

            if key and endpoint:
                progress("Running Vision LLM table extraction...")
                from openai import AzureOpenAI
                from .vision_llm import VisionLlmService

                client = AzureOpenAI(
                    api_key=key,
                    azure_endpoint=endpoint,
                    api_version="2024-12-01-preview",
                )
                vision_service = VisionLlmService(client, model)

                for i in range(1, len(page_images)):
                    _, page_bytes = cv2.imencode(".png", page_images[i])
                    extraction = await vision_service.extract_balloon_dimensions_with_usage(
                        page_bytes.tobytes(), i + 1
                    )
                    for num, dim in extraction.dimensions.items():
                        llm_dimensions.setdefault(num, dim)
                    llm_input_tokens += extraction.input_tokens
                    llm_output_tokens += extraction.output_tokens
                    llm_total_tokens += extraction.total_tokens
                    llm_calls += 1
            llm_ms = int((time.time() - step_start) * 1000)

            # Step 5: Merge results
            progress("Merging OCR + LLM results...")
            step_start = time.time()
            dimension_map: dict[str, dict] = {}
            bubble_results: list[dict] = []

            for file_name, number in ocr_results.items():
                if number is None:
                    continue
                crop_idx = int(file_name.replace("bubble_", "").replace(".png", "")) - 1
                if crop_idx < 0 or crop_idx >= len(bubbles):
                    continue
                bb = bubbles[crop_idx]["boundingBox"]
                cx = bb["x"] + bb["width"] // 2
                cy = bb["y"] + bb["height"] // 2
                r = bb["width"] // 2

                bubble_results.append({
                    "bubbleNumber": number,
                    "cx": cx,
                    "cy": cy,
                    "radius": r,
                    "boundingBox": bb,
                })

                tess_val = tesseract_dimensions.get(number)
                llm_val = llm_dimensions.get(number)
                dimension = tess_val or llm_val

                if tess_val is not None and llm_val is not None:
                    source = "Both"
                elif tess_val is not None:
                    source = "Tesseract"
                elif llm_val is not None:
                    source = "LLM"
                else:
                    source = "None"

                if tess_val is not None and llm_val is not None:
                    conf = confidence_score(tess_val, llm_val)
                else:
                    conf = 0.0

                has_conflict = (
                    tess_val is not None
                    and llm_val is not None
                    and not are_similar(tess_val, llm_val)
                )

                dimension_map[str(number)] = {
                    "balloonNo": number,
                    "dimension": dimension,
                    "source": source,
                    "tesseractValue": tess_val,
                    "llmValue": llm_val,
                    "hasConflict": has_conflict,
                    "confidence": round(conf, 4),
                }
            merge_ms = int((time.time() - step_start) * 1000)

            # Step 6: Generate overlay image
            progress("Generating overlay images...")
            _generate_overlay(
                page_images[0],
                bubble_results,
                dimension_map,
                os.path.join(overlay_dir, "page_1_overlay.png"),
            )

            total_ms = int((time.time() - total_start) * 1000)

            progress("Complete!")

            matched = sum(1 for d in dimension_map.values() if d.get("dimension") is not None)
            warnings_count = sum(
                1 for d in dimension_map.values()
                if d.get("confidence", 0) > 0 and d.get("confidence", 0) < 0.8
            )

            return {
                "runId": run_id,
                "pdfFilename": filename,
                "pageCount": page_count,
                "imageWidth": img_w,
                "imageHeight": img_h,
                "bubbles": sorted(bubble_results, key=lambda b: b["bubbleNumber"]),
                "dimensionMap": dimension_map,
                "totalBubbles": len(bubble_results),
                "matchedBubbles": matched,
                "unmatchedBubbles": len(bubble_results) - matched,
                "warnings": warnings_count,
                "metrics": {
                    "totalDurationMs": total_ms,
                    "renderDurationMs": render_ms,
                    "detectDurationMs": detect_ms,
                    "ocrDurationMs": ocr_ms,
                    "llmDurationMs": llm_ms,
                    "mergeDurationMs": merge_ms,
                    "peakMemoryMb": 0,
                },
                "tokenUsage": {
                    "inputTokens": llm_input_tokens,
                    "outputTokens": llm_output_tokens,
                    "totalTokens": llm_total_tokens,
                    "llmCalls": llm_calls,
                } if llm_calls > 0 else None,
                "status": "complete",
                "error": None,
            }
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


def _generate_overlay(
    page_image: np.ndarray,
    bubbles: list[dict],
    dimension_map: dict[str, dict],
    output_path: str,
) -> None:
    overlay = page_image.copy()

    for bubble in bubbles:
        num = bubble["bubbleNumber"]
        match = dimension_map.get(str(num))
        has_match = match is not None and match.get("dimension") is not None
        has_conflict = match.get("hasConflict", False) if match else False

        if has_match:
            color = (0, 200, 255) if has_conflict else (0, 200, 0)  # amber or green
        else:
            color = (0, 0, 255)  # red

        cv2.circle(overlay, (bubble["cx"], bubble["cy"]), bubble["radius"] + 3, color, 3)

        label = f"#{num}"
        cv2.putText(
            overlay, label,
            (bubble["cx"] - 12, bubble["cy"] - bubble["radius"] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )

        if match and match.get("dimension"):
            dim_label = match["dimension"]
            if len(dim_label) > 20:
                dim_label = dim_label[:20] + "…"
            cv2.putText(
                overlay, dim_label,
                (bubble["cx"] + bubble["radius"] + 8, bubble["cy"] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
            )

    cv2.imwrite(output_path, overlay)
