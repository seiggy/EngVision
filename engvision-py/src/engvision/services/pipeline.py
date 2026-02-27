"""Full analysis pipeline, ported from .NET PipelineService."""

from __future__ import annotations

import datetime
import json
import os
import time
import traceback
from typing import Any, AsyncGenerator, Callable, Optional

import cv2
import numpy as np

from ..config import EngVisionConfig
from .bubble_detection import BubbleDetectionService
from .bubble_ocr import BubbleOcrService
from .dimension_matcher import are_similar, confidence_score
from .leader_line_tracer import LeaderLineTracerService, CAPTURE_STEPS
from .pdf_renderer import PdfRendererService
from .table_ocr import TableOcrService


class PipelineService:
    def __init__(self, config: EngVisionConfig, tess_data_path: str) -> None:
        self._config = config
        self._tess_data_path = tess_data_path

    def _create_ocr_services(self) -> tuple:
        """Return (bubble_ocr, table_ocr) based on OCR_PROVIDER config.

        When "Azure", uses Azure Document Intelligence services.
        Otherwise falls back to local Tesseract.
        """
        if self._config.ocr_provider.lower() == "azure":
            ep = self._config.azure_docint_endpoint
            key = self._config.azure_docint_key
            if not ep or not key:
                print("  WARNING: OCR_PROVIDER=Azure but AZURE_DOCINT_ENDPOINT/KEY not set — falling back to Tesseract")
                return BubbleOcrService(self._tess_data_path), TableOcrService(self._tess_data_path)
            from .azure_bubble_ocr import AzureBubbleOcrService
            from .azure_table_ocr import AzureTableOcrService
            print(f"  Using Azure Document Intelligence for OCR ({ep})")
            return AzureBubbleOcrService(ep, key), AzureTableOcrService(ep, key)
        return BubbleOcrService(self._tess_data_path), TableOcrService(self._tess_data_path)

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
            ocr_service, table_ocr = self._create_ocr_services()
            ocr_results = ocr_service.extract_all(raw_crops_dir)

            # Step 3: Table OCR (pages 2+)
            progress("OCR-ing table data...")
            tesseract_dimensions: dict[int, str] = {}

            # Azure Doc Intelligence supports full-PDF mode for efficient table extraction
            if hasattr(table_ocr, "extract_balloon_dimensions_from_pdf") and pdf_path:
                with open(pdf_path, "rb") as f:
                    tesseract_dimensions = table_ocr.extract_balloon_dimensions_from_pdf(f.read())
            else:
                for i in range(1, len(page_images)):
                    page_dims = table_ocr.extract_balloon_dimensions(page_images[i])
                    for num, dim in page_dims.items():
                        tesseract_dimensions.setdefault(num, dim)
            ocr_ms = int((time.time() - step_start) * 1000)

            # Step 4: Trace leader lines from each bubble
            progress("Tracing leader lines...")
            step_start = time.time()
            tracer = LeaderLineTracerService()
            expanded_bubbles = tracer.trace_and_expand(bubbles, page_images[0])
            trace_ms = int((time.time() - step_start) * 1000)

            # Debug: dump every capture box crop + annotated overview
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_overlay = page_images[0].copy()
            for i, eb in enumerate(expanded_bubbles):
                bb = eb["boundingBox"]
                bcx = bb["x"] + bb["width"] // 2
                bcy = bb["y"] + bb["height"] // 2
                b_radius = bb["width"] // 2
                bnum = eb.get("bubbleNumber", i + 1)

                # Draw bubble circle on debug overlay
                cv2.circle(debug_overlay, (bcx, bcy), b_radius, (0, 255, 0), 1)
                cv2.putText(debug_overlay, str(bnum),
                            (bcx - 8, bcy - b_radius - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                cap = eb.get("captureBox")
                if cap is None:
                    # No leader direction found — note it
                    cv2.putText(debug_overlay, "NO_DIR",
                                (bcx + b_radius + 4, bcy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                    continue

                cx1 = max(0, cap["x"])
                cy1 = max(0, cap["y"])
                cx2 = min(img_w, cap["x"] + cap["width"])
                cy2 = min(img_h, cap["y"] + cap["height"])

                # Draw capture box rectangle on debug overlay
                cv2.rectangle(debug_overlay, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)
                # Draw line from bubble centre to capture box centre
                cap_cx = (cx1 + cx2) // 2
                cap_cy = (cy1 + cy2) // 2
                cv2.arrowedLine(debug_overlay, (bcx, bcy), (cap_cx, cap_cy),
                                (255, 0, 255), 1, tipLength=0.15)

                # Save individual capture box crop
                if cx2 - cx1 > 0 and cy2 - cy1 > 0:
                    crop = page_images[0][cy1:cy2, cx1:cx2]
                    cv2.imwrite(os.path.join(debug_dir,
                                f"capture_bubble_{bnum:03d}.png"), crop)

            # Save annotated overview showing all capture boxes
            cv2.imwrite(os.path.join(debug_dir, "capture_boxes_overview.png"),
                        debug_overlay)
            progress(f"Debug: saved {len(expanded_bubbles)} capture box crops to {debug_dir}")

            # Step 5: Vision LLM validation with progressive capture expansion
            # For each bubble with a table dimension, try progressively larger
            # capture boxes along the leader line until the LLM confirms a match.
            # Sizes: 128×128 → 256×128 → 512×256 → 1024×512
            llm_validations: dict[int, dict] = {}
            llm_input_tokens = 0
            llm_output_tokens = 0
            llm_total_tokens = 0
            llm_calls = 0

            step_start = time.time()
            endpoint = os.environ.get("AZURE_ENDPOINT", "")
            key = os.environ.get("AZURE_KEY", "")
            model = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-5.3-codex")

            if key and endpoint:
                progress("Validating dimensions with Vision LLM...")
                from openai import AzureOpenAI
                from .vision_llm import VisionLlmService

                client = AzureOpenAI(
                    api_key=key,
                    azure_endpoint=endpoint,
                    api_version="2024-12-01-preview",
                )
                vision_service = VisionLlmService(client, model)

                # Build a lookup from bubble index → expanded result
                expanded_by_idx: dict[int, dict] = {}
                for i, eb in enumerate(expanded_bubbles):
                    expanded_by_idx[i] = eb

                for file_name, number in ocr_results.items():
                    if number is None:
                        continue
                    crop_idx = int(file_name.replace("bubble_", "").replace(".png", "")) - 1
                    if crop_idx < 0 or crop_idx >= len(bubbles):
                        continue

                    table_dim = tesseract_dimensions.get(number)

                    eb = expanded_by_idx.get(crop_idx)
                    if eb is None or eb.get("leaderDirection") is None:
                        continue
                    ld = eb["leaderDirection"]
                    dx, dy = ld["dx"], ld["dy"]

                    # Use original bubble geometry for capture box placement
                    orig_bb = bubbles[crop_idx]["boundingBox"]
                    bcx = orig_bb["x"] + orig_bb["width"] // 2
                    bcy = orig_bb["y"] + orig_bb["height"] // 2
                    b_radius = orig_bb["width"] // 2

                    if table_dim:
                        # Progressive expansion: try each size, stop on match
                        last_validation = None
                        final_capture_size = None
                        for cap_w, cap_h in CAPTURE_STEPS:
                            cap = tracer.place_capture_box(
                                bcx, bcy, b_radius, dx, dy,
                                cap_w, cap_h, img_w, img_h,
                            )
                            x1 = max(0, cap["x"])
                            y1 = max(0, cap["y"])
                            x2 = min(img_w, cap["x"] + cap["width"])
                            y2 = min(img_h, cap["y"] + cap["height"])
                            if x2 - x1 < 4 or y2 - y1 < 4:
                                continue

                            crop = page_images[0][y1:y2, x1:x2]
                            _, crop_bytes = cv2.imencode(".png", crop)

                            # Save debug capture at this step
                            cv2.imwrite(os.path.join(
                                debug_dir,
                                f"capture_bubble_{number:03d}_{cap_w}x{cap_h}.png",
                            ), crop)

                            validation = await vision_service.validate_dimension(
                                crop_bytes.tobytes(), number, table_dim
                            )
                            last_validation = validation
                            final_capture_size = f"{cap_w}x{cap_h}"
                            llm_input_tokens += validation.input_tokens
                            llm_output_tokens += validation.output_tokens
                            llm_total_tokens += validation.total_tokens
                            llm_calls += 1

                            if validation.matches:
                                progress(f"  Bubble {number}: matched at {cap_w}x{cap_h}")
                                break
                            else:
                                progress(
                                    f"  Bubble {number}: no match at {cap_w}x{cap_h}"
                                    f" (saw '{validation.observed_dimension}'), expanding..."
                                )

                        # Use the last validation result (match or best guess at max size)
                        if last_validation is not None:
                            llm_validations[number] = {
                                "observedDimension": last_validation.observed_dimension,
                                "matches": last_validation.matches,
                                "confidence": last_validation.confidence,
                                "notes": last_validation.notes,
                                "captureSize": final_capture_size,
                            }
                    else:
                        # Discovery mode: table OCR missed this entry
                        cap_w, cap_h = CAPTURE_STEPS[0]
                        cap = tracer.place_capture_box(
                            bcx, bcy, b_radius, dx, dy,
                            cap_w, cap_h, img_w, img_h,
                        )
                        x1 = max(0, cap["x"])
                        y1 = max(0, cap["y"])
                        x2 = min(img_w, cap["x"] + cap["width"])
                        y2 = min(img_h, cap["y"] + cap["height"])
                        if x2 - x1 >= 4 and y2 - y1 >= 4:
                            crop = page_images[0][y1:y2, x1:x2]
                            _, crop_bytes = cv2.imencode(".png", crop)

                            cv2.imwrite(os.path.join(
                                debug_dir,
                                f"capture_bubble_{number:03d}_{cap_w}x{cap_h}.png",
                            ), crop)

                            discovery = await vision_service.discover_dimension(
                                crop_bytes.tobytes(), number
                            )
                            llm_input_tokens += discovery.input_tokens
                            llm_output_tokens += discovery.output_tokens
                            llm_total_tokens += discovery.total_tokens
                            llm_calls += 1
                            progress(f"  Bubble {number}: discovered '{discovery.observed_dimension}' (no table entry)")

                            llm_validations[number] = {
                                "observedDimension": discovery.observed_dimension,
                                "matches": discovery.matches,
                                "confidence": discovery.confidence,
                                "notes": f"[Table OCR miss] {discovery.notes}",
                                "captureSize": f"{cap_w}x{cap_h}",
                            }
            llm_ms = int((time.time() - step_start) * 1000)

            # Step 6: Merge results — table OCR + LLM validation
            progress("Merging OCR + LLM validation results...")
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
                validation = llm_validations.get(number)

                # The LLM validation tells us if the table dimension matches the drawing
                llm_matches = validation["matches"] if validation else None
                llm_observed = validation["observedDimension"] if validation else None
                llm_confidence = validation["confidence"] if validation else 0.0
                llm_notes = validation["notes"] if validation else None
                capture_size = validation["captureSize"] if validation else None

                # Determine conflict: table says one thing, drawing shows another
                has_conflict = validation is not None and not validation["matches"]

                # If LLM confirmed match, trust its confidence score directly.
                # Only use fuzzy string matching when there's a conflict (to quantify
                # how different the values are).
                if validation and llm_matches:
                    conf = llm_confidence
                elif tess_val and llm_observed:
                    conf = confidence_score(tess_val, llm_observed)
                elif validation:
                    conf = llm_confidence
                else:
                    conf = 0.0

                if tess_val is not None and validation is not None:
                    source = "Table+Validated"
                elif tess_val is not None:
                    source = "TableOnly"
                elif validation is not None:
                    source = "LLMOnly"
                else:
                    source = "None"

                dimension_map[str(number)] = {
                    "balloonNo": number,
                    "dimension": tess_val or llm_observed,
                    "source": source,
                    "tesseractValue": tess_val,
                    "llmObservedValue": llm_observed,
                    "llmMatches": llm_matches,
                    "llmConfidence": round(llm_confidence, 4),
                    "llmNotes": llm_notes,
                    "hasConflict": has_conflict,
                    "confidence": round(conf, 4),
                    "captureSize": capture_size,
                }
            merge_ms = int((time.time() - step_start) * 1000)

            # Step 7: Generate overlay image
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

            # Write benchmark.json
            _write_benchmark(output_dir, run_id, filename, [
                {"name": "render", "durationMs": render_ms},
                {"name": "detect", "durationMs": detect_ms},
                {"name": "ocr", "durationMs": ocr_ms},
                {"name": "trace", "durationMs": trace_ms},
                {"name": "validate", "durationMs": llm_ms},
                {"name": "merge", "durationMs": merge_ms},
                {"name": "overlay", "durationMs": 0},
            ], total_ms, len(bubble_results), matched, llm_calls,
                llm_input_tokens, llm_output_tokens, llm_total_tokens)

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
                    "traceDurationMs": trace_ms,
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

    async def run_stream(
        self,
        pdf_path: str,
        run_id: str,
        output_dir: str,
    ) -> AsyncGenerator[dict, None]:
        """Async generator that yields structured SSE event dicts."""
        os.makedirs(output_dir, exist_ok=True)
        pages_dir = os.path.join(output_dir, "pages")
        overlay_dir = os.path.join(output_dir, "overlays")
        os.makedirs(pages_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        filename = os.path.basename(pdf_path)

        try:
            total_start = time.time()

            # Step 1: Render pages
            yield {"type": "step", "step": 1, "totalSteps": 7, "name": "render", "message": "Rendering PDF pages..."}
            step_start = time.time()
            renderer = PdfRendererService(self._config.pdf_render_dpi)
            page_images = renderer.render_all_pages(pdf_path)
            render_ms = int((time.time() - step_start) * 1000)

            for i, img in enumerate(page_images):
                PdfRendererService.save_image(img, os.path.join(pages_dir, f"page_{i + 1}.png"))

            page_count = len(page_images)
            img_h, img_w = page_images[0].shape[:2]
            yield {"type": "stepComplete", "step": 1, "name": "render", "durationMs": render_ms,
                   "detail": {"pageCount": page_count, "imageSize": f"{img_w}x{img_h}"}}

            # Step 2: Detect bubbles on page 1
            yield {"type": "step", "step": 2, "totalSteps": 7, "name": "detect", "message": "Detecting bubbles and OCR-ing bubble numbers..."}
            step_start = time.time()
            bubble_detector = BubbleDetectionService(self._config)
            bubbles = bubble_detector.detect_bubbles(page_images[0], page_number=1)

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

            ocr_service, table_ocr = self._create_ocr_services()
            ocr_results = ocr_service.extract_all(raw_crops_dir)
            detect_ms = int((time.time() - step_start) * 1000)
            yield {"type": "stepComplete", "step": 2, "name": "detect", "durationMs": detect_ms,
                   "detail": {"bubbleCount": len(bubbles)}}

            # Step 3: Table OCR (pages 2+)
            yield {"type": "step", "step": 3, "totalSteps": 7, "name": "ocr", "message": "OCR-ing table data..."}
            step_start = time.time()
            tesseract_dimensions: dict[int, str] = {}

            # Azure Doc Intelligence supports full-PDF mode for efficient table extraction
            if hasattr(table_ocr, "extract_balloon_dimensions_from_pdf") and pdf_path:
                with open(pdf_path, "rb") as f:
                    tesseract_dimensions = table_ocr.extract_balloon_dimensions_from_pdf(f.read())
            else:
                for i in range(1, len(page_images)):
                    page_dims = table_ocr.extract_balloon_dimensions(page_images[i])
                    for num, dim in page_dims.items():
                        tesseract_dimensions.setdefault(num, dim)
            ocr_ms = int((time.time() - step_start) * 1000)
            yield {"type": "stepComplete", "step": 3, "name": "ocr", "durationMs": ocr_ms,
                   "detail": {"dimensionCount": len(tesseract_dimensions)}}

            # Step 4: Trace leader lines from each bubble
            yield {"type": "step", "step": 4, "totalSteps": 7, "name": "trace", "message": "Tracing leader lines..."}
            step_start = time.time()
            tracer = LeaderLineTracerService()
            expanded_bubbles = tracer.trace_and_expand(bubbles, page_images[0])
            trace_ms = int((time.time() - step_start) * 1000)
            yield {"type": "stepComplete", "step": 4, "name": "trace", "durationMs": trace_ms,
                   "detail": {"tracedCount": len(expanded_bubbles)}}

            # Debug: dump every capture box crop + annotated overview
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_overlay = page_images[0].copy()
            for i, eb in enumerate(expanded_bubbles):
                bb = eb["boundingBox"]
                bcx = bb["x"] + bb["width"] // 2
                bcy = bb["y"] + bb["height"] // 2
                b_radius = bb["width"] // 2
                bnum = eb.get("bubbleNumber", i + 1)

                cv2.circle(debug_overlay, (bcx, bcy), b_radius, (0, 255, 0), 1)
                cv2.putText(debug_overlay, str(bnum),
                            (bcx - 8, bcy - b_radius - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                cap = eb.get("captureBox")
                if cap is None:
                    cv2.putText(debug_overlay, "NO_DIR",
                                (bcx + b_radius + 4, bcy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                    continue

                cx1 = max(0, cap["x"])
                cy1 = max(0, cap["y"])
                cx2 = min(img_w, cap["x"] + cap["width"])
                cy2 = min(img_h, cap["y"] + cap["height"])

                cv2.rectangle(debug_overlay, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)
                cap_cx = (cx1 + cx2) // 2
                cap_cy = (cy1 + cy2) // 2
                cv2.arrowedLine(debug_overlay, (bcx, bcy), (cap_cx, cap_cy),
                                (255, 0, 255), 1, tipLength=0.15)

                if cx2 - cx1 > 0 and cy2 - cy1 > 0:
                    crop = page_images[0][cy1:cy2, cx1:cx2]
                    cv2.imwrite(os.path.join(debug_dir,
                                f"capture_bubble_{bnum:03d}.png"), crop)

            cv2.imwrite(os.path.join(debug_dir, "capture_boxes_overview.png"),
                        debug_overlay)

            # Step 5: Vision LLM validation with progressive capture expansion
            yield {"type": "step", "step": 5, "totalSteps": 7, "name": "validate", "message": "Validating dimensions with Vision LLM..."}
            llm_validations: dict[int, dict] = {}
            llm_input_tokens = 0
            llm_output_tokens = 0
            llm_total_tokens = 0
            llm_calls = 0

            step_start = time.time()
            endpoint = os.environ.get("AZURE_ENDPOINT", "")
            key = os.environ.get("AZURE_KEY", "")
            model = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-5.3-codex")

            if key and endpoint:
                from openai import AzureOpenAI
                from .vision_llm import VisionLlmService

                client = AzureOpenAI(
                    api_key=key,
                    azure_endpoint=endpoint,
                    api_version="2024-12-01-preview",
                )
                vision_service = VisionLlmService(client, model)

                expanded_by_idx: dict[int, dict] = {}
                for i, eb in enumerate(expanded_bubbles):
                    expanded_by_idx[i] = eb

                for file_name, number in ocr_results.items():
                    if number is None:
                        continue
                    crop_idx = int(file_name.replace("bubble_", "").replace(".png", "")) - 1
                    if crop_idx < 0 or crop_idx >= len(bubbles):
                        continue

                    table_dim = tesseract_dimensions.get(number)

                    eb = expanded_by_idx.get(crop_idx)
                    if eb is None or eb.get("leaderDirection") is None:
                        continue
                    ld = eb["leaderDirection"]
                    dx, dy = ld["dx"], ld["dy"]

                    orig_bb = bubbles[crop_idx]["boundingBox"]
                    bcx = orig_bb["x"] + orig_bb["width"] // 2
                    bcy = orig_bb["y"] + orig_bb["height"] // 2
                    b_radius = orig_bb["width"] // 2

                    if table_dim:
                        # Normal validation: compare drawing crop against table value
                        last_validation = None
                        final_capture_size = None
                        for cap_w, cap_h in CAPTURE_STEPS:
                            cap = tracer.place_capture_box(
                                bcx, bcy, b_radius, dx, dy,
                                cap_w, cap_h, img_w, img_h,
                            )
                            x1 = max(0, cap["x"])
                            y1 = max(0, cap["y"])
                            x2 = min(img_w, cap["x"] + cap["width"])
                            y2 = min(img_h, cap["y"] + cap["height"])
                            if x2 - x1 < 4 or y2 - y1 < 4:
                                continue

                            crop = page_images[0][y1:y2, x1:x2]
                            _, crop_bytes = cv2.imencode(".png", crop)

                            cv2.imwrite(os.path.join(
                                debug_dir,
                                f"capture_bubble_{number:03d}_{cap_w}x{cap_h}.png",
                            ), crop)

                            validation = await vision_service.validate_dimension(
                                crop_bytes.tobytes(), number, table_dim
                            )
                            last_validation = validation
                            final_capture_size = f"{cap_w}x{cap_h}"
                            llm_input_tokens += validation.input_tokens
                            llm_output_tokens += validation.output_tokens
                            llm_total_tokens += validation.total_tokens
                            llm_calls += 1

                            if validation.matches:
                                yield {
                                    "type": "bubble", "bubbleNumber": number,
                                    "captureSize": final_capture_size, "status": "match",
                                    "tableDim": table_dim,
                                    "observed": validation.observed_dimension,
                                    "confidence": validation.confidence,
                                }
                                break
                            else:
                                is_last = (cap_w, cap_h) == CAPTURE_STEPS[-1]
                                yield {
                                    "type": "bubble", "bubbleNumber": number,
                                    "captureSize": final_capture_size,
                                    "status": "bestGuess" if is_last else "expanding",
                                    "tableDim": table_dim,
                                    "observed": validation.observed_dimension,
                                    "confidence": validation.confidence,
                                }

                        if last_validation is not None:
                            llm_validations[number] = {
                                "observedDimension": last_validation.observed_dimension,
                                "matches": last_validation.matches,
                                "confidence": last_validation.confidence,
                                "notes": last_validation.notes,
                                "captureSize": final_capture_size,
                            }
                    else:
                        # Discovery mode: table OCR missed this entry — ask LLM
                        # what dimension it sees in the initial capture crop.
                        cap_w, cap_h = CAPTURE_STEPS[0]
                        cap = tracer.place_capture_box(
                            bcx, bcy, b_radius, dx, dy,
                            cap_w, cap_h, img_w, img_h,
                        )
                        x1 = max(0, cap["x"])
                        y1 = max(0, cap["y"])
                        x2 = min(img_w, cap["x"] + cap["width"])
                        y2 = min(img_h, cap["y"] + cap["height"])
                        if x2 - x1 >= 4 and y2 - y1 >= 4:
                            crop = page_images[0][y1:y2, x1:x2]
                            _, crop_bytes = cv2.imencode(".png", crop)

                            cv2.imwrite(os.path.join(
                                debug_dir,
                                f"capture_bubble_{number:03d}_{cap_w}x{cap_h}.png",
                            ), crop)

                            discovery = await vision_service.discover_dimension(
                                crop_bytes.tobytes(), number
                            )
                            llm_input_tokens += discovery.input_tokens
                            llm_output_tokens += discovery.output_tokens
                            llm_total_tokens += discovery.total_tokens
                            llm_calls += 1

                            yield {
                                "type": "bubble", "bubbleNumber": number,
                                "captureSize": f"{cap_w}x{cap_h}",
                                "status": "discovered",
                                "tableDim": None,
                                "observed": discovery.observed_dimension,
                                "confidence": discovery.confidence,
                            }

                            llm_validations[number] = {
                                "observedDimension": discovery.observed_dimension,
                                "matches": discovery.matches,
                                "confidence": discovery.confidence,
                                "notes": f"[Table OCR miss] {discovery.notes}",
                                "captureSize": f"{cap_w}x{cap_h}",
                            }
            llm_ms = int((time.time() - step_start) * 1000)
            yield {"type": "stepComplete", "step": 5, "name": "validate", "durationMs": llm_ms,
                   "detail": {"llmCalls": llm_calls, "validatedCount": len(llm_validations)}}

            # Step 6: Merge results
            yield {"type": "step", "step": 6, "totalSteps": 7, "name": "merge", "message": "Merging OCR + LLM validation results..."}
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
                validation = llm_validations.get(number)

                llm_matches = validation["matches"] if validation else None
                llm_observed = validation["observedDimension"] if validation else None
                llm_confidence = validation["confidence"] if validation else 0.0
                llm_notes = validation["notes"] if validation else None
                capture_size = validation["captureSize"] if validation else None

                has_conflict = validation is not None and not validation["matches"]

                if validation and llm_matches:
                    conf = llm_confidence
                elif tess_val and llm_observed:
                    conf = confidence_score(tess_val, llm_observed)
                elif validation:
                    conf = llm_confidence
                else:
                    conf = 0.0

                if tess_val is not None and validation is not None:
                    source = "Table+Validated"
                elif tess_val is not None:
                    source = "TableOnly"
                elif validation is not None:
                    source = "LLMOnly"
                else:
                    source = "None"

                dimension_map[str(number)] = {
                    "balloonNo": number,
                    "dimension": tess_val or llm_observed,
                    "source": source,
                    "tesseractValue": tess_val,
                    "llmObservedValue": llm_observed,
                    "llmMatches": llm_matches,
                    "llmConfidence": round(llm_confidence, 4),
                    "llmNotes": llm_notes,
                    "hasConflict": has_conflict,
                    "confidence": round(conf, 4),
                    "captureSize": capture_size,
                }
            merge_ms = int((time.time() - step_start) * 1000)
            yield {"type": "stepComplete", "step": 6, "name": "merge", "durationMs": merge_ms,
                   "detail": {"dimensionCount": len(dimension_map)}}

            # Step 7: Generate overlay image
            yield {"type": "step", "step": 7, "totalSteps": 7, "name": "overlay", "message": "Generating overlay images..."}
            step_start = time.time()
            _generate_overlay(
                page_images[0],
                bubble_results,
                dimension_map,
                os.path.join(overlay_dir, "page_1_overlay.png"),
            )
            overlay_ms = int((time.time() - step_start) * 1000)
            yield {"type": "stepComplete", "step": 7, "name": "overlay", "durationMs": overlay_ms,
                   "detail": {}}

            total_ms = int((time.time() - total_start) * 1000)

            matched = sum(1 for d in dimension_map.values() if d.get("dimension") is not None)
            warnings_count = sum(
                1 for d in dimension_map.values()
                if d.get("confidence", 0) > 0 and d.get("confidence", 0) < 0.8
            )

            # Write benchmark.json
            _write_benchmark(output_dir, run_id, filename, [
                {"name": "render", "durationMs": render_ms},
                {"name": "detect", "durationMs": detect_ms},
                {"name": "ocr", "durationMs": ocr_ms},
                {"name": "trace", "durationMs": trace_ms},
                {"name": "validate", "durationMs": llm_ms},
                {"name": "merge", "durationMs": merge_ms},
                {"name": "overlay", "durationMs": overlay_ms},
            ], total_ms, len(bubble_results), matched, llm_calls,
                llm_input_tokens, llm_output_tokens, llm_total_tokens)

            result = {
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
                    "traceDurationMs": trace_ms,
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

            yield {"type": "complete", "result": result}

        except Exception as ex:
            traceback.print_exc()
            yield {"type": "error", "message": str(ex)}


def _write_benchmark(
    output_dir: str,
    run_id: str,
    pdf_filename: str,
    steps: list[dict],
    total_ms: int,
    bubble_count: int,
    matched_bubbles: int,
    llm_calls: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> None:
    benchmark = {
        "runId": run_id,
        "backend": "python",
        "pdfFilename": pdf_filename,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "steps": steps,
        "totalDurationMs": total_ms,
        "bubbleCount": bubble_count,
        "matchedBubbles": matched_bubbles,
        "llmCalls": llm_calls,
        "tokenUsage": {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        },
    }
    path = os.path.join(output_dir, "benchmark.json")
    with open(path, "w") as f:
        json.dump(benchmark, f, indent=2)


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
