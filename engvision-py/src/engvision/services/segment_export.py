"""Region cropping and debug visualization, ported from .NET SegmentExportService."""

from __future__ import annotations

import os
import shutil

import cv2
import numpy as np

from ..models import RegionType


class SegmentExportService:
    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "bubbles"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pages"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    def export_regions(self, regions: list[dict], page_image: np.ndarray) -> list[dict]:
        exported = []
        img_h, img_w = page_image.shape[:2]

        for region in regions:
            bb = region["boundingBox"]
            x = max(0, bb["x"])
            y = max(0, bb["y"])
            w = min(bb["width"], img_w - x)
            h = min(bb["height"], img_h - y)
            if w <= 0 or h <= 0:
                continue

            cropped = page_image[y:y + h, x:x + w]
            region_type = region.get("type", 0)
            if region_type in (RegionType.BUBBLE.value, RegionType.BUBBLE_WITH_FIGURE.value):
                sub_dir = "bubbles"
            elif region_type == RegionType.TABLE_REGION.value:
                sub_dir = "tables"
            else:
                sub_dir = "pages"

            label = region.get("label") or f"region_{region['id']}"
            filename = f"{label}_p{region.get('pageNumber', 0)}.png"
            output_path = os.path.join(self._output_dir, sub_dir, filename)
            cv2.imwrite(output_path, cropped)

            exported.append({**region, "croppedImagePath": output_path})
        return exported

    def save_debug_visualization(
        self, regions: list[dict], page_image: np.ndarray, page_number: int, suffix: str = ""
    ) -> str:
        debug = page_image.copy()
        color_map = {
            RegionType.BUBBLE.value: (0, 0, 255),
            RegionType.BUBBLE_WITH_FIGURE.value: (0, 255, 0),
            RegionType.TABLE_REGION.value: (255, 0, 0),
        }

        for region in regions:
            bb = region["boundingBox"]
            color = color_map.get(region.get("type", 3), (255, 255, 0))
            cv2.rectangle(debug, (bb["x"], bb["y"]), (bb["x"] + bb["width"], bb["y"] + bb["height"]), color, 3)
            label = region.get("label") or f"#{region['id']}"
            cv2.putText(debug, label, (bb["x"], bb["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        path = os.path.join(self._output_dir, "debug", f"page_{page_number}_detections{suffix}.png")
        cv2.imwrite(path, debug)
        print(f"  Debug visualization saved: {path}")
        return path

    def save_full_page(self, page_image: np.ndarray, page_number: int) -> str:
        path = os.path.join(self._output_dir, "pages", f"page_{page_number}.png")
        cv2.imwrite(path, page_image)
        return path
