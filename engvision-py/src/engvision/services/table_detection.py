"""Table detection service using OpenCV, ported from .NET TableDetectionService."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EngVisionConfig
from ..models import RegionType


class TableDetectionService:
    def __init__(self, config: EngVisionConfig) -> None:
        self._config = config

    def detect_tables(self, page_image: np.ndarray, page_number: int) -> list[dict]:
        """Detect table regions on the given page image."""
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )

        horizontal = self._detect_lines(binary, is_horizontal=True)
        vertical = self._detect_lines(binary, is_horizontal=False)

        table_mask = cv2.add(horizontal, vertical)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        table_mask = cv2.dilate(table_mask, kernel, iterations=3)

        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        region_id = 1
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < self._config.table_min_width or h < self._config.table_min_height:
                continue
            aspect = w / h if h > 0 else 0
            if aspect > 20 or aspect < 0.05:
                continue
            regions.append({
                "id": region_id,
                "pageNumber": page_number,
                "type": RegionType.TABLE_REGION.value,
                "boundingBox": {"x": x, "y": y, "width": w, "height": h},
                "bubbleNumber": None,
                "label": f"Table_P{page_number}_{region_id}",
                "croppedImagePath": None,
            })
            region_id += 1

        print(f"  Page {page_number}: detected {len(regions)} table region(s)")
        return regions

    def get_full_page_region(self, page_image: np.ndarray, page_number: int) -> dict:
        h, w = page_image.shape[:2]
        return {
            "id": 1,
            "pageNumber": page_number,
            "type": RegionType.FULL_PAGE.value,
            "boundingBox": {"x": 0, "y": 0, "width": w, "height": h},
            "bubbleNumber": None,
            "label": f"FullPage_{page_number}",
            "croppedImagePath": None,
        }

    def _detect_lines(self, binary: np.ndarray, is_horizontal: bool) -> np.ndarray:
        if is_horizontal:
            width = max(binary.shape[1] // 30, 10)
            kernel_size = (width, 1)
        else:
            height = max(binary.shape[0] // 30, 10)
            kernel_size = (1, height)
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_kernel)
