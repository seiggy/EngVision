"""Table OCR service using pytesseract, ported from .NET TableOcrService."""

from __future__ import annotations

import os
import re
import shutil

import cv2
import numpy as np
import pytesseract

# Auto-detect Tesseract on Windows if not already on PATH
if not shutil.which("tesseract"):
    _win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(_win_default):
        pytesseract.pytesseract.tesseract_cmd = _win_default


class TableOcrService:
    def __init__(self, tess_data_path: str) -> None:
        self._tess_data_path = tess_data_path

    def extract_balloon_dimensions(self, page_image: np.ndarray) -> dict[int, str]:
        row_bounds, col_bounds = self._detect_grid(page_image)
        if len(row_bounds) < 3 or len(col_bounds) < 3:
            print("    Grid detection failed, falling back to full-page OCR")
            return self._extract_via_full_page_ocr(page_image)

        print(f"    Grid: {len(row_bounds) - 1} rows x {len(col_bounds) - 1} cols")

        balloon_col = -1
        dimension_col = -1
        for r in range(min(5, len(row_bounds) - 1)):
            for c in range(len(col_bounds) - 1):
                text = self._ocr_cell(page_image, row_bounds[r], row_bounds[r + 1], col_bounds, c).upper()
                if "BALLOON" in text or ("SN" in text and "NO" in text):
                    balloon_col = c
                elif "DIMENSION" in text:
                    dimension_col = c
            if balloon_col >= 0 and dimension_col >= 0:
                break

        if balloon_col < 0:
            balloon_col = 0
        if dimension_col < 0:
            dimension_col = 1

        result: dict[int, str] = {}
        for r in range(len(row_bounds) - 1):
            y1, y2 = row_bounds[r], row_bounds[r + 1]
            if y2 - y1 < 10:
                continue
            balloon_text = self._ocr_cell(page_image, y1, y2, col_bounds, balloon_col)
            digits = re.sub(r"[^0-9]", "", balloon_text)
            if not digits:
                continue
            num = int(digits)
            if num < 1 or num > 99:
                continue
            dimension = self._ocr_cell(page_image, y1, y2, col_bounds, dimension_col).strip()
            if dimension:
                result[num] = dimension

        return result

    def _detect_grid(self, page_image: np.ndarray) -> tuple[list[int], list[int]]:
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )

        h_kernel_w = max(page_image.shape[1] // 20, 50)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

        v_kernel_h = max(page_image.shape[0] // 40, 20)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_h))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

        return (
            self._extract_line_bounds(h_lines, is_horizontal=True),
            self._extract_line_bounds(v_lines, is_horizontal=False),
        )

    def _extract_line_bounds(self, line_mask: np.ndarray, is_horizontal: bool) -> list[int]:
        bounds: list[int] = []
        length = line_mask.shape[0] if is_horizontal else line_mask.shape[1]
        cross_len = line_mask.shape[1] if is_horizontal else line_mask.shape[0]

        for i in range(length):
            count = 0
            for j in range(0, cross_len, 4):
                val = line_mask[i, j] if is_horizontal else line_mask[j, i]
                if val > 0:
                    count += 1
            if count > cross_len // 16:
                if not bounds or i - bounds[-1] > 5:
                    bounds.append(i)
                else:
                    bounds[-1] = (bounds[-1] + i) // 2

        return bounds

    def _ocr_cell(
        self, page_image: np.ndarray, y1: int, y2: int, col_bounds: list[int], col_idx: int
    ) -> str:
        if col_idx < 0 or col_idx >= len(col_bounds) - 1:
            return ""
        x1 = max(0, col_bounds[col_idx] + 2)
        x2 = min(page_image.shape[1], col_bounds[col_idx + 1] - 2)
        y1 = max(0, y1 + 2)
        y2 = min(page_image.shape[0], y2 - 2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return ""

        cell = page_image[y1:y2, x1:x2]
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, (gray.shape[1] * 3, gray.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(upscaled, 160, 255, cv2.THRESH_BINARY)
        padded = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

        config = f"--tessdata-dir {self._tess_data_path} --psm 7"
        text = pytesseract.image_to_string(padded, lang="eng", config=config)
        return text.strip()

    def _extract_via_full_page_ocr(self, page_image: np.ndarray) -> dict[int, str]:
        result: dict[int, str] = {}
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        config = f"--tessdata-dir {self._tess_data_path} --psm 3"
        text = pytesseract.image_to_string(gray, lang="eng", config=config)

        for line in text.split("\n"):
            parts = line.split()
            if len(parts) < 2:
                continue
            digits = re.sub(r"[^0-9]", "", parts[0])
            if digits:
                num = int(digits)
                if 1 <= num <= 99:
                    result.setdefault(num, parts[1])
        return result
