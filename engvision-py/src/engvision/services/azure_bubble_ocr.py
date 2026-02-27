"""Azure Document Intelligence bubble number OCR service.

Drop-in replacement for BubbleOcrService that uses Azure Document Intelligence's
Read API instead of local Tesseract to read the number inside bubble crops.
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential


class AzureBubbleOcrService:
    """Reads bubble numbers from crop images using Azure Document Intelligence."""

    def __init__(self, endpoint: str, key: str) -> None:
        self._client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

    def extract_bubble_number(self, crop_image_path: str) -> int | None:
        """Read the number from a single bubble crop image."""
        src = cv2.imread(crop_image_path, cv2.IMREAD_COLOR)
        if src is None:
            return None
        return self._extract_from_mat(src)

    def extract_all(self, crop_directory: str) -> dict[str, int | None]:
        """Batch process all bubble_*.png files in a directory."""
        results: dict[str, int | None] = {}
        files = sorted(
            f for f in os.listdir(crop_directory)
            if f.startswith("bubble_") and f.endswith(".png")
        )
        for filename in files:
            path = os.path.join(crop_directory, filename)
            number = self.extract_bubble_number(path)
            results[filename] = number
        return results

    def _extract_from_mat(self, src: np.ndarray) -> int | None:
        """Preprocess and send image to Azure for OCR."""
        processed = self._preprocess_for_ocr(src)
        _, buf = cv2.imencode(".png", processed)

        poller = self._client.begin_analyze_document(
            "prebuilt-read",
            analyze_request=buf.tobytes(),
            content_type="application/octet-stream",
        )
        result = poller.result()

        # Collect all text from the result
        all_text = ""
        if result.content:
            all_text = result.content
        return self._parse_bubble_number(all_text)

    @staticmethod
    def _preprocess_for_ocr(src: np.ndarray) -> np.ndarray:
        """Remove blue circle pixels and upscale for better OCR accuracy."""
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([85, 25, 50]), np.array([125, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_blue = cv2.dilate(blue_mask, kernel, iterations=1)

        cleaned = src.copy()
        cleaned[dilated_blue > 0] = [255, 255, 255]

        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

        # Upscale 4x (Azure handles larger images well)
        upscaled = cv2.resize(
            gray, (gray.shape[1] * 4, gray.shape[0] * 4),
            interpolation=cv2.INTER_CUBIC,
        )

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )

        # White border padding
        padded = cv2.copyMakeBorder(
            binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255,
        )
        return padded

    @staticmethod
    def _parse_bubble_number(ocr_text: str) -> int | None:
        """Extract an integer bubble number (1-99) from OCR text."""
        if not ocr_text.strip():
            return None
        cleaned = (
            ocr_text
            .replace("#", "")
            .replace("O", "0").replace("o", "0")
            .replace("l", "1").replace("I", "1")
            .replace("S", "5").replace("B", "8")
            .replace(" ", "")
            .strip()
        )
        digits = re.sub(r"[^0-9]", "", cleaned)
        if digits:
            num = int(digits)
            if 1 <= num <= 99:
                return num
        return None
