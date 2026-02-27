"""Azure Document Intelligence table OCR service.

Drop-in replacement for TableOcrService that uses Azure Document Intelligence's
prebuilt-layout model instead of local Tesseract + morphological grid detection.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential


class AzureTableOcrService:
    """Extracts balloon→dimension mappings from table pages using Azure Doc Intelligence."""

    def __init__(self, endpoint: str, key: str) -> None:
        self._client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

    def extract_balloon_dimensions(self, page_image: np.ndarray) -> dict[int, str]:
        """Extract balloon→dimension from a single page image.

        Encodes the image as PNG and sends it to Azure Document Intelligence.
        """
        _, buf = cv2.imencode(".png", page_image)
        return self._extract_from_bytes(buf.tobytes())

    def extract_balloon_dimensions_from_pdf(self, pdf_bytes: bytes) -> dict[int, str]:
        """Extract balloon→dimension from all pages of a PDF in a single API call.

        This is the preferred entry point — sends the full PDF once and extracts
        all tables across all pages.
        """
        return self._extract_from_bytes(pdf_bytes)

    def _extract_from_bytes(self, document_bytes: bytes) -> dict[int, str]:
        """Send document bytes to Azure and parse the returned tables."""
        poller = self._client.begin_analyze_document(
            "prebuilt-layout",
            analyze_request=document_bytes,
            content_type="application/octet-stream",
        )
        result = poller.result()

        dimensions: dict[int, str] = {}

        if not result.tables:
            print("    Azure Doc Intelligence: no tables found")
            return dimensions

        for table in result.tables:
            balloon_col, dim_col = self._find_columns(table)
            if balloon_col is None or dim_col is None:
                continue

            for cell in table.cells:
                if cell.row_index == 0:
                    continue  # skip header row
                if cell.column_index == balloon_col:
                    balloon_no = self._parse_balloon_number(cell.content)
                    if balloon_no is not None:
                        # Find the dimension in the same row
                        dim_val = self._get_cell_content(
                            table, cell.row_index, dim_col
                        )
                        if dim_val:
                            dimensions.setdefault(balloon_no, dim_val)

        print(f"    Azure Doc Intelligence: extracted {len(dimensions)} balloon→dimension mappings")
        return dimensions

    @staticmethod
    def _find_columns(table) -> tuple[int | None, int | None]:
        """Identify the BALLOON and DIMENSION column indices from header row."""
        balloon_col = None
        dim_col = None

        for cell in table.cells:
            if cell.row_index != 0:
                continue
            text = (cell.content or "").upper().strip()
            if "BALLOON" in text or "BALL" in text or "NO" in text or "ITEM" in text:
                balloon_col = cell.column_index
            elif "DIM" in text or "NOMINAL" in text or "VALUE" in text or "SIZE" in text:
                dim_col = cell.column_index

        return balloon_col, dim_col

    @staticmethod
    def _get_cell_content(table, row_index: int, col_index: int) -> str:
        """Find a cell by row/col indices and return its content."""
        for cell in table.cells:
            if cell.row_index == row_index and cell.column_index == col_index:
                return (cell.content or "").strip()
        return ""

    @staticmethod
    def _parse_balloon_number(text: str) -> int | None:
        """Extract an integer balloon number (1-99) from cell text."""
        cleaned = re.sub(r"[^0-9]", "", text.strip())
        if cleaned:
            num = int(cleaned)
            if 1 <= num <= 99:
                return num
        return None
