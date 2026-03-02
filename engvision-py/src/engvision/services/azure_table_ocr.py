"""Azure Document Intelligence table OCR service.

Drop-in replacement for TableOcrService that uses Azure Document Intelligence's
prebuilt-layout model instead of local Tesseract + morphological grid detection.
Builds a full cell matrix (handling row_span/column_span) for accurate extraction.
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
        """Extract balloon→dimension from a single page image."""
        _, buf = cv2.imencode(".png", page_image)
        return self._extract_from_bytes(buf.tobytes())

    def extract_balloon_dimensions_from_pdf(self, pdf_bytes: bytes) -> dict[int, str]:
        """Extract balloon→dimension from all pages of a PDF in a single API call."""
        return self._extract_from_bytes(pdf_bytes)

    def _extract_from_bytes(self, document_bytes: bytes) -> dict[int, str]:
        """Send document bytes to Azure and parse the returned tables."""
        poller = self._client.begin_analyze_document(
            "prebuilt-layout",
            body=document_bytes,
            content_type="application/octet-stream",
        )
        result = poller.result()

        dimensions: dict[int, str] = {}

        if not result.tables:
            print("    Azure Doc Intelligence: no tables found")
            return dimensions

        print(f"    Azure Doc Intelligence: found {len(result.tables)} table(s)")

        for table in result.tables:
            # Build a full matrix handling row_span and column_span
            matrix = self._build_matrix(table)

            # Debug: dump first few rows
            dump_rows = min(5, table.row_count)
            for r in range(dump_rows):
                row_cells = " | ".join(
                    matrix[r][c][:30] + "…" if len(matrix[r][c]) > 30 else matrix[r][c]
                    for c in range(table.column_count)
                )
                print(f"      Row {r}: {row_cells}")

            # Find balloon and dimension columns from the matrix
            balloon_col, dim_col, header_row = self._find_columns_from_matrix(
                matrix, table.row_count, table.column_count
            )
            print(f"      → balloon_col={balloon_col}, dim_col={dim_col}, header_row={header_row}")

            if balloon_col is None or dim_col is None:
                continue

            # Extract balloon→dimension pairs from rows below the header
            for r in range(header_row + 1, table.row_count):
                balloon_text = matrix[r][balloon_col]
                balloon_no = self._parse_balloon_number(balloon_text)
                if balloon_no is None:
                    continue

                dim_val = matrix[r][dim_col].strip()
                if dim_val:
                    dimensions.setdefault(balloon_no, dim_val)
                    print(f"      Balloon {balloon_no} → {dim_val}")

        print(f"    Azure Doc Intelligence: extracted {len(dimensions)} balloon→dimension mappings")
        return dimensions

    @staticmethod
    def _build_matrix(table) -> list[list[str]]:
        """Build a 2D string matrix from table cells, handling row_span/column_span."""
        matrix: list[list[str]] = [
            ["" for _ in range(table.column_count)] for _ in range(table.row_count)
        ]

        for cell in table.cells:
            text = (cell.content or "").strip()
            row_span = getattr(cell, "row_span", 1) or 1
            col_span = getattr(cell, "column_span", 1) or 1

            for rr in range(cell.row_index, min(cell.row_index + row_span, table.row_count)):
                for cc in range(cell.column_index, min(cell.column_index + col_span, table.column_count)):
                    matrix[rr][cc] = text

        return matrix

    @staticmethod
    def _find_columns_from_matrix(
        matrix: list[list[str]], rows: int, cols: int
    ) -> tuple[int | None, int | None, int]:
        """Find balloon and dimension columns by scanning matrix rows for header keywords."""
        scan_rows = min(5, rows)
        for r in range(scan_rows):
            balloon_col = None
            dim_col = None
            for c in range(cols):
                text = matrix[r][c].upper()
                if balloon_col is None and ("BALLOON" in text or "BALL" in text or "ITEM" in text):
                    balloon_col = c
                elif dim_col is None and ("DIM" in text or "NOMINAL" in text or "VALUE" in text or "SIZE" in text):
                    dim_col = c
            if balloon_col is not None and dim_col is not None:
                return balloon_col, dim_col, r
        return None, None, 0

    @staticmethod
    def _parse_balloon_number(text: str) -> int | None:
        """Extract an integer balloon number (1-99) from cell text."""
        cleaned = re.sub(r"[^0-9]", "", text.strip())
        if cleaned:
            num = int(cleaned)
            if 1 <= num <= 99:
                return num
        return None
