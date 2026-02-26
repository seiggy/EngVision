"""PDF page rendering using PyMuPDF (fitz)."""

from __future__ import annotations

import os

import cv2
import fitz  # pymupdf
import numpy as np


class PdfRendererService:
    def __init__(self, dpi: int = 300) -> None:
        self._dpi = dpi

    def render_all_pages(self, pdf_path: str) -> list[np.ndarray]:
        """Render all pages of a PDF to a list of BGR numpy arrays."""
        doc = fitz.open(pdf_path)
        pages: list[np.ndarray] = []
        zoom = self._dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        print(f"PDF has {len(doc)} page(s), rendering at {self._dpi} DPI...")
        for i in range(len(doc)):
            mat = self._render_page_internal(doc, i, matrix)
            pages.append(mat)
            print(f"  Page {i + 1}: {mat.shape[1]}x{mat.shape[0]}")

        doc.close()
        return pages

    def render_page(self, pdf_path: str, page_index: int) -> np.ndarray:
        """Render a single page to a BGR numpy array."""
        doc = fitz.open(pdf_path)
        zoom = self._dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        mat = self._render_page_internal(doc, page_index, matrix)
        doc.close()
        return mat

    def _render_page_internal(self, doc: fitz.Document, page_index: int, matrix: fitz.Matrix) -> np.ndarray:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=matrix)
        # Convert to numpy BGR (OpenCV format)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> str:
        cv2.imwrite(output_path, image)
        return output_path

    def get_page_count(self, pdf_path: str) -> int:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
