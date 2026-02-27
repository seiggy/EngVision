"""Application configuration, mirroring EngVisionConfig from .NET."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class EngVisionConfig:
    openai_api_key: str = ""
    openai_model: str = "gpt-5.3-codex"
    openai_endpoint: str | None = None
    pdf_render_dpi: int = 300
    output_directory: str = "Output"

    # Bubble detection parameters
    hough_min_radius: int = 12
    hough_max_radius: int = 50
    hough_param1: float = 120.0
    hough_param2: float = 25.0
    bubble_context_padding: int = 150

    # Table detection parameters
    table_min_width: int = 200
    table_min_height: int = 100

    # OCR provider: "Tesseract" (default, local) or "Azure" (Document Intelligence)
    ocr_provider: str = "Tesseract"
    azure_docint_endpoint: str = ""
    azure_docint_key: str = ""

    @classmethod
    def from_env(cls, base_dir: str) -> "EngVisionConfig":
        return cls(
            pdf_render_dpi=300,
            output_directory=os.path.join(base_dir, "Output"),
            ocr_provider=os.environ.get("OCR_PROVIDER", "Tesseract"),
            azure_docint_endpoint=os.environ.get("AZURE_DOCINT_ENDPOINT", ""),
            azure_docint_key=os.environ.get("AZURE_DOCINT_KEY", ""),
        )
