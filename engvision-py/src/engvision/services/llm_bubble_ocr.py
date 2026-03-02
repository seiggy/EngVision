"""LLM vision-based bubble number OCR — faster and more accurate than Doc Intelligence."""

from __future__ import annotations

import asyncio
import base64
import os
import re
from typing import Callable

from openai import OpenAI


class LlmBubbleOcrService:
    """Uses GPT vision to read bubble numbers from crop images in parallel."""

    MAX_PARALLELISM = 5

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def extract_bubble_number(self, crop_image_path: str) -> int | None:
        """Synchronous single-image extraction."""
        with open(crop_image_path, "rb") as f:
            img_bytes = f.read()
        return self._call_llm(img_bytes)

    def _call_llm(self, image_bytes: bytes) -> int | None:
        b64 = base64.b64encode(image_bytes).decode()
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are reading a small cropped image from an engineering drawing. "
                            "The image contains a circle (balloon/bubble) with a number inside it. "
                            "The number is between 1 and 99.\n\n"
                            "Return ONLY the integer number you see. Nothing else. "
                            "If you cannot read a number, return \"null\"."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What number is in this bubble?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    },
                ],
                max_tokens=10,
            )
            text = (resp.choices[0].message.content or "").strip()
            digits = re.sub(r"[^0-9]", "", text)
            if digits:
                num = int(digits)
                if 1 <= num <= 99:
                    return num
            return None
        except Exception as ex:
            print(f"  [LLM-OCR] Error: {ex}")
            return None

    def extract_all(
        self,
        crop_directory: str,
        on_progress: Callable[[int, str, int | None], None] | None = None,
    ) -> dict[str, int | None]:
        """Extract all bubble numbers using thread-pool parallelism."""
        import glob
        from concurrent.futures import ThreadPoolExecutor, as_completed

        files = sorted(glob.glob(os.path.join(crop_directory, "bubble_*.png")))
        results: dict[str, int | None] = {}
        completed = 0

        def _process(path: str) -> tuple[str, int | None]:
            return os.path.basename(path), self.extract_bubble_number(path)

        with ThreadPoolExecutor(max_workers=self.MAX_PARALLELISM) as pool:
            futures = {pool.submit(_process, p): p for p in files}
            for future in as_completed(futures):
                filename, number = future.result()
                results[filename] = number
                completed += 1
                print(f"  [LLM-OCR] Bubble {completed}/{len(files)}: {filename} → {number}")
                if on_progress:
                    on_progress(completed, filename, number)

        return results
