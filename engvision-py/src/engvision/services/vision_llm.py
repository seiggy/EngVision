"""Vision LLM service using the OpenAI Python SDK, ported from .NET VisionLlmService."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AzureOpenAI


@dataclass
class LlmExtractionResult:
    dimensions: dict[int, str] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class VisionLlmService:
    def __init__(self, client: AzureOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def extract_balloon_dimensions_with_usage(
        self, page_image_bytes: bytes, page_number: int
    ) -> LlmExtractionResult:
        b64 = base64.b64encode(page_image_bytes).decode("utf-8")

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at reading engineering dimensional analysis tables.\n"
                            "This image shows a page from a dimensional analysis report.\n\n"
                            'The table has columns including "BALLOON NO." (or "SN") and "DIMENSION".\n'
                            "Extract EVERY row's balloon number and dimension value.\n\n"
                            "Return a JSON array of objects:\n"
                            '[\n    { "balloonNo": <integer>, "dimension": "<dimension text exactly as shown>" }\n]\n\n'
                            "Rules:\n"
                            "- Include ALL rows, even if some cells are hard to read\n"
                            "- The balloon number is always an integer (2 through 51)\n"
                            '- The dimension text should be copied exactly as shown (e.g., "0.81", "18°", "Ø.500", "MATERIAL")\n'
                            "- If a row has sub-rows or multiple dimension values, include each as a separate entry\n"
                            "- Do NOT skip any rows\n"
                            "- Return ONLY the JSON array, no other text"
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Extract all balloon number and dimension pairs from this table page (page {page_number}):",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    },
                ],
            )

            content = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            if not content:
                return LlmExtractionResult(
                    input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens
                )

            content = _strip_code_fences(content)
            rows = json.loads(content)

            result: dict[int, str] = {}
            for row in rows:
                balloon_no = row.get("balloonNo", 0)
                dimension = row.get("dimension", "")
                if balloon_no > 0 and dimension and dimension.strip():
                    result.setdefault(balloon_no, dimension.strip())

            return LlmExtractionResult(
                dimensions=result,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        except Exception as ex:
            print(f"  Error extracting balloon dimensions from page {page_number}: {ex}")
            return LlmExtractionResult()

    async def extract_balloon_dimensions(
        self, page_image_bytes: bytes, page_number: int
    ) -> dict[int, str]:
        result = await self.extract_balloon_dimensions_with_usage(page_image_bytes, page_number)
        return result.dimensions


def _strip_code_fences(content: str) -> str:
    if content.startswith("```"):
        lines = content.split("\n")
        start_idx = 1 if lines[0].startswith("```") else 0
        end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        content = "\n".join(lines[start_idx:end_idx])
    return content.strip()
