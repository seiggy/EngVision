"""Vision LLM service — validates table dimensions against the engineering drawing.

Uses the OpenAI Responses API (not Chat Completions) for compatibility with
models like gpt-5.3-codex that only support the Responses endpoint.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AzureOpenAI, OpenAI


@dataclass
class LlmValidationResult:
    """Result of validating a single bubble's dimension against the drawing."""

    balloon_no: int = 0
    table_dimension: str = ""
    observed_dimension: str = ""
    matches: bool = False
    confidence: float = 0.0
    notes: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LlmValidationBatchResult:
    """Aggregated results from validating multiple bubbles."""

    validations: dict[int, LlmValidationResult] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0


_SYSTEM_PROMPT_VALIDATE = (
    "You are an expert at reading engineering drawings and dimensional annotations.\n\n"
    "You will be given:\n"
    "1. A cropped region from an engineering drawing where a numbered balloon/bubble "
    "points via its leader line.\n"
    "2. A dimension value extracted from the inspection table for that balloon number.\n\n"
    "Your job is to:\n"
    "- Examine the cropped drawing region and find the dimension annotation visible there.\n"
    "- Compare it to the table dimension value provided.\n"
    "- Determine if they match (accounting for formatting differences like leading zeros, "
    "degree symbols, diameter symbols, etc.).\n\n"
    "Return a JSON object:\n"
    "{\n"
    '  "observedDimension": "<the dimension text you see on the drawing, or empty string if none visible>",\n'
    '  "matches": <true if the drawing dimension matches the table value, false otherwise>,\n'
    '  "confidence": <0.0 to 1.0 confidence in your assessment>,\n'
    '  "notes": "<brief explanation, e.g. \'exact match\', \'formatting difference only\', '
    "'dimension not visible in crop', etc.>\"\n"
    "}\n\n"
    "Rules:\n"
    '- If you cannot see any dimension annotation in the crop, set observedDimension to "" '
    "and matches to false with a low confidence.\n"
    "- Treat formatting variations as matches (e.g. '0.81' vs '.81', '18°' vs '18 DEG', "
    "'Ø.500' vs 'DIA .500').\n"
    "- Return ONLY the JSON object, no other text."
)

_SYSTEM_PROMPT_DISCOVER = (
    "You are an expert at reading engineering drawings and dimensional annotations.\n\n"
    "You will be given a cropped region from an engineering drawing where a numbered "
    "balloon/bubble points via its leader line.\n\n"
    "Your job is to find and read the dimension annotation visible in the crop.\n\n"
    "Return a JSON object:\n"
    "{\n"
    '  "observedDimension": "<the dimension text you see on the drawing, or empty string if none visible>",\n'
    '  "confidence": <0.0 to 1.0 confidence in your reading>,\n'
    '  "notes": "<brief description of what you see>"\n'
    "}\n\n"
    "Rules:\n"
    '- If you cannot see any dimension annotation, set observedDimension to "" '
    "with a low confidence.\n"
    "- Include the full dimension text with any symbols (Ø, °, ±, etc.).\n"
    "- Return ONLY the JSON object, no other text."
)


class VisionLlmService:
    def __init__(self, client: OpenAI | AzureOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def validate_dimension(
        self,
        crop_image_bytes: bytes,
        balloon_no: int,
        table_dimension: str,
    ) -> LlmValidationResult:
        """Validate that the dimension on the drawing matches the table value."""
        b64 = base64.b64encode(crop_image_bytes).decode("utf-8")

        user_text = (
            f"Balloon #{balloon_no}\n"
            f"Table dimension value: \"{table_dimension}\"\n\n"
            "Examine the drawing region below and validate whether the "
            "dimension annotation matches the table value:"
        )

        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=_SYSTEM_PROMPT_VALIDATE,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_text},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{b64}",
                            },
                        ],
                    }
                ],
            )

            content = response.output_text or ""
            usage = response.usage
            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0
            total_tokens = (input_tokens + output_tokens) if usage else 0

            if not content:
                return LlmValidationResult(
                    balloon_no=balloon_no,
                    table_dimension=table_dimension,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )

            content = _strip_code_fences(content)
            parsed = json.loads(content)

            return LlmValidationResult(
                balloon_no=balloon_no,
                table_dimension=table_dimension,
                observed_dimension=parsed.get("observedDimension", ""),
                matches=bool(parsed.get("matches", False)),
                confidence=float(parsed.get("confidence", 0.0)),
                notes=parsed.get("notes", ""),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        except Exception as ex:
            print(f"  Error validating balloon #{balloon_no}: {ex}")
            return LlmValidationResult(
                balloon_no=balloon_no,
                table_dimension=table_dimension,
            )

    async def discover_dimension(
        self,
        crop_image_bytes: bytes,
        balloon_no: int,
    ) -> LlmValidationResult:
        """Discover the dimension annotation visible in a crop when no table value exists."""
        b64 = base64.b64encode(crop_image_bytes).decode("utf-8")

        user_text = (
            f"Balloon #{balloon_no}\n\n"
            "Read the dimension annotation visible in this drawing region:"
        )

        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=_SYSTEM_PROMPT_DISCOVER,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_text},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{b64}",
                            },
                        ],
                    }
                ],
            )

            content = response.output_text or ""
            usage = response.usage
            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0
            total_tokens = (input_tokens + output_tokens) if usage else 0

            if not content:
                return LlmValidationResult(
                    balloon_no=balloon_no,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )

            content = _strip_code_fences(content)
            parsed = json.loads(content)

            observed = parsed.get("observedDimension", "")
            return LlmValidationResult(
                balloon_no=balloon_no,
                table_dimension="",
                observed_dimension=observed,
                matches=bool(observed),
                confidence=float(parsed.get("confidence", 0.0)),
                notes=parsed.get("notes", ""),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        except Exception as ex:
            print(f"  Error discovering dimension for balloon #{balloon_no}: {ex}")
            return LlmValidationResult(balloon_no=balloon_no)


def _strip_code_fences(content: str) -> str:
    if content.startswith("```"):
        lines = content.split("\n")
        start_idx = 1 if lines[0].startswith("```") else 0
        end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        content = "\n".join(lines[start_idx:end_idx])
    return content.strip()
