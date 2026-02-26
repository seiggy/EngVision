"""Pydantic models matching the .NET API's data contracts."""

from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class AnnotationPoint(BaseModel):
    x: int
    y: int


class RegionType(int, enum.Enum):
    BUBBLE = 0
    BUBBLE_WITH_FIGURE = 1
    TABLE_REGION = 2
    FULL_PAGE = 3


class DetectedRegion(BaseModel):
    id: int
    page_number: int = Field(alias="pageNumber", default=0)
    type: RegionType
    bounding_box: BoundingBox = Field(alias="boundingBox")
    bubble_number: Optional[int] = Field(alias="bubbleNumber", default=None)
    label: Optional[str] = None
    cropped_image_path: Optional[str] = Field(alias="croppedImagePath", default=None)

    model_config = {"populate_by_name": True}


class Annotation(BaseModel):
    id: str = ""
    bubble_number: Optional[int] = Field(alias="bubbleNumber", default=None)
    bubble_center: Optional[AnnotationPoint] = Field(alias="bubbleCenter", default=None)
    bounding_box: BoundingBox = Field(alias="boundingBox")
    label: Optional[str] = None
    notes: Optional[str] = None

    model_config = {"populate_by_name": True}


class PdfInfo(BaseModel):
    page_count: int = Field(alias="pageCount")
    width: int
    height: int

    model_config = {"populate_by_name": True}


class AccuracyResult(BaseModel):
    ground_truth: int = Field(alias="groundTruth")
    detected: int
    matched: int
    missed: int
    false_positives: int = Field(alias="falsePositives")
    precision: float
    recall: float
    f1: float

    model_config = {"populate_by_name": True}


class BubbleResult(BaseModel):
    bubble_number: int = Field(alias="bubbleNumber")
    cx: int
    cy: int
    radius: int
    bounding_box: BoundingBox = Field(alias="boundingBox")

    model_config = {"populate_by_name": True}


class DimensionMatch(BaseModel):
    balloon_no: int = Field(alias="balloonNo")
    dimension: Optional[str] = None
    source: str = "None"
    tesseract_value: Optional[str] = Field(alias="tesseractValue", default=None)
    llm_value: Optional[str] = Field(alias="llmValue", default=None)
    has_conflict: bool = Field(alias="hasConflict", default=False)
    confidence: float = 0.0

    model_config = {"populate_by_name": True}


class ProcessingMetrics(BaseModel):
    total_duration_ms: int = Field(alias="totalDurationMs")
    render_duration_ms: int = Field(alias="renderDurationMs")
    detect_duration_ms: int = Field(alias="detectDurationMs")
    ocr_duration_ms: int = Field(alias="ocrDurationMs")
    llm_duration_ms: int = Field(alias="llmDurationMs")
    merge_duration_ms: int = Field(alias="mergeDurationMs")
    peak_memory_mb: float = Field(alias="peakMemoryMb")

    model_config = {"populate_by_name": True}


class LlmTokenUsage(BaseModel):
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    total_tokens: int = Field(alias="totalTokens")
    llm_calls: int = Field(alias="llmCalls")

    model_config = {"populate_by_name": True}


class PipelineResult(BaseModel):
    run_id: str = Field(alias="runId")
    pdf_filename: str = Field(alias="pdfFilename")
    page_count: int = Field(alias="pageCount", default=0)
    image_width: int = Field(alias="imageWidth", default=0)
    image_height: int = Field(alias="imageHeight", default=0)
    bubbles: list[BubbleResult] = []
    dimension_map: dict[str, DimensionMatch] = Field(alias="dimensionMap", default_factory=dict)
    total_bubbles: int = Field(alias="totalBubbles", default=0)
    matched_bubbles: int = Field(alias="matchedBubbles", default=0)
    unmatched_bubbles: int = Field(alias="unmatchedBubbles", default=0)
    warnings: int = 0
    metrics: Optional[ProcessingMetrics] = None
    token_usage: Optional[LlmTokenUsage] = Field(alias="tokenUsage", default=None)
    status: str = "complete"
    error: Optional[str] = None

    model_config = {"populate_by_name": True}


class MeasurementData(BaseModel):
    bubble_number: int = Field(alias="bubbleNumber", default=0)
    dimension_name: Optional[str] = Field(alias="dimensionName", default=None)
    nominal_value: Optional[str] = Field(alias="nominalValue", default=None)
    unit: Optional[str] = None
    upper_tolerance: Optional[str] = Field(alias="upperTolerance", default=None)
    lower_tolerance: Optional[str] = Field(alias="lowerTolerance", default=None)
    actual_value: Optional[str] = Field(alias="actualValue", default=None)
    raw_text: Optional[str] = Field(alias="rawText", default=None)
    source_page: int = Field(alias="sourcePage", default=0)
    source_type: RegionType = Field(alias="sourceType", default=RegionType.BUBBLE)

    model_config = {"populate_by_name": True}


class ComparisonResult(BaseModel):
    bubble_number: int = Field(alias="bubbleNumber")
    bubble_measurement: Optional[MeasurementData] = Field(alias="bubbleMeasurement", default=None)
    table_measurement: Optional[MeasurementData] = Field(alias="tableMeasurement", default=None)
    match: bool = False
    discrepancy: Optional[str] = None

    model_config = {"populate_by_name": True}
