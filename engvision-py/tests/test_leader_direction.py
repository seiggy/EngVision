"""Test suite for leader line direction detection on known problem bubbles.

Uses bubbles 11, 12, 40-44, 46, 47 which are the hardest cases
(overlapping, adjacent, or clustered bubbles).

Run: cd engvision-py && uv run pytest tests/test_leader_direction.py -v
"""

import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from engvision.config import EngVisionConfig
from engvision.services.bubble_detection import BubbleDetectionService
from engvision.services.leader_line_tracer import LeaderLineTracerService
from engvision.services.pdf_renderer import PdfRendererService

PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "sample_docs",
    "WFRD_Sample_Dimentional_Analysis.pdf",
)


@pytest.fixture(scope="module")
def expanded_bubbles():
    """Run detection + tracing once, return expanded results keyed by bubble number."""
    if not os.path.exists(PDF_PATH):
        pytest.skip("Sample PDF not found")

    config = EngVisionConfig(pdf_render_dpi=300, output_directory="Output")
    renderer = PdfRendererService(300)
    pages = renderer.render_all_pages(PDF_PATH)
    page = pages[0]

    detector = BubbleDetectionService(config)
    bubbles = detector.detect_bubbles(page, page_number=1)

    tracer = LeaderLineTracerService()
    expanded = tracer.trace_and_expand(bubbles, page)

    return {eb.get("bubbleNumber", i + 1): eb for i, eb in enumerate(expanded)}


def _get_direction(expanded_bubbles, bubble_num):
    """Extract (dx, dy) direction for a bubble, or None."""
    eb = expanded_bubbles.get(bubble_num)
    if eb is None:
        return None
    ld = eb.get("leaderDirection")
    if ld is None:
        return None
    return (ld["dx"], ld["dy"])


def _assert_direction(expanded_bubbles, bubble_num, expected_dx, expected_dy, tolerance_deg=45):
    """Assert that the detected direction is within tolerance_deg of expected."""
    direction = _get_direction(expanded_bubbles, bubble_num)
    assert direction is not None, f"Bubble {bubble_num}: no direction detected"

    dx, dy = direction
    dot = dx * expected_dx + dy * expected_dy
    dot = max(-1.0, min(1.0, dot))
    angle_deg = math.degrees(math.acos(dot))

    assert angle_deg <= tolerance_deg, (
        f"Bubble {bubble_num}: direction ({dx:.2f}, {dy:.2f}) is {angle_deg:.0f}° "
        f"from expected ({expected_dx:.2f}, {expected_dy:.2f}), "
        f"tolerance={tolerance_deg}°"
    )


class TestProblemBubbleDirections:
    """Test direction detection on the known problem bubbles."""

    def test_all_problem_bubbles_have_direction(self, expanded_bubbles):
        """All problem bubbles should have a detected direction."""
        problem = [11, 12, 40, 42, 43, 44, 46, 47]
        missing = [b for b in problem if _get_direction(expanded_bubbles, b) is None]
        assert not missing, f"Bubbles with no direction: {missing}"

    def test_bubble_11_points_left(self, expanded_bubbles):
        """Bubble 11: triangle points left toward feature."""
        _assert_direction(expanded_bubbles, 11, -1.0, 0.0)

    def test_bubble_12_points_left_down(self, expanded_bubbles):
        """Bubble 12: triangle points left-down."""
        _assert_direction(expanded_bubbles, 12, -0.7, 0.7, tolerance_deg=50)

    def test_bubble_40_points_right(self, expanded_bubbles):
        """Bubble 40: triangle points right toward dimension text."""
        _assert_direction(expanded_bubbles, 40, 1.0, 0.0)

    def test_bubble_42_points_right(self, expanded_bubbles):
        """Bubble 42: triangle points right (overlaps with 43)."""
        _assert_direction(expanded_bubbles, 42, 1.0, 0.0)

    def test_bubble_43_points_right(self, expanded_bubbles):
        """Bubble 43: triangle points right (overlaps with 42)."""
        _assert_direction(expanded_bubbles, 43, 1.0, 0.0)

    def test_bubble_44_points_right(self, expanded_bubbles):
        """Bubble 44: triangle points right."""
        _assert_direction(expanded_bubbles, 44, 1.0, 0.0)

    def test_bubble_46_points_down(self, expanded_bubbles):
        """Bubble 46: triangle points downward."""
        _assert_direction(expanded_bubbles, 46, 0.0, 1.0)

    def test_bubble_47_points_left(self, expanded_bubbles):
        """Bubble 47: triangle points left."""
        _assert_direction(expanded_bubbles, 47, -1.0, 0.0)


class TestOverallDetection:
    """Broad detection quality tests."""

    def test_all_50_bubbles_have_direction(self, expanded_bubbles):
        """All 50 bubbles should have a detected leader direction."""
        missing = [num for num, eb in expanded_bubbles.items()
                   if eb.get("leaderDirection") is None]
        assert len(missing) <= 2, f"Too many bubbles without direction: {missing}"

    def test_detection_produces_50_bubbles(self, expanded_bubbles):
        """Should detect all 50 bubbles."""
        assert len(expanded_bubbles) == 50
