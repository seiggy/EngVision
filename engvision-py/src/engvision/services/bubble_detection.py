"""Bubble detection service using OpenCV, ported from .NET BubbleDetectionService."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from ..config import EngVisionConfig
from ..models import BoundingBox, DetectedRegion, RegionType


class BubbleDetectionService:
    def __init__(self, config: EngVisionConfig) -> None:
        self._config = config
        self.verbose = False

    def detect_bubbles(self, page_image: np.ndarray, page_number: int) -> list[dict]:
        """Detect numbered bubbles on a CAD drawing page. Returns list of DetectedRegion dicts."""
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(page_image, cv2.COLOR_BGR2HSV)

        all_candidates: list[tuple[int, int, int, str]] = []  # (cx, cy, radius, source)

        # PRIMARY: Blue-first detection
        blue_mask = cv2.inRange(hsv, np.array([85, 25, 50]), np.array([125, 255, 255]))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blue_closed = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, close_kernel)

        # Method A: HoughCircles on blue mask (multiple passes)
        blue_blur = cv2.GaussianBlur(blue_closed, (5, 5), 1.0)

        blue_pass_params = [
            (1.0, 18, 60, 12, 10, 28),
            (1.2, 20, 80, 15, 10, 28),
            (1.5, 18, 50, 10, 8, 30),
            (1.0, 15, 40, 8, 8, 25),
        ]

        for dp, min_dist, p1, p2, min_r, max_r in blue_pass_params:
            circles = cv2.HoughCircles(
                blue_blur, cv2.HOUGH_GRADIENT,
                dp=dp, minDist=min_dist, param1=p1, param2=p2,
                minRadius=min_r, maxRadius=max_r,
            )
            before = len(all_candidates)
            _add_unique(all_candidates, circles, "blue-hough")
            if self.verbose:
                added = len(all_candidates) - before
                raw = 0 if circles is None else len(circles[0])
                print(f"  Blue HoughCircles (dp={dp},p2={p2}): {raw} raw, {added} new")

        # Method B: Contour detection on blue mask
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        blue_dilated = cv2.morphologyEx(blue_closed, cv2.MORPH_DILATE, dilate_kernel)
        blue_contours, _ = cv2.findContours(blue_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        blue_contour_added = 0
        for contour in blue_contours:
            if len(contour) < 6:
                continue
            area = cv2.contourArea(contour)
            perim = cv2.arcLength(contour, True)
            if perim < 1:
                continue
            circularity = 4 * math.pi * area / (perim * perim)
            if circularity < 0.25:
                continue
            (cx_f, cy_f), enc_r = cv2.minEnclosingCircle(contour)
            if enc_r < 8 or enc_r > 35:
                continue
            cx, cy, r = int(cx_f), int(cy_f), int(enc_r)
            if not any(_distance(e[0], e[1], cx, cy) < 15 for e in all_candidates):
                all_candidates.append((cx, cy, r, "blue-contour"))
                blue_contour_added += 1

        if self.verbose:
            print(f"  Blue contour candidates: {blue_contour_added}")

        blue_total = len(all_candidates)
        print(f"  Blue-first candidates: {blue_total}")

        # SECONDARY: Grayscale HoughCircles as backup
        blur_coarse = cv2.GaussianBlur(gray, (9, 9), 2)
        blur_fine = cv2.GaussianBlur(gray, (5, 5), 1.0)

        gray_pass_params = [
            (blur_coarse, 1.2, 22, self._config.hough_param1, 23, self._config.hough_min_radius, self._config.hough_max_radius),
            (blur_fine, 1.0, 18, 100, 20, 8, 25),
            (blur_coarse, 1.5, 22, 80, 18, 10, 45),
            (blur_fine, 1.0, 15, 80, 15, 10, 22),
        ]

        gray_added = 0
        for img, dp, min_dist, p1, p2, min_r, max_r in gray_pass_params:
            circles = cv2.HoughCircles(
                img, cv2.HOUGH_GRADIENT,
                dp=dp, minDist=min_dist, param1=p1, param2=p2,
                minRadius=min_r, maxRadius=max_r,
            )
            before = len(all_candidates)
            _add_unique(all_candidates, circles, "gray-hough")
            gray_added += len(all_candidates) - before

        print(f"  Grayscale backup candidates: {gray_added}")

        # Contour-based backup
        contour_added = self._find_circular_contours(gray, all_candidates)
        print(f"  Contour backup candidates: {contour_added}")
        print(f"  Total unique candidates: {len(all_candidates)}")

        # VERIFICATION
        verified = self._verify_bubbles(all_candidates, gray, hsv)
        print(f"  Verified bubbles: {len(verified)}")

        # Sort top-to-bottom, left-to-right
        verified.sort(key=lambda b: (b[1] // 60, b[0]))

        regions = []
        img_h, img_w = page_image.shape[:2]
        for idx, (cx, cy, radius) in enumerate(verified, 1):
            bx = max(0, cx - radius)
            by = max(0, cy - radius)
            bw = min(radius * 2, img_w - bx)
            bh = min(radius * 2, img_h - by)
            regions.append({
                "id": idx,
                "pageNumber": page_number,
                "type": RegionType.BUBBLE.value,
                "boundingBox": {"x": bx, "y": by, "width": bw, "height": bh},
                "bubbleNumber": idx,
                "label": f"Bubble_{idx}",
                "croppedImagePath": None,
            })
        return regions

    def expand_to_figure_regions(
        self, bubbles: list[dict], page_image: np.ndarray, padding: int = -1
    ) -> list[dict]:
        if padding < 0:
            padding = self._config.bubble_context_padding
        img_h, img_w = page_image.shape[:2]
        expanded = []
        for bubble in bubbles:
            bb = bubble["boundingBox"]
            cx = bb["x"] + bb["width"] // 2
            cy = bb["y"] + bb["height"] // 2
            x = max(0, cx - bb["width"] // 2 - padding)
            y = max(0, cy - bb["height"] // 2 - padding)
            w = min(bb["width"] + 2 * padding, img_w - x)
            h = min(bb["height"] + 2 * padding, img_h - y)
            expanded.append({
                **bubble,
                "type": RegionType.BUBBLE_WITH_FIGURE.value,
                "boundingBox": {"x": x, "y": y, "width": w, "height": h},
            })
        return expanded

    def _verify_bubbles(
        self,
        circles: list[tuple[int, int, int, str]],
        gray: np.ndarray,
        hsv: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        passed: list[tuple[int, int, int, float]] = []

        for cx, cy, radius, source in circles:
            margin = 20
            roi_r = max(radius, 20)
            x1 = max(0, cx - roi_r - margin)
            y1 = max(0, cy - roi_r - margin)
            x2 = min(gray.shape[1], cx + roi_r + margin)
            y2 = min(gray.shape[0], cy + roi_r + margin)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            roi = gray[y1:y2, x1:x2]
            hsv_roi = hsv[y1:y2, x1:x2]
            roi_cx, roi_cy = cx - x1, cy - y1

            # Gate 1: Blue perimeter check
            blue_mask = cv2.inRange(hsv_roi, np.array([85, 25, 50]), np.array([125, 255, 255]))

            best_blue_ratio = 0.0
            best_blue_r = radius
            scan_min = 8
            scan_max = min(roi_cx, roi_cy) - 2
            scan_max = min(scan_max, min(roi.shape[1] - roi_cx, roi.shape[0] - roi_cy) - 2)
            scan_max = min(scan_max, 26)

            for scan_r in range(scan_min, scan_max + 1):
                perim_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.circle(perim_mask, (roi_cx, roi_cy), scan_r, 255, 3)
                total_perim_px = cv2.countNonZero(perim_mask)
                if total_perim_px == 0:
                    continue
                blue_perim = cv2.bitwise_and(blue_mask, perim_mask)
                blue_perim_px = cv2.countNonZero(blue_perim)
                ratio = blue_perim_px / total_perim_px
                if ratio > best_blue_ratio:
                    best_blue_ratio = ratio
                    best_blue_r = scan_r

            if best_blue_ratio < 0.15:
                continue
            if best_blue_r < 10 or best_blue_r > 22:
                continue
            verified_r = best_blue_r

            # Gate 2: Interior brightness
            inner_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            inner_r = max(3, int(verified_r * 0.60))
            cv2.circle(inner_mask, (roi_cx, roi_cy), inner_r, 255, -1)
            brightness = cv2.mean(roi, mask=inner_mask)[0]
            if brightness < 120:
                continue

            # Gate 3: Dark text present (3-60%)
            _, inner_binary = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
            dark_masked = cv2.bitwise_and(inner_binary, inner_mask)
            dark_pixels = cv2.countNonZero(dark_masked)
            total_pixels = cv2.countNonZero(inner_mask)
            dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0
            if dark_ratio < 0.03 or dark_ratio > 0.60:
                continue

            # Gate 4a: Continuous blue arc >= 180°
            blue_angles = [False] * 72
            for a in range(72):
                angle = a * 5 * math.pi / 180
                for dr in range(-1, 2):
                    sr = verified_r + dr
                    px = roi_cx + int(sr * math.cos(angle))
                    py = roi_cy + int(sr * math.sin(angle))
                    if 0 <= px < blue_mask.shape[1] and 0 <= py < blue_mask.shape[0]:
                        if blue_mask[py, px] > 0:
                            blue_angles[a] = True
                            break

            longest_run = 0
            current_run = 0
            for _ in range(2):
                for a in range(72):
                    if blue_angles[a]:
                        current_run += 1
                        longest_run = max(longest_run, current_run)
                    else:
                        current_run = 0
            longest_run = min(longest_run, 72)
            longest_arc_deg = longest_run * 5

            # Gate 4b: Triangle pointer detection
            outer_sector_blue = [0] * 12
            for a in range(72):
                angle = a * 5 * math.pi / 180
                sector = a // 6
                for r_off in range(2, 11):
                    sr = verified_r + r_off
                    px = roi_cx + int(sr * math.cos(angle))
                    py = roi_cy + int(sr * math.sin(angle))
                    if 0 <= px < blue_mask.shape[1] and 0 <= py < blue_mask.shape[0]:
                        if blue_mask[py, px] > 0:
                            outer_sector_blue[sector] += 1

            best_triangle_score = 0
            for s in range(12):
                sum3 = outer_sector_blue[s] + outer_sector_blue[(s + 1) % 12] + outer_sector_blue[(s + 2) % 12]
                best_triangle_score = max(best_triangle_score, sum3)
            has_triangle = best_triangle_score >= 8

            if longest_arc_deg < 180 or not has_triangle:
                continue

            score = best_blue_ratio * 100 + (longest_arc_deg / 360.0) * 50 + dark_ratio * 20 + brightness * 0.1
            passed.append((cx, cy, verified_r, score))

        # Deduplicate
        result: list[tuple[int, int, int]] = []
        sorted_passed = sorted(passed, key=lambda p: -p[3])
        used = [False] * len(sorted_passed)
        min_separation = 25.0

        for i in range(len(sorted_passed)):
            if used[i]:
                continue
            best = sorted_passed[i]
            for j in range(i + 1, len(sorted_passed)):
                if not used[j] and _distance(best[0], best[1], sorted_passed[j][0], sorted_passed[j][1]) < min_separation:
                    used[j] = True
            result.append((best[0], best[1], best[2]))

        return result

    def _find_circular_contours(
        self, gray: np.ndarray, candidates: list[tuple[int, int, int, str]]
    ) -> int:
        added = 0
        for block_size in [31, 51, 71]:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 10
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 8:
                    continue
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter < 1:
                    continue
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity < 0.65:
                    continue
                radius_est = math.sqrt(area / math.pi)
                if radius_est < 8 or radius_est > 50:
                    continue
                (cx_f, cy_f), enc_radius = cv2.minEnclosingCircle(contour)
                fill_ratio = area / (math.pi * enc_radius * enc_radius) if enc_radius > 0 else 0
                if fill_ratio < 0.6:
                    continue
                cx, cy, r = int(cx_f), int(cy_f), int(enc_radius)
                if not any(
                    _distance(e[0], e[1], cx, cy) < max(e[2], r) * 0.6 for e in candidates
                ):
                    candidates.append((cx, cy, r, "gray-contour"))
                    added += 1
        return added


def _add_unique(
    dest: list[tuple[int, int, int, str]],
    circles: Optional[np.ndarray],
    label: str,
) -> None:
    if circles is None:
        return
    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        if not any(_distance(e[0], e[1], cx, cy) < 15 for e in dest):
            dest.append((cx, cy, r, label))


def _distance(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
