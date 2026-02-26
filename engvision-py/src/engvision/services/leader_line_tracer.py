"""Leader line tracing service, ported from .NET LeaderLineTracerService."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..models import RegionType

SEARCH_RADIUS = 400
TEXT_SEARCH_RADIUS = 200
MIN_LEADER_LENGTH = 15
FINAL_PADDING = 30


class LeaderLineTracerService:
    def trace_and_expand(self, bubbles: list[dict], page_image: np.ndarray) -> list[dict]:
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        text_mask = self._build_text_mask(gray)
        img_h, img_w = page_image.shape[:2]

        expanded = []
        for bubble in bubbles:
            bb = bubble["boundingBox"]
            bcx = bb["x"] + bb["width"] // 2
            bcy = bb["y"] + bb["height"] // 2
            b_radius = bb["width"] // 2

            leader_endpoint = self._find_leader_endpoint(edges, bcx, bcy, b_radius, img_w, img_h)

            if leader_endpoint is not None:
                lx, ly = leader_endpoint
                text_rect = self._find_text_region_near(text_mask, lx, ly, bcx, bcy, img_w, img_h)
            else:
                text_rect = self._find_nearest_text_region(text_mask, bcx, bcy, img_w, img_h)

            min_x, min_y = bb["x"], bb["y"]
            max_x = bb["x"] + bb["width"]
            max_y = bb["y"] + bb["height"]

            if leader_endpoint is not None:
                min_x = min(min_x, leader_endpoint[0] - 5)
                min_y = min(min_y, leader_endpoint[1] - 5)
                max_x = max(max_x, leader_endpoint[0] + 5)
                max_y = max(max_y, leader_endpoint[1] + 5)

            if text_rect and text_rect[2] > 0 and text_rect[3] > 0:
                tx, ty, tw, th = text_rect
                min_x = min(min_x, tx)
                min_y = min(min_y, ty)
                max_x = max(max_x, tx + tw)
                max_y = max(max_y, ty + th)

            min_x = max(0, min_x - FINAL_PADDING)
            min_y = max(0, min_y - FINAL_PADDING)
            max_x = min(img_w, max_x + FINAL_PADDING)
            max_y = min(img_h, max_y + FINAL_PADDING)

            expanded.append({
                **bubble,
                "type": RegionType.BUBBLE_WITH_FIGURE.value,
                "boundingBox": {"x": min_x, "y": min_y, "width": max_x - min_x, "height": max_y - min_y},
            })
        return expanded

    def _find_leader_endpoint(
        self, edges: np.ndarray, bcx: int, bcy: int, b_radius: int, img_w: int, img_h: int
    ) -> tuple[int, int] | None:
        search_pad = SEARCH_RADIUS
        rx1 = max(0, bcx - search_pad)
        ry1 = max(0, bcy - search_pad)
        rx2 = min(img_w, bcx + search_pad)
        ry2 = min(img_h, bcy + search_pad)
        if rx2 - rx1 < 20 or ry2 - ry1 < 20:
            return None

        roi_edges = edges[ry1:ry2, rx1:rx2]
        mask = np.full(roi_edges.shape[:2], 255, dtype=np.uint8)
        roi_bcx, roi_bcy = bcx - rx1, bcy - ry1
        cv2.circle(mask, (roi_bcx, roi_bcy), b_radius + 3, 0, -1)
        masked_edges = cv2.bitwise_and(roi_edges, mask)

        lines = cv2.HoughLinesP(
            masked_edges, rho=1, theta=math.pi / 180, threshold=15,
            minLineLength=MIN_LEADER_LENGTH, maxLineGap=10,
        )
        if lines is None or len(lines) == 0:
            return None

        best_endpoint = None
        best_score = float("inf")

        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1 += rx1; y1 += ry1; x2 += rx1; y2 += ry1

            d1 = _dist(x1, y1, bcx, bcy)
            d2 = _dist(x2, y2, bcx, bcy)

            if d1 < d2:
                near_dist = d1; far_x, far_y = x2, y2
            else:
                near_dist = d2; far_x, far_y = x1, y1

            dist_from_edge = abs(near_dist - b_radius)
            if dist_from_edge > b_radius * 0.8:
                continue
            line_len = _dist(x1, y1, x2, y2)
            if line_len < MIN_LEADER_LENGTH:
                continue

            score = dist_from_edge - line_len * 0.3
            if score < best_score:
                best_score = score
                best_endpoint = (int(far_x), int(far_y))

        return best_endpoint

    def _find_text_region_near(
        self, text_mask: np.ndarray, leader_x: int, leader_y: int,
        bcx: int, bcy: int, img_w: int, img_h: int,
    ) -> tuple[int, int, int, int] | None:
        dx = leader_x - bcx
        dy = leader_y - bcy
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            return self._find_nearest_text_region(text_mask, bcx, bcy, img_w, img_h)

        ndx = dx / length
        ndy = dy / length
        sx = int(leader_x + ndx * 20)
        sy = int(leader_y + ndy * 20)
        search_w = search_h = TEXT_SEARCH_RADIUS

        rx1 = max(0, sx - search_w)
        ry1 = max(0, sy - search_h)
        rx2 = min(img_w, sx + search_w)
        ry2 = min(img_h, sy + search_h)
        if rx2 - rx1 < 10 or ry2 - ry1 < 10:
            return None

        roi = text_mask[ry1:ry2, rx1:rx2]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_rect = None
        best_dist = float("inf")
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 5:
                continue
            page_rect = (x + rx1, y + ry1, w, h)
            tcx = page_rect[0] + page_rect[2] // 2
            tcy = page_rect[1] + page_rect[3] // 2
            dist = _dist(tcx, tcy, leader_x, leader_y)
            if dist < best_dist:
                best_dist = dist
                best_rect = page_rect

        if best_rect and best_rect[2] > 0:
            best_rect = self._expand_to_full_text_block(text_mask, best_rect, img_w, img_h)
        return best_rect

    def _find_nearest_text_region(
        self, text_mask: np.ndarray, bcx: int, bcy: int, img_w: int, img_h: int,
    ) -> tuple[int, int, int, int] | None:
        search_r = SEARCH_RADIUS
        rx1 = max(0, bcx - search_r)
        ry1 = max(0, bcy - search_r)
        rx2 = min(img_w, bcx + search_r)
        ry2 = min(img_h, bcy + search_r)
        if rx2 - rx1 < 10 or ry2 - ry1 < 10:
            return None

        roi = text_mask[ry1:ry2, rx1:rx2]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect = None
        best_dist = float("inf")
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 15 or h < 8:
                continue
            page_rect = (x + rx1, y + ry1, w, h)
            tcx = page_rect[0] + page_rect[2] // 2
            tcy = page_rect[1] + page_rect[3] // 2
            dist = _dist(tcx, tcy, bcx, bcy)
            if dist < 25:
                continue
            if dist < best_dist:
                best_dist = dist
                best_rect = page_rect

        if best_rect and best_rect[2] > 0:
            best_rect = self._expand_to_full_text_block(text_mask, best_rect, img_w, img_h)
        return best_rect

    def _expand_to_full_text_block(
        self, text_mask: np.ndarray, seed: tuple[int, int, int, int], img_w: int, img_h: int,
    ) -> tuple[int, int, int, int]:
        expand = 60
        sx, sy, sw, sh = seed
        rx1 = max(0, sx - expand)
        ry1 = max(0, sy - expand)
        rx2 = min(img_w, sx + sw + expand)
        ry2 = min(img_h, sy + sh + expand)

        roi = text_mask[ry1:ry2, rx1:rx2]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(roi, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        seed_in_roi = (sx - rx1, sy - ry1, sw, sh)
        best = seed

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if _rects_overlap((x, y, w, h), seed_in_roi):
                page_rect = (x + rx1, y + ry1, w, h)
                bx1 = min(best[0], page_rect[0])
                by1 = min(best[1], page_rect[1])
                bx2 = max(best[0] + best[2], page_rect[0] + page_rect[2])
                by2 = max(best[1] + best[3], page_rect[1] + page_rect[3])
                best = (bx1, by1, bx2 - bx1, by2 - by1)

        return best

    def _build_text_mask(self, gray: np.ndarray) -> np.ndarray:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
        text_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_kernel)
        d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 4))
        text_mask = cv2.dilate(text_mask, d_kernel, iterations=1)
        return text_mask


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _rects_overlap(a: tuple, b: tuple) -> bool:
    return (a[0] < b[0] + b[2] and a[0] + a[2] > b[0] and
            a[1] < b[1] + b[3] and a[1] + a[3] > b[1])
