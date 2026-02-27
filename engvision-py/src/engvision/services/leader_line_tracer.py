"""Leader line direction finder using Harris corner detection + connectivity.

Each bubble has a small triangular arrowhead physically attached to its
perimeter.  We use connected-component analysis to ensure we only consider
blue shapes that are actually connected to the target circle — not a
neighbouring bubble.  Harris corner detection then identifies the sharp
triangle vertices within the connected component.

This eliminates the neighbour-contamination problem that occurs when
bubbles overlap or are very close together.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..models import RegionType

# Progressive capture box sizes (width, height) — wider than tall.
# Pipeline tries each in order; stops when the LLM confirms a match.
CAPTURE_STEPS = [
    (128, 128),   # start square
    (256, 128),   # expand width
    (512, 256),   # keep expanding
    (1024, 512),  # max: 1024 px wide
]


class LeaderLineTracerService:
    def trace_and_expand(
        self, bubbles: list[dict], page_image: np.ndarray
    ) -> list[dict]:
        """For each bubble, find the leader-line direction by detecting the
        triangle pointer, then produce an expanded region + capture box."""
        hsv = cv2.cvtColor(page_image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([85, 25, 50]), np.array([125, 255, 255]))
        img_h, img_w = page_image.shape[:2]

        # Collect bubble geometry
        bubble_info: list[tuple[int, int, int]] = []
        for bubble in bubbles:
            bb = bubble["boundingBox"]
            bcx = bb["x"] + bb["width"] // 2
            bcy = bb["y"] + bb["height"] // 2
            b_radius = bb["width"] // 2
            bubble_info.append((bcx, bcy, b_radius))

        expanded = []
        for i, bubble in enumerate(bubbles):
            bcx, bcy, b_radius = bubble_info[i]
            bb = bubble["boundingBox"]

            direction = _find_triangle_direction(
                blue_mask, bcx, bcy, b_radius, bubble_info, i, img_w, img_h
            )

            if direction is None:
                expanded.append({
                    **bubble,
                    "type": RegionType.BUBBLE_WITH_FIGURE.value,
                    "captureBox": None,
                    "leaderDirection": None,
                })
                continue

            dx, dy = direction

            # Place initial capture box along the ray (smallest step)
            init_w, init_h = CAPTURE_STEPS[0]
            cap_box = self.place_capture_box(
                bcx, bcy, b_radius, dx, dy, init_w, init_h, img_w, img_h
            )

            # Expanded bounding box = union of bubble + capture box
            min_x = min(bb["x"], cap_box["x"])
            min_y = min(bb["y"], cap_box["y"])
            max_x = max(bb["x"] + bb["width"], cap_box["x"] + cap_box["width"])
            max_y = max(bb["y"] + bb["height"], cap_box["y"] + cap_box["height"])

            expanded.append({
                **bubble,
                "type": RegionType.BUBBLE_WITH_FIGURE.value,
                "boundingBox": {
                    "x": min_x, "y": min_y,
                    "width": max_x - min_x, "height": max_y - min_y,
                },
                "captureBox": cap_box,
                "leaderDirection": {"dx": dx, "dy": dy},
            })
        return expanded

    @staticmethod
    def place_capture_box(
        bcx: int, bcy: int, b_radius: int,
        dx: float, dy: float,
        width: int, height: int,
        img_w: int, img_h: int,
    ) -> dict:
        """Place a width×height capture box along the ray (dx, dy) from the
        bubble centre.  The box centre is at a fixed distance from the bubble
        edge (anchored to the initial 128×128 step), so larger boxes expand
        outward from the same position rather than shifting away."""
        half_w = width // 2
        half_h = height // 2

        # Fixed anchor: centre is always at bubble_edge + 68px along the ray,
        # matching the 128×128 step.  Larger boxes just grow around this point.
        anchor_half = CAPTURE_STEPS[0][0] // 2  # 64
        box_dist = b_radius + anchor_half + 4
        box_cx = bcx + dx * box_dist
        box_cy = bcy + dy * box_dist

        # Ensure no corner lands behind the bubble centre
        corners = [
            (box_cx - half_w, box_cy - half_h),
            (box_cx + half_w, box_cy - half_h),
            (box_cx - half_w, box_cy + half_h),
            (box_cx + half_w, box_cy + half_h),
        ]
        min_dot = min((cx - bcx) * dx + (cy - bcy) * dy for cx, cy in corners)

        if min_dot < 0:
            push = -min_dot + 1.0
            box_cx += dx * push
            box_cy += dy * push

        x1 = max(0, int(box_cx - half_w))
        y1 = max(0, int(box_cy - half_h))
        x2 = min(img_w, int(box_cx + half_w))
        y2 = min(img_h, int(box_cy + half_h))

        return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def _find_triangle_direction(
    blue_mask: np.ndarray,
    bcx: int, bcy: int, b_radius: int,
    all_bubbles: list[tuple[int, int, int]],
    self_idx: int,
    img_w: int, img_h: int,
) -> tuple[float, float] | None:
    """Find the triangle pointer using flood-fill connectivity + Harris corners.

    Instead of erasing ALL circles and hoping the triangle remnant survives,
    we keep the target circle intact, erase only OTHER circles, then flood-fill
    from the target center to find exactly which blue pixels are connected to
    it.  After extracting that component, we erase the circle interior — the
    full triangle remains because it was never severed from its base.

    Harris corner detection on the surviving triangle gives the direction.
    """
    search_r = b_radius * 3
    rx1 = max(0, bcx - search_r)
    ry1 = max(0, bcy - search_r)
    rx2 = min(img_w, bcx + search_r)
    ry2 = min(img_h, bcy + search_r)
    if rx2 - rx1 < 10 or ry2 - ry1 < 10:
        return None

    # Work on a copy — erase OTHER circles but keep the target intact
    roi = blue_mask[ry1:ry2, rx1:rx2].copy()
    roi_cx = bcx - rx1
    roi_cy = bcy - ry1

    for j, (ocx, ocy, o_r) in enumerate(all_bubbles):
        if j == self_idx:
            continue
        ox = ocx - rx1
        oy = ocy - ry1
        if -o_r * 2 < ox < roi.shape[1] + o_r * 2 and \
           -o_r * 2 < oy < roi.shape[0] + o_r * 2:
            cv2.circle(roi, (ox, oy), o_r + 1, 0, -1)

    # Find a blue seed pixel on the target circle's perimeter
    seed = None
    for angle_deg in range(0, 360, 5):
        angle_rad = math.radians(angle_deg)
        px = int(roi_cx + b_radius * math.cos(angle_rad))
        py = int(roi_cy + b_radius * math.sin(angle_rad))
        if 0 <= px < roi.shape[1] and 0 <= py < roi.shape[0]:
            if roi[py, px] == 255:
                seed = (px, py)
                break

    if seed is None:
        # Try slightly inside/outside the perimeter
        for offset in [-1, 1, -2, 2]:
            for angle_deg in range(0, 360, 5):
                angle_rad = math.radians(angle_deg)
                px = int(roi_cx + (b_radius + offset) * math.cos(angle_rad))
                py = int(roi_cy + (b_radius + offset) * math.sin(angle_rad))
                if 0 <= px < roi.shape[1] and 0 <= py < roi.shape[0]:
                    if roi[py, px] == 255:
                        seed = (px, py)
                        break
            if seed:
                break

    if seed is None:
        return None

    # Flood-fill from the perimeter seed to find the connected component
    # (target circle perimeter + its triangle pointer)
    flood_img = roi.copy()
    flood_mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood_img, flood_mask, seed, 128)

    # Extract only the flooded component (target circle + its triangle)
    component = (flood_img == 128).astype(np.uint8) * 255

    # Erase the circle including its full perimeter stroke.
    # The perimeter ring extends ~2px beyond the nominal radius, so erase
    # at r+2 to remove it completely.  Only the triangle tip that protrudes
    # beyond the perimeter ring will survive.
    cv2.circle(component, (roi_cx, roi_cy), b_radius + 2, 0, -1)

    # Check we have enough pixels remaining
    remaining = cv2.countNonZero(component)
    if remaining < 3:
        return None

    # ── Harris corner detection on the triangle ──────────────────────────────
    harris = cv2.cornerHarris(component, blockSize=3, ksize=3, k=0.04)
    h_max = harris.max()
    if h_max <= 0:
        # Fallback: centroid of remaining pixels
        pts = np.argwhere(component > 0)  # [y, x]
        if len(pts) == 0:
            return None
        mean_y, mean_x = pts.mean(axis=0)
        dx = mean_x - roi_cx
        dy = mean_y - roi_cy
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            return None
        return (dx / length, dy / length)

    threshold = 0.01 * h_max
    corner_ys, corner_xs = np.where(harris > threshold)
    if len(corner_xs) == 0:
        return None

    sum_dx = 0.0
    sum_dy = 0.0
    total_weight = 0.0

    for cx, cy in zip(corner_xs, corner_ys):
        dx = float(cx) - roi_cx
        dy = float(cy) - roi_cy
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < b_radius * 0.3 or dist > b_radius * 3.0:
            continue

        response = harris[cy, cx]
        edge_weight = max(0.1, 1.0 - abs(dist - b_radius) / (b_radius * 1.5))
        weight = response * edge_weight

        sum_dx += dx * weight
        sum_dy += dy * weight
        total_weight += weight

    if total_weight < 1e-6:
        # Fallback: centroid of remaining pixels
        pts = np.argwhere(component > 0)
        if len(pts) == 0:
            return None
        mean_y, mean_x = pts.mean(axis=0)
        dx = mean_x - roi_cx
        dy = mean_y - roi_cy
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            return None
        return (dx / length, dy / length)

    length = math.sqrt(sum_dx * sum_dx + sum_dy * sum_dy)
    if length < 1:
        return None
    return (sum_dx / length, sum_dy / length)

