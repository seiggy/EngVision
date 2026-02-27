"""Dump all capture box snapshots for debugging leader line detection."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from engvision.config import EngVisionConfig
from engvision.services.pdf_renderer import PdfRendererService
from engvision.services.bubble_detection import BubbleDetectionService
from engvision.services.leader_line_tracer import LeaderLineTracerService, CAPTURE_STEPS

PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sample_docs",
    "WFRD_Sample_Dimentional_Analysis.pdf",
)
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "Output", "debug_captures")
os.makedirs(DEBUG_DIR, exist_ok=True)

config = EngVisionConfig(
    pdf_render_dpi=300,
    output_directory=os.path.join(os.path.dirname(__file__), "..", "Output"),
)

# Step 1: Render
renderer = PdfRendererService(300)
pages = renderer.render_all_pages(PDF_PATH)
page = pages[0]
img_h, img_w = page.shape[:2]
print(f"Page size: {img_w}x{img_h}")

# Step 2: Detect bubbles
detector = BubbleDetectionService(config)
bubbles = detector.detect_bubbles(page, page_number=1)
print(f"Detected {len(bubbles)} bubbles")

# Step 3: Trace leader lines
tracer = LeaderLineTracerService()
expanded = tracer.trace_and_expand(bubbles, page)

# Step 4: Dump debug output
overlay = page.copy()
captured = 0
no_dir = 0

for i, eb in enumerate(expanded):
    bb = eb["boundingBox"]
    bcx = bb["x"] + bb["width"] // 2
    bcy = bb["y"] + bb["height"] // 2
    r = bb["width"] // 2
    bnum = eb.get("bubbleNumber", i + 1)

    # Draw bubble circle on overlay
    cv2.circle(overlay, (bcx, bcy), r, (0, 255, 0), 1)
    cv2.putText(
        overlay, str(bnum),
        (bcx - 8, bcy - r - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
    )

    cap = eb.get("captureBox")
    ld = eb.get("leaderDirection")

    if cap is None:
        no_dir += 1
        cv2.putText(
            overlay, "NO_DIR",
            (bcx + r + 4, bcy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
        )
        print(f"  Bubble {bnum}: NO blue pixel found")
        continue

    cx1 = max(0, cap["x"])
    cy1 = max(0, cap["y"])
    cx2 = min(img_w, cap["x"] + cap["width"])
    cy2 = min(img_h, cap["y"] + cap["height"])

    # Draw initial capture box rectangle + arrow on overlay
    cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)
    cap_cx = (cx1 + cx2) // 2
    cap_cy = (cy1 + cy2) // 2
    cv2.arrowedLine(
        overlay, (bcx, bcy), (cap_cx, cap_cy),
        (255, 0, 255), 1, tipLength=0.15,
    )

    dx_val = ld["dx"] if ld else 0
    dy_val = ld["dy"] if ld else 0

    # Dump all progressive capture box sizes
    orig_bb = bubbles[i]["boundingBox"]
    orig_bcx = orig_bb["x"] + orig_bb["width"] // 2
    orig_bcy = orig_bb["y"] + orig_bb["height"] // 2
    orig_r = orig_bb["width"] // 2

    for cap_w, cap_h in CAPTURE_STEPS:
        step_cap = tracer.place_capture_box(
            orig_bcx, orig_bcy, orig_r, dx_val, dy_val,
            cap_w, cap_h, img_w, img_h,
        )
        sx1 = max(0, step_cap["x"])
        sy1 = max(0, step_cap["y"])
        sx2 = min(img_w, step_cap["x"] + step_cap["width"])
        sy2 = min(img_h, step_cap["y"] + step_cap["height"])
        if sx2 - sx1 > 0 and sy2 - sy1 > 0:
            step_crop = page[sy1:sy2, sx1:sx2]
            fname = os.path.join(
                DEBUG_DIR, f"capture_bubble_{bnum:03d}_{cap_w}x{cap_h}.png"
            )
            cv2.imwrite(fname, step_crop)
    captured += 1

    print(
        f"  Bubble {bnum}: dir=({dx_val:.2f},{dy_val:.2f}) "
            f"box=({cx1},{cy1})-({cx2},{cy2}) size={cx2-cx1}x{cy2-cy1}"
        )

# Save annotated overview
overview_path = os.path.join(DEBUG_DIR, "capture_boxes_overview.png")
cv2.imwrite(overview_path, overlay)

print(f"\nSaved {captured} capture crops, {no_dir} with no direction")
print(f"Debug output: {os.path.abspath(DEBUG_DIR)}")
print(f"Overview: {overview_path}")
