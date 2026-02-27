# Bubble Detection

[Bubble Detection Flow](diagrams/bubble-detection.excalidraw.json)

## What Are "Bubbles"?

Bubbles are numbered circle annotations found on CAD/engineering drawings that reference measurement callouts or inspection points. Visually they appear as **blue circles with a white interior** containing a **dark number** (the callout ID). Each bubble has a **triangular pointer (leader line)** that extends from the circle edge toward the feature or dimension it references.

Detecting these reliably is challenging because drawings also contain many other circular shapes — holes, bolt patterns, dimension arcs — that must be rejected.

## Detection Strategy Overview

The algorithm uses a **two-phase candidate generation** followed by **multi-gate verification**:

1. **Phase 1 — Blue-first detection** (primary): exploits the fact that bubbles are drawn in blue ink. Generates candidates from the blue color channel using both Hough circle detection and contour analysis.
2. **Phase 2 — Grayscale backup** (secondary): catches bubbles that may have faded or shifted out of the blue range by running Hough and contour detection on the grayscale image.
3. **Verification gates**: every candidate — regardless of source — must pass five gates that confirm blue perimeter, white interior, dark text content, continuous arc, and triangular pointer.

## Phase 1: Blue-First Detection

### Color Isolation

The image is converted to [**HSV color space**](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html) and a blue mask is created. HSV separates color into three channels: **Hue** (the color itself, 0–180 in OpenCV), **Saturation** (color intensity, 0–255), and **Value** (brightness, 0–255). Unlike RGB, HSV makes it easy to isolate a specific color regardless of lighting conditions — we select pixels where the hue falls in the blue range (85–125) with sufficient saturation and brightness:

```python
hsv = cv2.cvtColor(page_image, cv2.COLOR_BGR2HSV)
blue_mask = cv2.inRange(hsv, np.array([85, 25, 50]), np.array([125, 255, 255]))
```

| Channel | Min | Max |
|---------|-----|-----|
| H (Hue) | 85 | 125 |
| S (Saturation) | 25 | 255 |
| V (Value) | 50 | 255 |

The mask is cleaned with a [**morphological close**](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) using a 3×3 elliptical kernel to fill small gaps in the blue rings. A morphological close is a **dilation** (expand white regions) followed by an **erosion** (shrink them back), which bridges small breaks without changing the overall shape.

### Method A: HoughCircles on Blue Mask

A 5×5 [**Gaussian blur**](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html) (σ=1.0) is applied to the closed blue mask to smooth out noise before circle detection. Gaussian blur replaces each pixel with the weighted average of its neighbors using a bell-curve distribution — this prevents noise from creating false circle edges.

Then **four parameter passes** of [`cv2.HoughCircles`](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad8c33f4a3b8e5d27) are run. The [Hough Circle Transform](https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html) works by looking at edge pixels in the image and voting for possible circle centers — for each edge pixel, it traces arcs of all plausible radii back to where the center would be. Where many arcs intersect (accumulate votes), a circle is detected. Key parameters:

- **`dp`** — inverse ratio of accumulator resolution to image resolution (1.0 = same resolution, higher = coarser but faster)
- **`minDist`** — minimum pixel distance between detected circle centers
- **`param1`** — higher threshold for the internal Canny edge detector (the lower threshold is half this)
- **`param2`** — accumulator vote threshold — lower values detect more circles but with more false positives
- **`minRadius` / `maxRadius`** — pixel range for accepted circle radii

| Pass | `dp` | `minDist` | `param1` | `param2` | `minRadius` | `maxRadius` |
|------|------|-----------|----------|----------|-------------|-------------|
| 1 | 1.0 | 18 | 60 | 12 | 10 | 28 |
| 2 | 1.2 | 20 | 80 | 15 | 10 | 28 |
| 3 | 1.5 | 18 | 50 | 10 | 8 | 30 |
| 4 | 1.0 | 15 | 40 | 8 | 8 | 25 |

Each pass progressively relaxes sensitivity to catch circles that earlier passes may miss. Detected circles are deduplicated — a new circle is only added if no existing candidate is within 15 px.

```python
for dp, min_dist, p1, p2, min_r, max_r in blue_pass_params:
    circles = cv2.HoughCircles(
        blue_blur, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=min_dist, param1=p1, param2=p2,
        minRadius=min_r, maxRadius=max_r,
    )
    _add_unique(all_candidates, circles, "blue-hough")
```

### Method B: Contour Detection on Blue Mask

The closed blue mask is [**dilated**](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) (expanded outward by 2 px using an elliptical kernel) to merge nearby fragments, then [**contours**](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html) — the outlines of connected white regions in the binary image — are extracted. Each contour must satisfy:

- **Minimum points**: ≥6
- **Circularity** ≥ 0.25 — computed as `4π × area / perimeter²`. A perfect circle scores 1.0; a square scores ~0.785; values below 0.25 indicate a highly irregular shape that is unlikely to be a bubble.
- **Enclosing radius**: 8–35 px — [`cv2.minEnclosingCircle`](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html) finds the smallest circle that fully contains the contour.
- **Uniqueness**: no existing candidate within 15 px

```python
circularity = 4 * math.pi * area / (perim * perim)
if circularity < 0.25:
    continue
(cx_f, cy_f), enc_r = cv2.minEnclosingCircle(contour)
if enc_r < 8 or enc_r > 35:
    continue
```

## Phase 2: Grayscale Backup

For drawings where bubble ink has faded or color reproduction is poor, a grayscale path provides additional candidates.

### Gaussian Blur at Two Scales

| Scale | Kernel | σ |
|-------|--------|---|
| Coarse | 9×9 | 2.0 |
| Fine | 5×5 | 1.0 |

### HoughCircles Passes (Grayscale)

The same [Hough Circle Transform](#method-a-houghcircles-on-blue-mask) used in Phase 1 is applied here against grayscale images instead of the blue mask. Four passes run against the blurred grayscale images. The first pass uses configurable thresholds from `EngVisionConfig`; the remaining passes use fixed parameters:

| Pass | Image | `dp` | `minDist` | `param1` | `param2` | `minRadius` | `maxRadius` |
|------|-------|------|-----------|----------|----------|-------------|-------------|
| 1 | Coarse | 1.2 | 22 | config | 23 | config | config |
| 2 | Fine | 1.0 | 18 | 100 | 20 | 8 | 25 |
| 3 | Coarse | 1.5 | 22 | 80 | 18 | 10 | 45 |
| 4 | Fine | 1.0 | 15 | 80 | 15 | 10 | 22 |

### Contour-Based Circular Detection

[Adaptive thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) is applied at **three block sizes** (31, 51, 71) with a constant offset of 10. Unlike global thresholding (one cutoff for the whole image), adaptive thresholding computes a per-pixel threshold from the local neighborhood mean, making it robust to uneven lighting across the scanned drawing. For each threshold result, contours are filtered by:

- **Minimum points**: ≥8
- **Circularity** ≥ 0.65
- **Estimated radius** (`√(area/π)`): 8–50 px
- **Fill ratio** (area vs. enclosing circle area) ≥ 0.6
- **Uniqueness**: no existing candidate within `max(existing_r, new_r) × 0.6` px

```python
for block_size in [31, 51, 71]:
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 10
    )
```

## Verification Gates

Every candidate circle (from any source) must pass all five gates. A local ROI is extracted around each candidate with a 20 px margin for analysis.

### Gate 1: Blue Perimeter Check

Scans radii from **8 to 26 px**, drawing a 3 px-thick perimeter ring at each radius and measuring overlap with the blue mask:

```python
for scan_r in range(scan_min, scan_max + 1):
    perim_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(perim_mask, (roi_cx, roi_cy), scan_r, 255, 3)
    blue_perim = cv2.bitwise_and(blue_mask, perim_mask)
    ratio = cv2.countNonZero(blue_perim) / cv2.countNonZero(perim_mask)
```

- **Threshold**: ≥ 15% of the perimeter must be blue (`best_blue_ratio < 0.15` → reject)
- The radius with the highest blue ratio is selected as the **verified radius**
- Verified radius must be **10–22 px** (rejects oversized or undersized matches)

### Gate 2: Interior Brightness

A filled circular mask at **60% of the verified radius** samples the interior:

```python
inner_r = max(3, int(verified_r * 0.60))
cv2.circle(inner_mask, (roi_cx, roi_cy), inner_r, 255, -1)
brightness = cv2.mean(roi, mask=inner_mask)[0]
if brightness < 120:
    continue
```

- **Threshold**: mean grayscale brightness ≥ **120** (confirms white-ish interior)
- Rejects candidates centered on dark regions (holes, filled circles)

### Gate 3: Dark Text Ratio

A binary threshold at **128** identifies dark pixels inside the interior mask:

```python
_, inner_binary = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
dark_masked = cv2.bitwise_and(inner_binary, inner_mask)
dark_ratio = dark_pixels / total_pixels
if dark_ratio < 0.03 or dark_ratio > 0.60:
    continue
```

- **Threshold**: dark pixels must be **3–60%** of the interior area
- Too few dark pixels → empty circle (not a numbered bubble)
- Too many dark pixels → filled shape or noise

### Gate 4a: Continuous Blue Arc

Samples **72 points** (every 5°) around the verified radius (±1 px tolerance). Finds the longest continuous run of blue-positive samples, wrapping around 360°:

```python
blue_angles = [False] * 72
for a in range(72):
    angle = a * 5 * math.pi / 180
    for dr in range(-1, 2):
        sr = verified_r + dr
        px = roi_cx + int(sr * math.cos(angle))
        py = roi_cy + int(sr * math.sin(angle))
        if blue_mask[py, px] > 0:
            blue_angles[a] = True
            break
```

- **Threshold**: longest continuous arc must be ≥ **180°** (36 consecutive samples × 5°)
- The wrap-around check iterates through the array twice to handle arcs crossing the 0°/360° boundary
- This gate distinguishes true bubble circles from partial arcs or dimension curves

### Gate 4b: Triangle Pointer Detection

Divides the **outer ring** (verified radius + 2 to verified radius + 10 px) into **12 sectors** (30° each). Counts blue pixels in each sector:

```python
outer_sector_blue = [0] * 12
for a in range(72):
    angle = a * 5 * math.pi / 180
    sector = a // 6
    for r_off in range(2, 11):
        sr = verified_r + r_off
        # ... sample blue_mask at (px, py)
        if blue_mask[py, px] > 0:
            outer_sector_blue[sector] += 1
```

The best **3 adjacent sectors** (wrapping) are summed:

- **Threshold**: combined blue hits ≥ **8**
- This confirms the triangular pointer/leader line extending from the bubble edge
- Both Gate 4a and Gate 4b must pass (`longest_arc_deg < 180 or not has_triangle` → reject)

## Deduplication

Verified candidates are scored with a composite formula:

```
score = blue_ratio × 100 + arc_coverage × 50 + dark_ratio × 20 + brightness × 0.1
```

Where `arc_coverage = longest_arc_deg / 360.0`.

```python
score = best_blue_ratio * 100 + (longest_arc_deg / 360.0) * 50 + dark_ratio * 20 + brightness * 0.1
```

Candidates are sorted by descending score. For any pair within **25 px**, only the higher-scoring candidate is kept:

```python
sorted_passed = sorted(passed, key=lambda p: -p[3])
for i in range(len(sorted_passed)):
    if used[i]:
        continue
    best = sorted_passed[i]
    for j in range(i + 1, len(sorted_passed)):
        if not used[j] and _distance(best[0], best[1], sorted_passed[j][0], sorted_passed[j][1]) < 25:
            used[j] = True
    result.append((best[0], best[1], best[2]))
```

Final results are sorted **top-to-bottom, left-to-right** (by row bands of 60 px, then by x-coordinate).

## Leader Line Tracing & Triangle Pointer Detection

After bubbles are detected, the `LeaderLineTracerService` determines the **direction** each bubble's leader line points toward on the drawing, then places progressive capture boxes along that direction for LLM validation.

### The Problem

Each bubble has a small **triangular arrowhead** physically attached to its perimeter that points toward the dimension annotation on the drawing. Detecting this triangle's direction is challenging because:

- Bubbles can **overlap** (e.g., bubbles 42/43 are only −5 px apart)
- Adjacent bubbles' blue perimeters **contaminate** each other's search regions
- The triangle is small (typically 5–15 px beyond the circle edge)

### Algorithm Evolution

The detection went through four iterations:

| Version | Method | Problem |
|---|---|---|
| v1 | Nearest blue pixel in annular ring | Adjacent bubbles contaminate ring |
| v2 | Shi-Tomasi corners on clean mask (all circles erased) | Overlapping circles sever triangle base |
| v3 | Harris corners + connected components | Circle erasure at r+1 left only 3 px of triangle |
| **v4 (current)** | **Flood-fill from perimeter + Harris corners** | **Handles all overlap cases** |

### Current Algorithm: Flood-Fill + Harris Corner Detection

Source: [`leader_line_tracer.py` → `_find_triangle_direction()`](../engvision-py/src/engvision/services/leader_line_tracer.py)

#### Step 1 — Extract ROI with target circle intact

A search region of `3 × bubble_radius` is extracted from the **original blue mask** (not a cleaned version). The target circle remains intact:

```python
search_r = b_radius * 3
roi = blue_mask[ry1:ry2, rx1:rx2].copy()
```

#### Step 2 — Erase only OTHER circles

Every circle except the target is erased at `radius + 1`:

```python
for j, (ocx, ocy, o_r) in enumerate(all_bubbles):
    if j == self_idx:
        continue
    cv2.circle(roi, (ox, oy), o_r + 1, 0, -1)
```

This is the key insight: by keeping the target circle intact, its triangle base stays connected to the circle perimeter even when adjacent bubbles overlap.

#### Step 3 — Find a blue seed on the perimeter

Since bubble interiors are **white** (not blue), we must seed the flood-fill from a blue pixel on the circle's perimeter ring:

```python
for angle_deg in range(0, 360, 5):
    angle_rad = math.radians(angle_deg)
    px = int(roi_cx + b_radius * math.cos(angle_rad))
    py = int(roi_cy + b_radius * math.sin(angle_rad))
    if roi[py, px] == 255:
        seed = (px, py)
        break
```

If exact perimeter fails, offsets of ±1 and ±2 pixels are tried.

#### Step 4 — Flood-fill to isolate the connected component

[`cv2.floodFill`](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga366aae45a6c1289b341d140c2c547f2f) from the seed pixel marks all blue pixels connected to the target circle. **Flood-fill** works like a paint-bucket tool — starting at a seed pixel, it spreads to all adjacent pixels of the same color, marking them with a new value (128 in this case). The result is the target circle's perimeter ring plus its attached triangle pointer, isolated from everything else:

```python
flood_img = roi.copy()
flood_mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), dtype=np.uint8)
cv2.floodFill(flood_img, flood_mask, seed, 128)
component = (flood_img == 128).astype(np.uint8) * 255
```

#### Step 5 — Erase the circle to leave only the triangle

The circle perimeter ring extends ~2 px beyond the nominal radius, so we erase at `r + 2` to remove it completely:

```python
cv2.circle(component, (roi_cx, roi_cy), b_radius + 2, 0, -1)
```

Only the triangle tip that protrudes beyond the perimeter ring survives.

#### Step 6 — Harris corner detection on the triangle

[**Harris corner detection**](https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html) identifies the sharp vertices of the remaining triangle pixels. The Harris algorithm examines small windows of pixels and computes how much the intensity changes when the window shifts in any direction — flat regions have no change, edges have change in one direction, and **corners** have significant change in all directions. The algorithm outputs a "response" score at each pixel: high positive values indicate corners.

```python
harris = cv2.cornerHarris(component, blockSize=3, ksize=3, k=0.04)
threshold = 0.01 * h_max
corner_ys, corner_xs = np.where(harris > threshold)
```

- **`blockSize=3`** — the 3×3 neighborhood window for computing intensity gradients
- **`ksize=3`** — size of the [Sobel kernel](https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html) used to compute image derivatives
- **`k=0.04`** — Harris sensitivity parameter (typical range 0.04–0.06); lower values detect more corners

Each corner is weighted by its Harris response and proximity to the circle edge. Corners too close to center (`< 0.3 × radius`) or too far (`> 3.0 × radius`) are rejected:

```python
edge_weight = max(0.1, 1.0 - abs(dist - b_radius) / (b_radius * 1.5))
weight = response * edge_weight
sum_dx += dx * weight
sum_dy += dy * weight
```

The weighted average gives a unit direction vector `(dx, dy)` from the bubble center toward the triangle pointer.

**Fallback**: If no Harris corners pass filtering, the centroid of remaining pixels is used as the direction.

### Performance

On the sample PDF (50 bubbles), the flood-fill + Harris approach runs in **~18–20 ms** total and detects all 50 directions correctly, including the problem bubbles (40–44, 46–47, 11–12).

---

## Progressive Capture Box Expansion

Once the leader direction is known, the pipeline places **capture boxes** along the ray for LLM validation. Instead of sending one large image, it starts small and progressively expands only if the LLM can't find the dimension.

### Capture Steps

| Step | Width × Height | Purpose |
|------|----------------|---------|
| 1 | 128 × 128 | Initial square capture |
| 2 | 256 × 128 | Expand width |
| 3 | 512 × 256 | Keep expanding |
| 4 | 1024 × 512 | Maximum capture area |

Boxes are **wider than tall** because dimension annotations typically extend horizontally.

### Fixed-Anchor Placement

All capture boxes share the **same center point**, computed from the smallest step. This prevents larger boxes from shifting away from the original capture area:

```python
anchor_half = CAPTURE_STEPS[0][0] // 2  # 64
box_dist = b_radius + anchor_half + 4
box_cx = bcx + dx * box_dist
box_cy = bcy + dy * box_dist
```

Larger boxes simply grow outward from this fixed anchor rather than moving further along the ray.

### Corner Constraint

A dot-product check ensures no corner of the capture box lands on the opposite side of the bubble center from the direction vector. If any corner violates this, the box is pushed further along the ray:

```python
min_dot = min((cx - bcx) * dx + (cy - bcy) * dy for cx, cy in corners)
if min_dot < 0:
    push = -min_dot + 1.0
    box_cx += dx * push
    box_cy += dy * push
```

### Pipeline Integration

The pipeline tries each step in order and stops when the LLM confirms a match:

1. Capture 128×128 → send to LLM → if match, done
2. Capture 256×128 → send to LLM → if match, done
3. Capture 512×256 → send to LLM → if match, done
4. Capture 1024×512 → send to LLM → use as best guess

If the LLM never confirms a match, the last validation's observed dimension is used as a best guess with the LLM's confidence score.

For the full processing flow from PDF ingestion through bubble detection to AI analysis, see [Pipeline Orchestration](pipeline.md). For system-level architecture including the .NET API and Python service integration, see [Architecture](architecture.md).
