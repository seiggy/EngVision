# Table OCR

## Overview

Pages 2+ of the CAD PDF contain dimensional analysis tables that map balloon numbers to dimension values. Extracting these mappings is a core step in the engineering vision pipeline.

Two parallel OCR tracks are used:

1. **[Tesseract](https://github.com/tesseract-ocr/tesseract) grid OCR** — morphological grid detection followed by per-cell Tesseract OCR (this document). Tesseract is an open-source OCR engine originally developed by HP and now maintained by Google; it converts images of text into machine-readable strings.
2. **Vision LLM validation** — validates table dimensions against the drawing (see [LLM Validation](llm-extraction.md))

Both tracks produce `balloon_number → dimension_text` dictionaries that are merged downstream in the [Pipeline Orchestration](pipeline.md) step.

---

## Table Region Detection (`table_detection.py`)

`TableDetectionService.detect_tables()` locates rectangular table regions on a page image using morphological line detection.

### Algorithm

1. [**Adaptive threshold**](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) — convert to grayscale, then binarize with mean adaptive threshold (block size 15, constant 10). Unlike a global threshold that uses one cutoff for the entire image, **adaptive thresholding** computes a different threshold for each pixel based on the mean intensity of its surrounding neighborhood (a 15×15 block here). This handles uneven lighting across the scanned drawing. The constant (10) is subtracted from the computed mean to fine-tune sensitivity:

   ```python
   binary = cv2.adaptiveThreshold(
       gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
   )
   ```

2. **Horizontal line detection** — [**morphological open**](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) with a wide kernel `(width/30, 1)`. A morphological open is an **erosion** (shrink white pixels) followed by a **dilation** (grow them back). With a very wide, 1-pixel-tall kernel, only horizontal structures at least `width/30` pixels long survive — all other shapes (text, small marks) are erased. This isolates the horizontal lines of the table grid:

   ```python
   width = max(binary.shape[1] // 30, 10)
   kernel_size = (width, 1)
   line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
   return cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_kernel)
   ```

3. **Vertical line detection** — same morphological open approach, but with a tall, 1-pixel-wide kernel `(1, height/30)` that only preserves vertical structures:

   ```python
   height = max(binary.shape[0] // 30, 10)
   kernel_size = (1, height)
   ```

4. **Combine & dilate** — add the two line masks, then [**dilate**](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) with a 5×5 rectangular kernel (3 iterations) to merge nearby line fragments. Dilation expands white regions outward — repeated 3 times, it bridges small gaps between line segments that should form a continuous table grid:

   ```python
   table_mask = cv2.add(horizontal, vertical)
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
   table_mask = cv2.dilate(table_mask, kernel, iterations=3)
   ```

5. **Contour extraction & filtering** — find [external contours](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html) (outlines of connected white regions), then keep only those meeting minimum size and aspect-ratio constraints:

   - Minimum width: `table_min_width` (default 200 px)
   - Minimum height: `table_min_height` (default 100 px)
   - Aspect ratio between 0.05 and 20

   ```python
   contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   for contour in contours:
       x, y, w, h = cv2.boundingRect(contour)
       if w < self._config.table_min_width or h < self._config.table_min_height:
           continue
       aspect = w / h if h > 0 else 0
       if aspect > 20 or aspect < 0.05:
           continue
   ```

### Fallback

If no table regions are detected, `get_full_page_region()` returns a single region covering the entire page image, ensuring downstream OCR always has something to process.

---

## Grid-Based Table OCR (`table_ocr.py`)

`TableOcrService.extract_balloon_dimensions()` reads the content of a detected table region by first finding its internal grid lines, then OCR-ing individual cells.

### Grid Detection

The internal grid uses slightly different morphological parameters than table-region detection to capture finer cell boundaries:

- **Horizontal lines**: kernel width = `image_width / 20` (min 50)
- **Vertical lines**: kernel height = `image_height / 40` (min 20)

```python
h_kernel_w = max(page_image.shape[1] // 20, 50)
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

v_kernel_h = max(page_image.shape[0] // 40, 20)
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_h))
v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
```

### Line Bound Extraction

`_extract_line_bounds()` scans each row (or column) of the line mask, counting white pixels sampled every 4th position. A row is classified as a line boundary when pixel count exceeds `crossLen / 16`:

```python
for i in range(length):
    count = 0
    for j in range(0, cross_len, 4):
        val = line_mask[i, j] if is_horizontal else line_mask[j, i]
        if val > 0:
            count += 1
    if count > cross_len // 16:
        if not bounds or i - bounds[-1] > 5:
            bounds.append(i)
        else:
            bounds[-1] = (bounds[-1] + i) // 2
```

Adjacent bounds within 5 px are merged by averaging, preventing duplicate boundaries from thick lines.

### Column Identification

The first 5 data rows are OCR'd to locate the **balloon** and **dimension** columns by header text:

- **Balloon column**: cell text contains `"BALLOON"` or both `"SN"` and `"NO"`
- **Dimension column**: cell text contains `"DIMENSION"`

If headers are not found, the service defaults to column 0 (balloon) and column 1 (dimension).

### Cell OCR Pipeline

Each cell is OCR'd through a multi-step preprocessing pipeline:

1. **Crop** — extract the cell with 2 px inset to avoid grid lines
2. **Grayscale** conversion
3. **3× upscale** using cubic interpolation
4. **Binary threshold** at intensity 160
5. **Pad** with a 10 px white border on all sides
6. **Tesseract PSM 7** (treat image as a single text line)

```python
cell = page_image[y1:y2, x1:x2]
gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
upscaled = cv2.resize(gray, (gray.shape[1] * 3, gray.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
_, binary = cv2.threshold(upscaled, 160, 255, cv2.THRESH_BINARY)
padded = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

config = f"--tessdata-dir {self._tess_data_path} --psm 7"
text = pytesseract.image_to_string(padded, lang="eng", config=config)
```

Balloon cell text is stripped to digits and validated in the range 1–99.

### Full-Page OCR Fallback

When grid detection fails (fewer than 3 row or column bounds), `_extract_via_full_page_ocr()` falls back to:

1. Grayscale the full page image
2. Run Tesseract in **PSM 3** (fully automatic page segmentation)
3. Parse output line by line — first whitespace-delimited token is the balloon number, second is the dimension value

```python
for line in text.split("\n"):
    parts = line.split()
    if len(parts) < 2:
        continue
    digits = re.sub(r"[^0-9]", "", parts[0])
    if digits:
        num = int(digits)
        if 1 <= num <= 99:
            result.setdefault(num, parts[1])
```

`setdefault` ensures the first occurrence wins when duplicates appear.

---

## Bubble Number OCR (`bubble_ocr.py`)

`BubbleOcrService` reads the number inside a [detected bubble](bubble-detection.md) crop. These numbers are used to correlate the physical annotation on page 1 with the table rows on pages 2+.

### Preprocessing

Bubble crops contain a blue circle annotation that must be removed before OCR:

1. **Blue circle removal** — convert to HSV, mask the blue range `(85–125 H, 25–255 S, 50–255 V)`, dilate the mask with a 3×3 elliptical kernel, then replace masked pixels with white:

   ```python
   hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
   blue_mask = cv2.inRange(hsv, np.array([85, 25, 50]), np.array([125, 255, 255]))
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
   dilated_blue = cv2.dilate(blue_mask, kernel, iterations=1)

   cleaned = src.copy()
   cleaned[dilated_blue > 0] = [255, 255, 255]
   ```

2. **Grayscale** conversion
3. **6× upscale** using cubic interpolation
4. **Adaptive threshold** — Gaussian method, block size 31, constant 10
5. **20 px white border** padding on all sides

```python
upscaled = cv2.resize(gray, (gray.shape[1] * 6, gray.shape[0] * 6), interpolation=cv2.INTER_CUBIC)
binary = cv2.adaptiveThreshold(
    upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
)
padded = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
```

### OCR

Tesseract runs in **PSM 8** (treat image as a single word):

```python
config = f"--tessdata-dir {self._tess_data_path} --psm 8"
text = pytesseract.image_to_string(processed, lang="eng", config=config)
```

### Number Parsing

`_parse_bubble_number()` applies common OCR-error substitutions before extracting digits:

| OCR character | Replaced with |
|---------------|---------------|
| `O`, `o`      | `0`           |
| `l`, `I`      | `1`           |
| `S`           | `5`           |
| `B`           | `8`           |
| `#`           | *(removed)*   |

```python
cleaned = (
    ocr_text
    .replace("#", "")
    .replace("O", "0").replace("o", "0")
    .replace("l", "1").replace("I", "1")
    .replace("S", "5").replace("B", "8")
    .replace(" ", "")
    .strip()
)
digits = re.sub(r"[^0-9]", "", cleaned)
```

The resulting integer is validated in the range **1–99**; anything outside that range returns `None`.

---

## Data Flow

The table OCR results and bubble OCR results are combined in the merge step of the [Pipeline Orchestration](pipeline.md):

```
Page 1                         Pages 2+
┌──────────────┐               ┌──────────────────────┐
│ Bubble       │               │ Table Detection      │
│ Detection    │               │ (table_detection.py) │
│   ↓          │               │   ↓                  │
│ Bubble OCR   │               │ Grid Table OCR       │
│ (bubble_ocr) │               │ (table_ocr.py)       │
│   ↓          │               │   ↓                  │
│ bubble_id →  │               │ balloon_num →        │
│ (x, y) loc   │               │ dimension_text       │
└──────┬───────┘               └──────────┬───────────┘
       │                                  │
       └──────────┬───────────────────────┘
                  ↓
          Merge: match bubble_id
          to table balloon_num
                  ↓
         Final dimension map
```

1. **[Bubble Detection](bubble-detection.md)** finds annotated circles on page 1 and crops them.
2. **Bubble OCR** (`bubble_ocr.py`) reads the number inside each crop.
3. **Table Detection** (`table_detection.py`) locates table regions on pages 2+.
4. **Grid Table OCR** (`table_ocr.py`) extracts `balloon_number → dimension_text` from each table.
5. The pipeline merges results by matching bubble numbers to table balloon numbers.

See also:
- [Bubble Detection](bubble-detection.md)
- [LLM Validation](llm-extraction.md)
- [Pipeline Orchestration](pipeline.md)
