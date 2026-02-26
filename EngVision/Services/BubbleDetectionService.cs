using EngVision.Models;
using OpenCvSharp;

namespace EngVision.Services;

/// <summary>
/// Detects numbered bubbles (circle annotations) on a CAD drawing page using OpenCV.
/// Annotation bubbles are blue circles with a white interior containing a dark number.
/// PRIMARY strategy: find blue circles directly in color space (accuracy-first).
/// SECONDARY: grayscale HoughCircles with blue verification as backup.
/// </summary>
public class BubbleDetectionService
{
    private readonly EngVisionConfig _config;
    public bool Verbose { get; set; }

    public BubbleDetectionService(EngVisionConfig config)
    {
        _config = config;
    }

    public List<DetectedRegion> DetectBubbles(Mat pageImage, int pageNumber)
    {
        using var gray = new Mat();
        Cv2.CvtColor(pageImage, gray, ColorConversionCodes.BGR2GRAY);
        using var hsv = new Mat();
        Cv2.CvtColor(pageImage, hsv, ColorConversionCodes.BGR2HSV);

        var allCandidates = new List<(int Cx, int Cy, int Radius, string Source)>();

        // ═══════════════════════════════════════════════════════════════════
        // PRIMARY: Blue-first detection — find circles in the blue channel
        // ═══════════════════════════════════════════════════════════════════

        // Create blue mask
        using var blueMask = new Mat();
        Cv2.InRange(hsv, new Scalar(85, 25, 50), new Scalar(125, 255, 255), blueMask);

        // Morphological cleanup: close small gaps in circle outlines, then dilate slightly
        using var closeKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
        using var blueClosed = new Mat();
        Cv2.MorphologyEx(blueMask, blueClosed, MorphTypes.Close, closeKernel);

        // Method A: HoughCircles on blue mask (multiple passes)
        using var blueBlur = new Mat();
        Cv2.GaussianBlur(blueClosed, blueBlur, new Size(5, 5), 1.0);

        var bluePassParams = new (double dp, double minDist, double p1, double p2, int minR, int maxR)[]
        {
            (1.0, 18, 60, 12, 10, 28),  // fine, sensitive
            (1.2, 20, 80, 15, 10, 28),  // medium
            (1.5, 18, 50, 10, 8, 30),   // relaxed
            (1.0, 15, 40, 8, 8, 25),    // very sensitive
        };

        foreach (var (dp, minDist, p1, p2, minR, maxR) in bluePassParams)
        {
            var circles = Cv2.HoughCircles(blueBlur, HoughModes.Gradient,
                dp: dp, minDist: minDist, param1: p1, param2: p2,
                minRadius: minR, maxRadius: maxR);
            int before = allCandidates.Count;
            AddUnique(allCandidates, circles, "blue-hough");
            if (Verbose) Console.WriteLine($"  Blue HoughCircles (dp={dp},p2={p2}): {circles.Length} raw, {allCandidates.Count - before} new");
        }

        // Method B: Contour detection on blue mask
        using var dilateKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(2, 2));
        using var blueDilated = new Mat();
        Cv2.MorphologyEx(blueClosed, blueDilated, MorphTypes.Dilate, dilateKernel);

        Cv2.FindContours(blueDilated, out Point[][] blueContours, out _, RetrievalModes.List,
            ContourApproximationModes.ApproxSimple);

        int blueContourAdded = 0;
        foreach (var contour in blueContours)
        {
            if (contour.Length < 6) continue;
            double area = Cv2.ContourArea(contour);
            double perim = Cv2.ArcLength(contour, true);
            if (perim < 1) continue;
            double circularity = 4 * Math.PI * area / (perim * perim);
            if (circularity < 0.25) continue;  // Very lenient — partial arcs OK

            Cv2.MinEnclosingCircle(contour, out Point2f center, out float encR);
            if (encR < 8 || encR > 35) continue;

            int cx = (int)center.X, cy = (int)center.Y, r = (int)encR;
            if (!allCandidates.Any(e => Distance(e.Cx, e.Cy, cx, cy) < 15))
            {
                allCandidates.Add((cx, cy, r, "blue-contour"));
                blueContourAdded++;
            }
        }
        if (Verbose) Console.WriteLine($"  Blue contour candidates: {blueContourAdded}");

        int blueTotal = allCandidates.Count;
        Console.WriteLine($"  Blue-first candidates: {blueTotal}");

        // ═══════════════════════════════════════════════════════════════════
        // SECONDARY: Grayscale HoughCircles as backup for missed blue circles
        // ═══════════════════════════════════════════════════════════════════

        using var blurCoarse = new Mat();
        using var blurFine = new Mat();
        Cv2.GaussianBlur(gray, blurCoarse, new Size(9, 9), 2);
        Cv2.GaussianBlur(gray, blurFine, new Size(5, 5), 1.0);

        var grayPassParams = new (Mat img, double dp, double minDist, double p1, double p2, int minR, int maxR)[]
        {
            (blurCoarse, 1.2, 22, _config.HoughParam1, 23, _config.HoughMinRadius, _config.HoughMaxRadius),
            (blurFine, 1.0, 18, 100, 20, 8, 25),
            (blurCoarse, 1.5, 22, 80, 18, 10, 45),
            (blurFine, 1.0, 15, 80, 15, 10, 22),  // extra sensitive pass
        };

        int grayAdded = 0;
        foreach (var (img, dp, minDist, p1, p2, minR, maxR) in grayPassParams)
        {
            var circles = Cv2.HoughCircles(img, HoughModes.Gradient,
                dp: dp, minDist: minDist, param1: p1, param2: p2,
                minRadius: minR, maxRadius: maxR);
            int before = allCandidates.Count;
            AddUnique(allCandidates, circles, "gray-hough");
            grayAdded += allCandidates.Count - before;
        }
        Console.WriteLine($"  Grayscale backup candidates: {grayAdded}");

        // Contour-based backup
        int contourAdded = FindCircularContours(gray, allCandidates);
        Console.WriteLine($"  Contour backup candidates: {contourAdded}");
        Console.WriteLine($"  Total unique candidates: {allCandidates.Count}");

        // ═══════════════════════════════════════════════════════════════════
        // VERIFICATION: confirm each candidate is a real blue annotation bubble
        // ═══════════════════════════════════════════════════════════════════

        var verified = VerifyBubbles(allCandidates, gray, hsv);
        Console.WriteLine($"  Verified bubbles: {verified.Count}");

        // Sort top-to-bottom, left-to-right
        verified.Sort((a, b) =>
        {
            int rowDiff = a.Cy / 60 - b.Cy / 60;
            return rowDiff != 0 ? rowDiff : a.Cx.CompareTo(b.Cx);
        });

        var regions = new List<DetectedRegion>();
        int id = 1;
        foreach (var (cx, cy, radius) in verified)
        {
            regions.Add(new DetectedRegion
            {
                Id = id,
                PageNumber = pageNumber,
                Type = RegionType.Bubble,
                BoundingBox = new BoundingBox(
                    Math.Max(0, cx - radius),
                    Math.Max(0, cy - radius),
                    Math.Min(radius * 2, pageImage.Width - Math.Max(0, cx - radius)),
                    Math.Min(radius * 2, pageImage.Height - Math.Max(0, cy - radius))),
                BubbleNumber = id,
                Label = $"Bubble_{id}"
            });
            id++;
        }
        return regions;
    }

    /// <summary>
    /// For each detected bubble, expands the bounding box to include the associated
    /// CAD figure region that the bubble is annotating (simple padding fallback).
    /// </summary>
    public List<DetectedRegion> ExpandToFigureRegions(
        List<DetectedRegion> bubbles, Mat pageImage, int padding = -1)
    {
        if (padding < 0) padding = _config.BubbleContextPadding;
        var expanded = new List<DetectedRegion>();
        int imgW = pageImage.Width, imgH = pageImage.Height;

        foreach (var bubble in bubbles)
        {
            var bb = bubble.BoundingBox;
            int cx = bb.X + bb.Width / 2;
            int cy = bb.Y + bb.Height / 2;
            int x = Math.Max(0, cx - bb.Width / 2 - padding);
            int y = Math.Max(0, cy - bb.Height / 2 - padding);
            int w = Math.Min(bb.Width + 2 * padding, imgW - x);
            int h = Math.Min(bb.Height + 2 * padding, imgH - y);

            expanded.Add(bubble with
            {
                Type = RegionType.BubbleWithFigure,
                BoundingBox = new BoundingBox(x, y, w, h)
            });
        }
        return expanded;
    }

    /// <summary>
    /// Verification: confirm candidates are real annotation bubbles.
    /// Uses wide radius scanning and blue perimeter + interior checks.
    /// </summary>
    private List<(int Cx, int Cy, int Radius)> VerifyBubbles(
        List<(int Cx, int Cy, int Radius, string Source)> circles, Mat gray, Mat hsv)
    {
        var passed = new List<(int Cx, int Cy, int Radius, double Score)>();

        foreach (var (cx, cy, radius, source) in circles)
        {
            // Use generous margin so ROI can accommodate radius scanning
            int margin = 20;
            int roiR = Math.Max(radius, 20); // ensure ROI is at least 20px from center
            int x1 = Math.Max(0, cx - roiR - margin);
            int y1 = Math.Max(0, cy - roiR - margin);
            int x2 = Math.Min(gray.Width, cx + roiR + margin);
            int y2 = Math.Min(gray.Height, cy + roiR + margin);
            if (x2 - x1 < 20 || y2 - y1 < 20) continue;

            using var roi = new Mat(gray, new Rect(x1, y1, x2 - x1, y2 - y1));
            using var hsvRoi = new Mat(hsv, new Rect(x1, y1, x2 - x1, y2 - y1));
            int roiCx = cx - x1, roiCy = cy - y1;

            // ── Gate 1: BLUE perimeter check ──
            // Scan a wide range of radii (8 to 24) to find the blue ring
            using var blueMask = new Mat();
            Cv2.InRange(hsvRoi, new Scalar(85, 25, 50), new Scalar(125, 255, 255), blueMask);

            double bestBlueRatio = 0;
            int bestBlueR = radius;
            int scanMin = 8;
            int scanMax = Math.Min(Math.Min(roiCx, roiCy) - 2, 26);
            scanMax = Math.Min(scanMax, Math.Min(roi.Width - roiCx, roi.Height - roiCy) - 2);

            for (int scanR = scanMin; scanR <= scanMax; scanR++)
            {
                using var perimMask = new Mat(roi.Size(), MatType.CV_8UC1, Scalar.Black);
                Cv2.Circle(perimMask, new Point(roiCx, roiCy), scanR, Scalar.White, 3);
                int totalPerimPx = Cv2.CountNonZero(perimMask);
                if (totalPerimPx == 0) continue;

                using var bluePerim = new Mat();
                Cv2.BitwiseAnd(blueMask, perimMask, bluePerim);
                int bluePerimPx = Cv2.CountNonZero(bluePerim);
                double ratio = (double)bluePerimPx / totalPerimPx;
                if (ratio > bestBlueRatio)
                {
                    bestBlueRatio = ratio;
                    bestBlueR = scanR;
                }
            }

            if (Verbose && bestBlueRatio >= 0.10)
                Console.WriteLine($"    [{source}] ({cx},{cy}) r={radius}→{bestBlueR} blue={bestBlueRatio:P1}");

            if (bestBlueRatio < 0.15) continue;
            // Restrict valid radius to realistic bubble size (r=10-22)
            if (bestBlueR < 10 || bestBlueR > 22) continue;
            int verifiedR = bestBlueR;

            // ── Gate 2: Interior brightness ──
            using var innerMask = new Mat(roi.Size(), MatType.CV_8UC1, Scalar.Black);
            int innerR = Math.Max(3, (int)(verifiedR * 0.60));
            Cv2.Circle(innerMask, new Point(roiCx, roiCy), innerR, Scalar.White, -1);
            double brightness = Cv2.Mean(roi, innerMask).Val0;

            if (Verbose && bestBlueRatio >= 0.15)
                Console.WriteLine($"      brightness={brightness:F1} innerR={innerR}");

            if (brightness < 120) continue; // Loosened from 130

            // ── Gate 3: Dark text present — 3-60% ──
            using var innerBinary = new Mat();
            Cv2.Threshold(roi, innerBinary, 128, 255, ThresholdTypes.BinaryInv);
            using var darkMasked = new Mat();
            Cv2.BitwiseAnd(innerBinary, innerMask, darkMasked);
            int darkPixels = Cv2.CountNonZero(darkMasked);
            int totalPixels = Cv2.CountNonZero(innerMask);
            double darkRatio = totalPixels > 0 ? (double)darkPixels / totalPixels : 0;

            if (Verbose && brightness >= 120)
                Console.WriteLine($"      darkRatio={darkRatio:P1} ({darkPixels}/{totalPixels})");

            if (darkRatio < 0.03 || darkRatio > 0.60) continue;

            // ── Gate 4a: Continuous blue arc ≥ 180° ──
            // Real annotation bubbles always show at least half the circle even when
            // partially occluded by an overlapping bubble. Sample 72 points (every 5°)
            // at verifiedR ±1px tolerance, find the longest continuous blue run (wrapping).
            var blueAngles = new bool[72];
            for (int a = 0; a < 72; a++)
            {
                double angle = a * 5 * Math.PI / 180;
                for (int dr = -1; dr <= 1; dr++)
                {
                    int sr = verifiedR + dr;
                    int px = roiCx + (int)(sr * Math.Cos(angle));
                    int py = roiCy + (int)(sr * Math.Sin(angle));
                    if (px >= 0 && px < blueMask.Width && py >= 0 && py < blueMask.Height
                        && blueMask.At<byte>(py, px) > 0)
                    {
                        blueAngles[a] = true;
                        break;
                    }
                }
            }
            // Find longest continuous run (wrap-around: iterate twice)
            int longestRun = 0, currentRun = 0;
            for (int pass = 0; pass < 2; pass++)
            {
                for (int a = 0; a < 72; a++)
                {
                    if (blueAngles[a]) { currentRun++; longestRun = Math.Max(longestRun, currentRun); }
                    else currentRun = 0;
                }
            }
            longestRun = Math.Min(longestRun, 72); // cap at full circle
            int longestArcDeg = longestRun * 5;

            // ── Gate 4b: Triangle pointer detection ──
            // Real annotation bubbles have a small triangular pointer (leader arrow)
            // extending outward from the circle. Look for a localized cluster of blue
            // pixels just beyond the circle radius (verifiedR+2 to verifiedR+10).
            // Divide into 12 sectors (30° each). A triangle will concentrate blue
            // beyond the perimeter in 1-3 adjacent sectors.
            var outerSectorBlue = new int[12];
            int outerSampleTotal = 0;
            for (int a = 0; a < 72; a++)
            {
                double angle = a * 5 * Math.PI / 180;
                int sector = a / 6; // 72/12 = 6 samples per sector
                for (int rOff = 2; rOff <= 10; rOff++)
                {
                    int sr = verifiedR + rOff;
                    int px = roiCx + (int)(sr * Math.Cos(angle));
                    int py = roiCy + (int)(sr * Math.Sin(angle));
                    if (px >= 0 && px < blueMask.Width && py >= 0 && py < blueMask.Height
                        && blueMask.At<byte>(py, px) > 0)
                    {
                        outerSectorBlue[sector]++;
                        outerSampleTotal++;
                    }
                }
            }
            // The triangle pointer shows as a peak in 1-3 adjacent sectors.
            // Find the max sum of any 3 adjacent sectors (wrapping).
            int bestTriangleScore = 0;
            for (int s = 0; s < 12; s++)
            {
                int sum3 = outerSectorBlue[s] + outerSectorBlue[(s + 1) % 12] + outerSectorBlue[(s + 2) % 12];
                bestTriangleScore = Math.Max(bestTriangleScore, sum3);
            }
            bool hasTriangle = bestTriangleScore >= 8; // at least 8 blue hits in 3 adj sectors

            if (Verbose && darkRatio >= 0.03)
                Console.WriteLine($"      arc: {longestArcDeg}° continuous | triangle: {bestTriangleScore} (outer total={outerSampleTotal}) {(hasTriangle ? "YES" : "no")}");

            // Require ≥180° continuous arc AND a triangle pointer
            if (longestArcDeg < 180 || !hasTriangle) continue;

            double score = bestBlueRatio * 100 + (longestArcDeg / 360.0) * 50 + darkRatio * 20 + brightness * 0.1;
            passed.Add((cx, cy, verifiedR, score));
        }

        // Deduplicate: any two detections within 25px → keep higher score
        var result = new List<(int Cx, int Cy, int Radius)>();
        var sorted = passed.OrderByDescending(p => p.Score).ToList();
        var used = new bool[sorted.Count];
        const double minSeparation = 25;

        for (int i = 0; i < sorted.Count; i++)
        {
            if (used[i]) continue;
            var best = sorted[i];
            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (!used[j] && Distance(best.Cx, best.Cy, sorted[j].Cx, sorted[j].Cy) < minSeparation)
                    used[j] = true;
            }
            result.Add((best.Cx, best.Cy, best.Radius));
        }

        return result;
    }

    /// <summary>
    /// Find circular contours in grayscale as backup candidate source.
    /// </summary>
    private int FindCircularContours(Mat gray, List<(int Cx, int Cy, int Radius, string Source)> candidates)
    {
        int added = 0;
        int[] blockSizes = [31, 51, 71];

        foreach (int blockSize in blockSizes)
        {
            using var binary = new Mat();
            Cv2.AdaptiveThreshold(gray, binary, 255, AdaptiveThresholdTypes.GaussianC,
                ThresholdTypes.BinaryInv, blockSize, 10);

            using var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(2, 2));
            using var cleaned = new Mat();
            Cv2.MorphologyEx(binary, cleaned, MorphTypes.Close, kernel);

            Cv2.FindContours(cleaned, out Point[][] contours, out _, RetrievalModes.List,
                ContourApproximationModes.ApproxSimple);

            foreach (var contour in contours)
            {
                if (contour.Length < 8) continue;
                double area = Cv2.ContourArea(contour);
                double perimeter = Cv2.ArcLength(contour, true);
                if (perimeter < 1) continue;

                double circularity = 4 * Math.PI * area / (perimeter * perimeter);
                if (circularity < 0.65) continue;

                double radius = Math.Sqrt(area / Math.PI);
                if (radius < 8 || radius > 50) continue;

                Cv2.MinEnclosingCircle(contour, out Point2f center, out float encRadius);
                double fillRatio = area / (Math.PI * encRadius * encRadius);
                if (fillRatio < 0.6) continue;

                int cx = (int)center.X, cy = (int)center.Y, r = (int)encRadius;
                if (!candidates.Any(e => Distance(e.Cx, e.Cy, cx, cy) < Math.Max(e.Radius, r) * 0.6))
                {
                    candidates.Add((cx, cy, r, "gray-contour"));
                    added++;
                }
            }
        }
        return added;
    }

    private static void AddUnique(List<(int Cx, int Cy, int Radius, string Source)> dest,
        CircleSegment[] source, string label)
    {
        foreach (var c in source)
        {
            int cx = (int)c.Center.X, cy = (int)c.Center.Y, r = (int)c.Radius;
            if (!dest.Any(e => Distance(e.Cx, e.Cy, cx, cy) < 15))
                dest.Add((cx, cy, r, label));
        }
    }

    private static double Distance(int x1, int y1, int x2, int y2)
        => Math.Sqrt((x1 - x2) * (double)(x1 - x2) + (y1 - y2) * (double)(y1 - y2));
}
