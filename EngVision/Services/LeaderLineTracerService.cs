using EngVision.Models;
using OpenCvSharp;

namespace EngVision.Services;

/// <summary>
/// Traces leader lines from detected bubbles to find the associated dimension text,
/// then builds bounding boxes that encompass: bubble + leader line + text region.
/// </summary>
public class LeaderLineTracerService
{
    private const int SearchRadius = 400;    // max pixels to search for leader line endpoint
    private const int TextSearchRadius = 200; // how far beyond leader endpoint to look for text
    private const int MinLeaderLength = 15;   // minimum leader line length in pixels
    private const int FinalPadding = 30;      // padding around the final composite bounding box

    /// <summary>
    /// For each bubble, traces the leader line, finds the associated text region,
    /// and returns an expanded DetectedRegion encompassing all of it.
    /// </summary>
    public List<DetectedRegion> TraceAndExpand(List<DetectedRegion> bubbles, Mat pageImage)
    {
        using var gray = new Mat();
        using var edges = new Mat();
        Cv2.CvtColor(pageImage, gray, ColorConversionCodes.BGR2GRAY);

        // Build a text mask: morphological operations to find text clusters
        using var textMask = BuildTextMask(gray);

        // Build an edge image for leader line detection
        Cv2.Canny(gray, edges, 50, 150);

        var expanded = new List<DetectedRegion>();
        int imgW = pageImage.Width, imgH = pageImage.Height;

        foreach (var bubble in bubbles)
        {
            var bb = bubble.BoundingBox;
            int bcx = bb.X + bb.Width / 2;
            int bcy = bb.Y + bb.Height / 2;
            int bRadius = bb.Width / 2;

            // Step 1: Find the leader line direction by analyzing lines near the bubble
            var leaderEndpoint = FindLeaderEndpoint(edges, bcx, bcy, bRadius, imgW, imgH);

            // Step 2: Find nearby text region(s) in the leader direction
            Rect textRect;
            if (leaderEndpoint.HasValue)
            {
                var (lx, ly) = leaderEndpoint.Value;
                textRect = FindTextRegionNear(textMask, lx, ly, bcx, bcy, imgW, imgH);
            }
            else
            {
                // No leader line found — search radially for nearest text cluster
                textRect = FindNearestTextRegion(textMask, bcx, bcy, imgW, imgH);
            }

            // Step 3: Build composite bounding box: bubble + leader line + text
            int minX = bb.X;
            int minY = bb.Y;
            int maxX = bb.X + bb.Width;
            int maxY = bb.Y + bb.Height;

            if (leaderEndpoint.HasValue)
            {
                minX = Math.Min(minX, leaderEndpoint.Value.X - 5);
                minY = Math.Min(minY, leaderEndpoint.Value.Y - 5);
                maxX = Math.Max(maxX, leaderEndpoint.Value.X + 5);
                maxY = Math.Max(maxY, leaderEndpoint.Value.Y + 5);
            }

            if (textRect.Width > 0 && textRect.Height > 0)
            {
                minX = Math.Min(minX, textRect.X);
                minY = Math.Min(minY, textRect.Y);
                maxX = Math.Max(maxX, textRect.X + textRect.Width);
                maxY = Math.Max(maxY, textRect.Y + textRect.Height);
            }

            // Apply final padding and clamp
            minX = Math.Max(0, minX - FinalPadding);
            minY = Math.Max(0, minY - FinalPadding);
            maxX = Math.Min(imgW, maxX + FinalPadding);
            maxY = Math.Min(imgH, maxY + FinalPadding);

            expanded.Add(bubble with
            {
                Type = RegionType.BubbleWithFigure,
                BoundingBox = new BoundingBox(minX, minY, maxX - minX, maxY - minY)
            });
        }

        return expanded;
    }

    /// <summary>
    /// Finds where the leader line from a bubble points to, using HoughLinesP
    /// in the annular region just outside the bubble circle.
    /// </summary>
    private (int X, int Y)? FindLeaderEndpoint(Mat edges, int bcx, int bcy, int bRadius, int imgW, int imgH)
    {
        // Search in a region around the bubble for lines
        int searchPad = SearchRadius;
        int rx1 = Math.Max(0, bcx - searchPad);
        int ry1 = Math.Max(0, bcy - searchPad);
        int rx2 = Math.Min(imgW, bcx + searchPad);
        int ry2 = Math.Min(imgH, bcy + searchPad);
        if (rx2 - rx1 < 20 || ry2 - ry1 < 20) return null;

        using var roiEdges = new Mat(edges, new Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));

        // Mask out the bubble interior so we only see lines leaving the bubble
        using var mask = new Mat(roiEdges.Size(), MatType.CV_8UC1, new Scalar(255));
        int roiBcx = bcx - rx1, roiBcy = bcy - ry1;
        Cv2.Circle(mask, roiBcx, roiBcy, bRadius + 3, Scalar.Black, -1);
        using var maskedEdges = new Mat();
        Cv2.BitwiseAnd(roiEdges, mask, maskedEdges);

        // Detect lines in the region outside the bubble
        var lines = Cv2.HoughLinesP(maskedEdges,
            rho: 1, theta: Math.PI / 180, threshold: 15,
            minLineLength: MinLeaderLength, maxLineGap: 10);

        if (lines.Length == 0) return null;

        // Find the line segment whose closest point to the bubble center is just outside
        // the bubble, and whose far end is the leader endpoint
        (int X, int Y)? bestEndpoint = null;
        double bestScore = double.MaxValue;

        foreach (var seg in lines)
        {
            int x1 = seg.P1.X + rx1, y1 = seg.P1.Y + ry1;
            int x2 = seg.P2.X + rx1, y2 = seg.P2.Y + ry1;

            double d1 = Distance(x1, y1, bcx, bcy);
            double d2 = Distance(x2, y2, bcx, bcy);

            // One end should be near the bubble edge, the other end farther away
            double nearDist, farX, farY;
            if (d1 < d2)
            {
                nearDist = d1; farX = x2; farY = y2;
            }
            else
            {
                nearDist = d2; farX = x1; farY = y1;
            }

            // The near end should be close to the bubble edge (within bRadius + tolerance)
            double distFromEdge = Math.Abs(nearDist - bRadius);
            if (distFromEdge > bRadius * 0.8) continue; // too far from bubble edge

            double lineLen = Distance(x1, y1, x2, y2);
            if (lineLen < MinLeaderLength) continue;

            // Score: prefer lines that start near the bubble edge and are long
            double score = distFromEdge - lineLen * 0.3;
            if (score < bestScore)
            {
                bestScore = score;
                bestEndpoint = ((int)farX, (int)farY);
            }
        }

        return bestEndpoint;
    }

    /// <summary>
    /// Finds the text region near a leader line endpoint, searching in the direction
    /// away from the bubble center.
    /// </summary>
    private Rect FindTextRegionNear(Mat textMask, int leaderX, int leaderY, int bcx, int bcy, int imgW, int imgH)
    {
        // Direction from bubble to leader endpoint
        double dx = leaderX - bcx;
        double dy = leaderY - bcy;
        double len = Math.Sqrt(dx * dx + dy * dy);
        if (len < 1) return FindNearestTextRegion(textMask, bcx, bcy, imgW, imgH);

        double ndx = dx / len;
        double ndy = dy / len;

        // Search area: a rectangle centered on the leader endpoint, biased in the leader direction
        int searchW = TextSearchRadius;
        int searchH = TextSearchRadius;
        int sx = (int)(leaderX + ndx * 20); // slightly beyond the endpoint
        int sy = (int)(leaderY + ndy * 20);

        int rx1 = Math.Max(0, sx - searchW);
        int ry1 = Math.Max(0, sy - searchH);
        int rx2 = Math.Min(imgW, sx + searchW);
        int ry2 = Math.Min(imgH, sy + searchH);
        if (rx2 - rx1 < 10 || ry2 - ry1 < 10) return new Rect();

        using var roi = new Mat(textMask, new Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));
        Cv2.FindContours(roi, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        if (contours.Length == 0) return new Rect();

        // Find the text cluster closest to the leader endpoint (in the leader direction)
        Rect bestRect = new Rect();
        double bestDist = double.MaxValue;

        foreach (var contour in contours)
        {
            var rect = Cv2.BoundingRect(contour);
            if (rect.Width < 10 || rect.Height < 5) continue; // too small to be text

            // Transform to page coordinates
            var pageRect = new Rect(rect.X + rx1, rect.Y + ry1, rect.Width, rect.Height);
            int tcx = pageRect.X + pageRect.Width / 2;
            int tcy = pageRect.Y + pageRect.Height / 2;

            double dist = Distance(tcx, tcy, leaderX, leaderY);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestRect = pageRect;
            }
        }

        // If we found a text region, expand it to grab the full text line/block
        if (bestRect.Width > 0)
        {
            bestRect = ExpandToFullTextBlock(textMask, bestRect, imgW, imgH);
        }

        return bestRect;
    }

    /// <summary>
    /// Fallback: finds the nearest text cluster to the bubble when no leader line is detected.
    /// Searches in 8 radial directions and picks the closest text blob.
    /// </summary>
    private Rect FindNearestTextRegion(Mat textMask, int bcx, int bcy, int imgW, int imgH)
    {
        int searchR = SearchRadius;
        int rx1 = Math.Max(0, bcx - searchR);
        int ry1 = Math.Max(0, bcy - searchR);
        int rx2 = Math.Min(imgW, bcx + searchR);
        int ry2 = Math.Min(imgH, bcy + searchR);
        if (rx2 - rx1 < 10 || ry2 - ry1 < 10) return new Rect();

        using var roi = new Mat(textMask, new Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));
        Cv2.FindContours(roi, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        Rect bestRect = new Rect();
        double bestDist = double.MaxValue;

        foreach (var contour in contours)
        {
            var rect = Cv2.BoundingRect(contour);
            if (rect.Width < 15 || rect.Height < 8) continue;

            var pageRect = new Rect(rect.X + rx1, rect.Y + ry1, rect.Width, rect.Height);
            int tcx = pageRect.X + pageRect.Width / 2;
            int tcy = pageRect.Y + pageRect.Height / 2;

            double dist = Distance(tcx, tcy, bcx, bcy);

            // Skip text that's too close (probably the number inside the bubble itself)
            if (dist < 25) continue;

            if (dist < bestDist)
            {
                bestDist = dist;
                bestRect = pageRect;
            }
        }

        if (bestRect.Width > 0)
            bestRect = ExpandToFullTextBlock(textMask, bestRect, imgW, imgH);

        return bestRect;
    }

    /// <summary>
    /// Expands a text region rect to encompass the full connected text block
    /// (dimensions often have multiple text elements: value, tolerance, unit).
    /// </summary>
    private Rect ExpandToFullTextBlock(Mat textMask, Rect seed, int imgW, int imgH)
    {
        // Look for additional text blobs near the seed rect
        int expand = 60;
        int rx1 = Math.Max(0, seed.X - expand);
        int ry1 = Math.Max(0, seed.Y - expand);
        int rx2 = Math.Min(imgW, seed.X + seed.Width + expand);
        int ry2 = Math.Min(imgH, seed.Y + seed.Height + expand);

        using var roi = new Mat(textMask, new Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));

        // Dilate to merge nearby text elements
        var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(15, 5));
        using var dilated = new Mat();
        Cv2.Dilate(roi, dilated, kernel, iterations: 2);

        Cv2.FindContours(dilated, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        // Find the contour that overlaps with the seed rect
        var seedInRoi = new Rect(seed.X - rx1, seed.Y - ry1, seed.Width, seed.Height);
        Rect bestRect = seed;

        foreach (var contour in contours)
        {
            var rect = Cv2.BoundingRect(contour);
            // Check overlap with seed
            if (RectsOverlap(rect, seedInRoi))
            {
                var pageRect = new Rect(rect.X + rx1, rect.Y + ry1, rect.Width, rect.Height);
                // Union with current best
                int bx1 = Math.Min(bestRect.X, pageRect.X);
                int by1 = Math.Min(bestRect.Y, pageRect.Y);
                int bx2 = Math.Max(bestRect.X + bestRect.Width, pageRect.X + pageRect.Width);
                int by2 = Math.Max(bestRect.Y + bestRect.Height, pageRect.Y + pageRect.Height);
                bestRect = new Rect(bx1, by1, bx2 - bx1, by2 - by1);
            }
        }

        return bestRect;
    }

    /// <summary>
    /// Builds a binary mask of text regions using morphological operations.
    /// Text appears as clusters of small dark features on a light background.
    /// </summary>
    private Mat BuildTextMask(Mat gray)
    {
        var textMask = new Mat();

        // Threshold to get dark features
        using var binary = new Mat();
        Cv2.AdaptiveThreshold(gray, binary, 255,
            AdaptiveThresholdTypes.GaussianC, ThresholdTypes.BinaryInv, 15, 8);

        // Use morphological closing to merge characters within a text block
        var horizontalKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(12, 3));
        Cv2.MorphologyEx(binary, textMask, MorphTypes.Close, horizontalKernel);

        // Dilate slightly to connect nearby text elements
        var dilateKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(8, 4));
        Cv2.Dilate(textMask, textMask, dilateKernel, iterations: 1);

        return textMask;
    }

    private static double Distance(double x1, double y1, double x2, double y2)
        => Math.Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

    private static bool RectsOverlap(Rect a, Rect b)
        => a.X < b.X + b.Width && a.X + a.Width > b.X &&
           a.Y < b.Y + b.Height && a.Y + a.Height > b.Y;
}
