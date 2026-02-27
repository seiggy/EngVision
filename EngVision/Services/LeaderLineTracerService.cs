using EngVision.Models;
using OpenCvSharp;

namespace EngVision.Services;

/// <summary>
/// Finds the leader-line direction from each bubble by detecting the triangular
/// arrowhead pointer using flood-fill connectivity + Harris corner detection.
///
/// Instead of erasing ALL circles (which severs the triangle base on overlapping
/// bubbles), we keep the target circle intact, erase only OTHER circles, then
/// flood-fill from the target's perimeter to isolate exactly the blue pixels
/// connected to it.  After removing the circle interior, Harris corner detection
/// on the remaining triangle pixels gives the direction.
/// </summary>
public class LeaderLineTracerService
{
    /// Progressive capture box sizes (width, height) — wider than tall.
    /// Pipeline tries each in order; stops when the LLM confirms a match.
    public static readonly (int Width, int Height)[] CaptureSteps =
    [
        (128, 128),   // start square
        (256, 128),   // expand width
        (512, 256),   // keep expanding
        (1024, 512),  // max: 1024 px wide
    ];

    /// <summary>
    /// For each bubble, find the leader-line direction via triangle pointer
    /// detection and produce an expanded region plus a focused capture box.
    /// </summary>
    public List<DetectedRegion> TraceAndExpand(List<DetectedRegion> bubbles, Mat pageImage)
    {
        using var hsv = new Mat();
        Cv2.CvtColor(pageImage, hsv, ColorConversionCodes.BGR2HSV);
        using var blueMask = new Mat();
        Cv2.InRange(hsv, new Scalar(85, 25, 50), new Scalar(125, 255, 255), blueMask);

        int imgW = pageImage.Width, imgH = pageImage.Height;

        // Collect bubble geometry
        var bubbleInfo = bubbles.Select(b =>
        {
            var bb = b.BoundingBox;
            return (Cx: bb.X + bb.Width / 2, Cy: bb.Y + bb.Height / 2, R: bb.Width / 2);
        }).ToList();

        var expanded = new List<DetectedRegion>();
        for (int i = 0; i < bubbles.Count; i++)
        {
            var bubble = bubbles[i];
            var (bcx, bcy, bRadius) = bubbleInfo[i];
            var bb = bubble.BoundingBox;

            var direction = FindTriangleDirection(
                blueMask, bcx, bcy, bRadius, bubbleInfo, i, imgW, imgH);

            if (direction is null)
            {
                expanded.Add(bubble with
                {
                    Type = RegionType.BubbleWithFigure,
                    CaptureBox = null,
                    LeaderDirection = null
                });
                continue;
            }

            var (dx, dy) = direction.Value;

            var (initW, initH) = CaptureSteps[0];
            var capBox = PlaceCaptureBox(bcx, bcy, bRadius, dx, dy, initW, initH, imgW, imgH);

            int minX = Math.Min(bb.X, capBox.X);
            int minY = Math.Min(bb.Y, capBox.Y);
            int maxX = Math.Max(bb.X + bb.Width, capBox.X + capBox.Width);
            int maxY = Math.Max(bb.Y + bb.Height, capBox.Y + capBox.Height);

            expanded.Add(bubble with
            {
                Type = RegionType.BubbleWithFigure,
                BoundingBox = new BoundingBox(minX, minY, maxX - minX, maxY - minY),
                CaptureBox = capBox,
                LeaderDirection = (dx, dy)
            });
        }

        return expanded;
    }

    /// <summary>
    /// Finds the triangle pointer direction using flood-fill connectivity + Harris corners.
    ///
    /// 1. Extract ROI from the original blue mask (target circle intact).
    /// 2. Erase only OTHER circles.
    /// 3. Flood-fill from a blue seed on the target's perimeter to find
    ///    the connected component (circle + triangle).
    /// 4. Erase the circle at r+2 to remove the full perimeter ring.
    /// 5. Harris corner detection on the remaining triangle pixels.
    /// </summary>
    private static (double Dx, double Dy)? FindTriangleDirection(
        Mat blueMask,
        int bcx, int bcy, int bRadius,
        List<(int Cx, int Cy, int R)> allBubbles,
        int selfIdx,
        int imgW, int imgH)
    {
        int searchR = bRadius * 3;
        int rx1 = Math.Max(0, bcx - searchR);
        int ry1 = Math.Max(0, bcy - searchR);
        int rx2 = Math.Min(imgW, bcx + searchR);
        int ry2 = Math.Min(imgH, bcy + searchR);
        if (rx2 - rx1 < 10 || ry2 - ry1 < 10) return null;

        // Work on a copy — erase OTHER circles but keep the target intact
        using var roi = new Mat(blueMask, new Rect(rx1, ry1, rx2 - rx1, ry2 - ry1)).Clone();
        int roiCx = bcx - rx1;
        int roiCy = bcy - ry1;

        for (int j = 0; j < allBubbles.Count; j++)
        {
            if (j == selfIdx) continue;
            var (ocx, ocy, oR) = allBubbles[j];
            int ox = ocx - rx1;
            int oy = ocy - ry1;
            if (ox > -oR * 2 && ox < roi.Width + oR * 2 &&
                oy > -oR * 2 && oy < roi.Height + oR * 2)
            {
                Cv2.Circle(roi, new Point(ox, oy), oR + 1, Scalar.Black, -1);
            }
        }

        // Find a blue seed pixel on the target circle's perimeter
        Point? seed = null;
        for (int angleDeg = 0; angleDeg < 360; angleDeg += 5)
        {
            double angleRad = angleDeg * Math.PI / 180.0;
            int px = (int)(roiCx + bRadius * Math.Cos(angleRad));
            int py = (int)(roiCy + bRadius * Math.Sin(angleRad));
            if (px >= 0 && px < roi.Width && py >= 0 && py < roi.Height &&
                roi.At<byte>(py, px) == 255)
            {
                seed = new Point(px, py);
                break;
            }
        }

        // Try slightly inside/outside the perimeter if exact perimeter failed
        if (seed is null)
        {
            foreach (int offset in new[] { -1, 1, -2, 2 })
            {
                for (int angleDeg = 0; angleDeg < 360; angleDeg += 5)
                {
                    double angleRad = angleDeg * Math.PI / 180.0;
                    int px = (int)(roiCx + (bRadius + offset) * Math.Cos(angleRad));
                    int py = (int)(roiCy + (bRadius + offset) * Math.Sin(angleRad));
                    if (px >= 0 && px < roi.Width && py >= 0 && py < roi.Height &&
                        roi.At<byte>(py, px) == 255)
                    {
                        seed = new Point(px, py);
                        break;
                    }
                }
                if (seed is not null) break;
            }
        }

        if (seed is null) return null;

        // Flood-fill from the perimeter seed to find the connected component
        using var floodImg = roi.Clone();
        using var floodMask = new Mat(roi.Height + 2, roi.Width + 2, MatType.CV_8UC1, Scalar.Black);
        Cv2.FloodFill(floodImg, floodMask, seed.Value, new Scalar(128));

        // Extract only the flooded component (target circle + its triangle)
        using var component = new Mat();
        Cv2.Compare(floodImg, new Scalar(128), component, CmpTypes.EQ);

        // Erase the circle including its full perimeter stroke at r+2
        Cv2.Circle(component, new Point(roiCx, roiCy), bRadius + 2, Scalar.Black, -1);

        int remaining = Cv2.CountNonZero(component);
        if (remaining < 3) return null;

        // Harris corner detection on the triangle
        using var harrisMat = new Mat();
        Cv2.CornerHarris(component, harrisMat, blockSize: 3, ksize: 3, k: 0.04);

        double hMax;
        Cv2.MinMaxLoc(harrisMat, out _, out hMax);

        if (hMax <= 0)
        {
            // Fallback: centroid of remaining pixels
            return CentroidDirection(component, roiCx, roiCy);
        }

        double threshold = 0.01 * hMax;
        double sumDx = 0, sumDy = 0, totalWeight = 0;

        {
            var indexer = harrisMat.GetGenericIndexer<float>();
            for (int y = 0; y < harrisMat.Height; y++)
            {
                for (int x = 0; x < harrisMat.Width; x++)
                {
                    float response = indexer[y, x];
                    if (response <= threshold) continue;

                    double cdx = x - roiCx;
                    double cdy = y - roiCy;
                    double dist = Math.Sqrt(cdx * cdx + cdy * cdy);

                    if (dist < bRadius * 0.3 || dist > bRadius * 3.0) continue;

                    double edgeWeight = Math.Max(0.1, 1.0 - Math.Abs(dist - bRadius) / (bRadius * 1.5));
                    double weight = response * edgeWeight;

                    sumDx += cdx * weight;
                    sumDy += cdy * weight;
                    totalWeight += weight;
                }
            }
        }

        if (totalWeight < 1e-6)
            return CentroidDirection(component, roiCx, roiCy);

        double length = Math.Sqrt(sumDx * sumDx + sumDy * sumDy);
        if (length < 1) return null;
        return (sumDx / length, sumDy / length);
    }

    /// <summary>
    /// Fallback direction from centroid of remaining non-zero pixels.
    /// </summary>
    private static (double Dx, double Dy)? CentroidDirection(Mat mask, int cx, int cy)
    {
        double sumX = 0, sumY = 0;
        int count = 0;
        var indexer = mask.GetGenericIndexer<byte>();
        for (int y = 0; y < mask.Height; y++)
        {
            for (int x = 0; x < mask.Width; x++)
            {
                if (indexer[y, x] > 0)
                {
                    sumX += x;
                    sumY += y;
                    count++;
                }
            }
        }
        if (count == 0) return null;
        double dx = sumX / count - cx;
        double dy = sumY / count - cy;
        double length = Math.Sqrt(dx * dx + dy * dy);
        if (length < 1) return null;
        return (dx / length, dy / length);
    }

    /// <summary>
    /// Places a width×height capture box along the ray (dx, dy) from the bubble centre.
    /// The box centre is at a fixed distance from the bubble edge (anchored to the
    /// initial 128×128 step), so larger boxes expand outward from the same position
    /// rather than shifting away.
    /// </summary>
    public static BoundingBox PlaceCaptureBox(
        int bcx, int bcy, int bRadius,
        double dx, double dy, int width, int height,
        int imgW, int imgH)
    {
        int halfW = width / 2;
        int halfH = height / 2;

        // Fixed anchor: centre is always at bubble_edge + 68px along the ray,
        // matching the 128×128 step.  Larger boxes just grow around this point.
        int anchorHalf = CaptureSteps[0].Width / 2; // 64
        double boxDist = bRadius + anchorHalf + 4;
        double boxCx = bcx + dx * boxDist;
        double boxCy = bcy + dy * boxDist;

        // Check the dot-product constraint for every corner
        (double cx, double cy)[] corners =
        [
            (boxCx - halfW, boxCy - halfH),
            (boxCx + halfW, boxCy - halfH),
            (boxCx - halfW, boxCy + halfH),
            (boxCx + halfW, boxCy + halfH),
        ];
        double minDot = corners.Min(c => (c.cx - bcx) * dx + (c.cy - bcy) * dy);

        if (minDot < 0)
        {
            // Push box further along the ray so every corner is past centre
            double push = -minDot + 1.0;
            boxCx += dx * push;
            boxCy += dy * push;
        }

        // Clamp to image bounds
        int x1 = Math.Max(0, (int)(boxCx - halfW));
        int y1 = Math.Max(0, (int)(boxCy - halfH));
        int x2 = Math.Min(imgW, (int)(boxCx + halfW));
        int y2 = Math.Min(imgH, (int)(boxCy + halfH));

        return new BoundingBox(x1, y1, x2 - x1, y2 - y1);
    }
}
