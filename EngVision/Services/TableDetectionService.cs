using EngVision.Models;
using OpenCvSharp;

namespace EngVision.Services;

/// <summary>
/// Detects table regions on PDF pages using line detection and contour analysis.
/// </summary>
public class TableDetectionService
{
    private readonly EngVisionConfig _config;

    public TableDetectionService(EngVisionConfig config)
    {
        _config = config;
    }

    /// <summary>
    /// Detects table regions on the given page image.
    /// Tables are identified by detecting a grid of horizontal and vertical lines.
    /// </summary>
    public List<DetectedRegion> DetectTables(Mat pageImage, int pageNumber)
    {
        var regions = new List<DetectedRegion>();
        using var gray = new Mat();
        using var binary = new Mat();

        Cv2.CvtColor(pageImage, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.AdaptiveThreshold(gray, binary, 255,
            AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 15, 10);

        // Detect horizontal lines
        using var horizontal = DetectLines(binary, isHorizontal: true);
        // Detect vertical lines
        using var vertical = DetectLines(binary, isHorizontal: false);

        // Combine horizontal and vertical lines to find table structure
        using var tableMask = new Mat();
        Cv2.Add(horizontal, vertical, tableMask);

        // Dilate to connect nearby lines
        var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
        Cv2.Dilate(tableMask, tableMask, kernel, iterations: 3);

        // Find contours of combined table regions
        Cv2.FindContours(tableMask, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        int id = 1;
        foreach (var contour in contours)
        {
            var rect = Cv2.BoundingRect(contour);

            // Filter: tables must be reasonably large
            if (rect.Width < _config.TableMinWidth || rect.Height < _config.TableMinHeight)
                continue;

            // Filter: table must have a reasonable aspect ratio (not a thin line)
            double aspect = (double)rect.Width / rect.Height;
            if (aspect > 20 || aspect < 0.05)
                continue;

            regions.Add(new DetectedRegion
            {
                Id = id,
                PageNumber = pageNumber,
                Type = RegionType.TableRegion,
                BoundingBox = new BoundingBox(rect.X, rect.Y, rect.Width, rect.Height),
                Label = $"Table_P{pageNumber}_{id}"
            });
            id++;
        }

        Console.WriteLine($"  Page {pageNumber}: detected {regions.Count} table region(s)");
        return regions;
    }

    /// <summary>
    /// Detects a full-page region (fallback if table detection doesn't find clear structure).
    /// </summary>
    public DetectedRegion GetFullPageRegion(Mat pageImage, int pageNumber)
    {
        return new DetectedRegion
        {
            Id = 1,
            PageNumber = pageNumber,
            Type = RegionType.FullPage,
            BoundingBox = new BoundingBox(0, 0, pageImage.Width, pageImage.Height),
            Label = $"FullPage_{pageNumber}"
        };
    }

    private Mat DetectLines(Mat binary, bool isHorizontal)
    {
        var result = new Mat();

        // Create morphological kernel for line detection
        Size kernelSize;
        if (isHorizontal)
        {
            int width = Math.Max(binary.Width / 30, 10);
            kernelSize = new Size(width, 1);
        }
        else
        {
            int height = Math.Max(binary.Height / 30, 10);
            kernelSize = new Size(1, height);
        }

        var lineKernel = Cv2.GetStructuringElement(MorphShapes.Rect, kernelSize);
        Cv2.MorphologyEx(binary, result, MorphTypes.Open, lineKernel);

        return result;
    }
}
