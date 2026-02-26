using EngVision.Models;
using OpenCvSharp;

namespace EngVision.Services;

/// <summary>
/// Exports detected regions as cropped images and generates metadata.
/// </summary>
public class SegmentExportService
{
    private readonly string _outputDir;

    public SegmentExportService(string outputDir)
    {
        _outputDir = outputDir;

        // Clean previous output
        if (Directory.Exists(outputDir))
            Directory.Delete(outputDir, recursive: true);

        Directory.CreateDirectory(outputDir);
        Directory.CreateDirectory(Path.Combine(outputDir, "bubbles"));
        Directory.CreateDirectory(Path.Combine(outputDir, "tables"));
        Directory.CreateDirectory(Path.Combine(outputDir, "pages"));
        Directory.CreateDirectory(Path.Combine(outputDir, "debug"));
    }

    /// <summary>
    /// Crops and saves all detected regions from a page image.
    /// Returns updated regions with CroppedImagePath set.
    /// </summary>
    public List<DetectedRegion> ExportRegions(List<DetectedRegion> regions, Mat pageImage)
    {
        var exported = new List<DetectedRegion>();

        foreach (var region in regions)
        {
            var bb = region.BoundingBox;
            // Clamp to image bounds
            int x = Math.Max(0, bb.X);
            int y = Math.Max(0, bb.Y);
            int w = Math.Min(bb.Width, pageImage.Width - x);
            int h = Math.Min(bb.Height, pageImage.Height - y);

            if (w <= 0 || h <= 0) continue;

            using var cropped = new Mat(pageImage, new Rect(x, y, w, h));

            string subDir = region.Type switch
            {
                RegionType.Bubble or RegionType.BubbleWithFigure => "bubbles",
                RegionType.TableRegion => "tables",
                _ => "pages"
            };

            string filename = $"{region.Label ?? $"region_{region.Id}"}_p{region.PageNumber}.png";
            string outputPath = Path.Combine(_outputDir, subDir, filename);
            Cv2.ImWrite(outputPath, cropped);

            exported.Add(region with { CroppedImagePath = outputPath });
        }

        return exported;
    }

    /// <summary>
    /// Saves a debug visualization of all detected regions overlaid on the page image.
    /// </summary>
    public string SaveDebugVisualization(List<DetectedRegion> regions, Mat pageImage, int pageNumber, string suffix = "")
    {
        using var debug = pageImage.Clone();

        foreach (var region in regions)
        {
            var bb = region.BoundingBox;
            var color = region.Type switch
            {
                RegionType.Bubble => new Scalar(0, 0, 255),           // Red
                RegionType.BubbleWithFigure => new Scalar(0, 255, 0), // Green
                RegionType.TableRegion => new Scalar(255, 0, 0),      // Blue
                _ => new Scalar(255, 255, 0)                           // Cyan
            };

            Cv2.Rectangle(debug, new Rect(bb.X, bb.Y, bb.Width, bb.Height), color, 3);

            string label = region.Label ?? $"#{region.Id}";
            Cv2.PutText(debug, label, new Point(bb.X, bb.Y - 10),
                HersheyFonts.HersheySimplex, 0.7, color, 2);
        }

        string path = Path.Combine(_outputDir, "debug", $"page_{pageNumber}_detections{suffix}.png");
        Cv2.ImWrite(path, debug);
        Console.WriteLine($"  Debug visualization saved: {path}");
        return path;
    }

    /// <summary>
    /// Saves the full rendered page as a PNG.
    /// </summary>
    public string SaveFullPage(Mat pageImage, int pageNumber)
    {
        string path = Path.Combine(_outputDir, "pages", $"page_{pageNumber}.png");
        Cv2.ImWrite(path, pageImage);
        return path;
    }
}
