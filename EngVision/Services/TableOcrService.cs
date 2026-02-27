using OpenCvSharp;
using TesseractOCR;
using TesseractOCR.Enums;

namespace EngVision.Services;

/// <summary>
/// Extracts balloon# → dimension mapping from inspection report table pages
/// using grid detection + cell-by-cell Tesseract OCR.
/// </summary>
public class TableOcrService : ITableOcrService
{
    private readonly Engine _engine;

    public TableOcrService(string tessDataPath)
    {
        _engine = new Engine(tessDataPath, Language.English, EngineMode.Default);
    }

    /// <summary>
    /// Extract balloon number → dimension text mapping from a table page image.
    /// </summary>
    public Dictionary<int, string> ExtractBalloonDimensions(Mat pageImage)
    {
        var result = new Dictionary<int, string>();

        var (rowBounds, colBounds) = DetectGrid(pageImage);
        if (rowBounds.Count < 3 || colBounds.Count < 3)
        {
            Console.WriteLine("    Grid detection failed, falling back to full-page OCR");
            return ExtractViaFullPageOcr(pageImage);
        }

        Console.WriteLine($"    Grid: {rowBounds.Count - 1} rows x {colBounds.Count - 1} cols");

        // Find which columns are balloon# and dimension
        int balloonCol = -1, dimensionCol = -1;
        for (int r = 0; r < Math.Min(5, rowBounds.Count - 1); r++)
        {
            for (int c = 0; c < colBounds.Count - 1; c++)
            {
                string text = OcrCell(pageImage, rowBounds[r], rowBounds[r + 1], colBounds, c).ToUpperInvariant();
                if (text.Contains("BALLOON") || (text.Contains("SN") && text.Contains("NO")))
                    balloonCol = c;
                else if (text.Contains("DIMENSION"))
                    dimensionCol = c;
            }
            if (balloonCol >= 0 && dimensionCol >= 0) break;
        }

        if (balloonCol < 0) balloonCol = 0;
        if (dimensionCol < 0) dimensionCol = 1;

        // Extract data rows
        for (int r = 0; r < rowBounds.Count - 1; r++)
        {
            int y1 = rowBounds[r], y2 = rowBounds[r + 1];
            if (y2 - y1 < 10) continue;

            string balloonText = OcrCell(pageImage, y1, y2, colBounds, balloonCol);
            string digits = new string(balloonText.Where(char.IsDigit).ToArray());
            if (!int.TryParse(digits, out int num) || num < 1 || num > 99) continue;

            string dimension = OcrCell(pageImage, y1, y2, colBounds, dimensionCol).Trim();
            if (!string.IsNullOrWhiteSpace(dimension))
                result[num] = dimension;
        }

        return result;
    }

    private (List<int> RowBounds, List<int> ColBounds) DetectGrid(Mat pageImage)
    {
        using var gray = new Mat();
        using var binary = new Mat();
        Cv2.CvtColor(pageImage, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.AdaptiveThreshold(gray, binary, 255, AdaptiveThresholdTypes.MeanC,
            ThresholdTypes.BinaryInv, 15, 10);

        int hKernelW = Math.Max(pageImage.Width / 20, 50);
        using var hKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(hKernelW, 1));
        using var hLines = new Mat();
        Cv2.MorphologyEx(binary, hLines, MorphTypes.Open, hKernel);

        int vKernelH = Math.Max(pageImage.Height / 40, 20);
        using var vKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(1, vKernelH));
        using var vLines = new Mat();
        Cv2.MorphologyEx(binary, vLines, MorphTypes.Open, vKernel);

        return (ExtractLineBounds(hLines, true), ExtractLineBounds(vLines, false));
    }

    private List<int> ExtractLineBounds(Mat lineMask, bool isHorizontal)
    {
        var bounds = new List<int>();
        int len = isHorizontal ? lineMask.Height : lineMask.Width;
        int crossLen = isHorizontal ? lineMask.Width : lineMask.Height;

        for (int i = 0; i < len; i++)
        {
            int count = 0;
            for (int j = 0; j < crossLen; j += 4)
            {
                byte val = isHorizontal ? lineMask.At<byte>(i, j) : lineMask.At<byte>(j, i);
                if (val > 0) count++;
            }

            if (count > crossLen / 16)
            {
                if (bounds.Count == 0 || i - bounds[^1] > 5)
                    bounds.Add(i);
                else
                    bounds[^1] = (bounds[^1] + i) / 2;
            }
        }

        return bounds;
    }

    private string OcrCell(Mat pageImage, int y1, int y2, List<int> colBounds, int colIdx)
    {
        if (colIdx < 0 || colIdx >= colBounds.Count - 1) return "";

        int x1 = Math.Max(0, colBounds[colIdx] + 2);
        int x2 = Math.Min(pageImage.Width, colBounds[colIdx + 1] - 2);
        y1 = Math.Max(0, y1 + 2);
        y2 = Math.Min(pageImage.Height, y2 - 2);
        if (x2 - x1 < 5 || y2 - y1 < 5) return "";

        using var cell = new Mat(pageImage, new OpenCvSharp.Rect(x1, y1, x2 - x1, y2 - y1));
        using var gray = new Mat();
        Cv2.CvtColor(cell, gray, ColorConversionCodes.BGR2GRAY);

        using var upscaled = new Mat();
        Cv2.Resize(gray, upscaled, new Size(gray.Width * 3, gray.Height * 3),
            interpolation: InterpolationFlags.Cubic);

        using var binary = new Mat();
        Cv2.Threshold(upscaled, binary, 160, 255, ThresholdTypes.Binary);

        using var padded = new Mat();
        Cv2.CopyMakeBorder(binary, padded, 10, 10, 10, 10, BorderTypes.Constant, new Scalar(255));

        Cv2.ImEncode(".png", padded, out var buf);
        using var img = TesseractOCR.Pix.Image.LoadFromMemory(buf);
        using var page = _engine.Process(img, PageSegMode.SingleLine);
        return page.Text?.Trim() ?? "";
    }

    private Dictionary<int, string> ExtractViaFullPageOcr(Mat pageImage)
    {
        var result = new Dictionary<int, string>();
        using var gray = new Mat();
        Cv2.CvtColor(pageImage, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.ImEncode(".png", gray, out var buf);
        using var img = TesseractOCR.Pix.Image.LoadFromMemory(buf);
        using var page = _engine.Process(img, PageSegMode.Auto);

        foreach (var line in (page.Text ?? "").Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) continue;
            string digits = new string(parts[0].Where(char.IsDigit).ToArray());
            if (int.TryParse(digits, out int num) && num >= 1 && num <= 99)
                result.TryAdd(num, parts[1]);
        }
        return result;
    }

    public void Dispose()
    {
        _engine.Dispose();
    }
}
