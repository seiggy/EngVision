using System.Text.RegularExpressions;
using Azure;
using Azure.AI.DocumentIntelligence;
using OpenCvSharp;

namespace EngVision.Services;

/// <summary>
/// Azure Document Intelligence bubble OCR — drop-in replacement for BubbleOcrService.
/// Uses the prebuilt-read model to read numbers from bubble crop images.
/// </summary>
public class AzureBubbleOcrService : IBubbleOcrService
{
    private readonly DocumentIntelligenceClient _client;

    public AzureBubbleOcrService(string endpoint, string key)
    {
        _client = new DocumentIntelligenceClient(new Uri(endpoint), new AzureKeyCredential(key));
    }

    public int? ExtractBubbleNumber(string cropImagePath)
    {
        var src = Cv2.ImRead(cropImagePath, ImreadModes.Color);
        if (src.Empty()) return null;
        return ExtractBubbleNumber(src);
    }

    public int? ExtractBubbleNumber(Mat src)
    {
        var processed = PreprocessForOcr(src);
        Cv2.ImEncode(".png", processed, out var buf);

        var binaryData = BinaryData.FromBytes(buf);
        var operation = _client.AnalyzeDocument(WaitUntil.Completed, "prebuilt-read", binaryData);
        var result = operation.Value;

        var allText = result.Content ?? "";
        return ParseBubbleNumber(allText);
    }

    public Dictionary<string, int?> ExtractAll(string cropDirectory)
    {
        var results = new Dictionary<string, int?>();
        var files = Directory.GetFiles(cropDirectory, "bubble_*.png")
            .OrderBy(f => f)
            .ToArray();

        foreach (var path in files)
        {
            var filename = Path.GetFileName(path);
            results[filename] = ExtractBubbleNumber(path);
        }

        return results;
    }

    private static Mat PreprocessForOcr(Mat src)
    {
        // Remove blue circle pixels
        var hsv = new Mat();
        Cv2.CvtColor(src, hsv, ColorConversionCodes.BGR2HSV);
        var blueMask = new Mat();
        Cv2.InRange(hsv, new Scalar(85, 25, 50), new Scalar(125, 255, 255), blueMask);
        var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
        var dilatedBlue = new Mat();
        Cv2.Dilate(blueMask, dilatedBlue, kernel, iterations: 1);

        var cleaned = src.Clone();
        cleaned.SetTo(new Scalar(255, 255, 255), dilatedBlue);

        var gray = new Mat();
        Cv2.CvtColor(cleaned, gray, ColorConversionCodes.BGR2GRAY);

        // Upscale 4x
        var upscaled = new Mat();
        Cv2.Resize(gray, upscaled, new Size(gray.Width * 4, gray.Height * 4), interpolation: InterpolationFlags.Cubic);

        // Adaptive threshold
        var binary = new Mat();
        Cv2.AdaptiveThreshold(upscaled, binary, 255,
            AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 31, 10);

        // White border
        var padded = new Mat();
        Cv2.CopyMakeBorder(binary, padded, 20, 20, 20, 20, BorderTypes.Constant, new Scalar(255));

        return padded;
    }

    private static int? ParseBubbleNumber(string ocrText)
    {
        if (string.IsNullOrWhiteSpace(ocrText)) return null;

        var cleaned = ocrText
            .Replace("#", "")
            .Replace("O", "0").Replace("o", "0")
            .Replace("l", "1").Replace("I", "1")
            .Replace("S", "5").Replace("B", "8")
            .Replace(" ", "")
            .Trim();

        var digits = Regex.Replace(cleaned, @"[^0-9]", "");
        if (int.TryParse(digits, out var num) && num is >= 1 and <= 99)
            return num;
        return null;
    }

    public void Dispose() { }
}
