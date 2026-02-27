using OpenCvSharp;
using TesseractOCR;
using TesseractOCR.Enums;

namespace EngVision.Services;

/// <summary>
/// OCR service for extracting bubble numbers from tight bubble crop images.
/// Preprocesses crops (remove blue circle, upscale, threshold) before Tesseract OCR.
/// </summary>
public class BubbleOcrService : IBubbleOcrService
{
    private readonly Engine _engine;

    public BubbleOcrService(string tessDataPath)
    {
        _engine = new Engine(tessDataPath, Language.English, EngineMode.Default);
    }

    /// <summary>
    /// Extract the bubble number from a crop image file.
    /// Returns null if OCR fails or no digit is found.
    /// </summary>
    public int? ExtractBubbleNumber(string cropImagePath)
    {
        using var src = Cv2.ImRead(cropImagePath, ImreadModes.Color);
        if (src.Empty()) return null;
        return ExtractBubbleNumber(src);
    }

    /// <summary>
    /// Extract the bubble number from a crop Mat.
    /// </summary>
    public int? ExtractBubbleNumber(Mat src)
    {
        using var processed = PreprocessForOcr(src);
        var text = RunOcr(processed);
        return ParseBubbleNumber(text);
    }

    /// <summary>
    /// Batch process: extract bubble numbers from all crop files in a directory.
    /// Returns dictionary mapping crop filename → bubble number.
    /// </summary>
    public Dictionary<string, int?> ExtractAll(string cropDirectory)
    {
        var results = new Dictionary<string, int?>();
        var files = Directory.GetFiles(cropDirectory, "bubble_*.png")
            .OrderBy(f => f)
            .ToArray();

        foreach (var file in files)
        {
            var number = ExtractBubbleNumber(file);
            results[Path.GetFileName(file)] = number;
        }
        return results;
    }

    /// <summary>
    /// Preprocess a bubble crop for OCR:
    /// 1. Remove blue circle pixels (replace with white)
    /// 2. Upscale 6x for better OCR accuracy
    /// 3. Convert to grayscale and threshold
    /// 4. Add white border padding
    /// </summary>
    private Mat PreprocessForOcr(Mat src)
    {
        // Remove blue circle: convert blue pixels to white
        using var hsv = new Mat();
        Cv2.CvtColor(src, hsv, ColorConversionCodes.BGR2HSV);
        using var blueMask = new Mat();
        Cv2.InRange(hsv, new Scalar(85, 25, 50), new Scalar(125, 255, 255), blueMask);
        // Dilate blue mask slightly to catch anti-aliased edges
        using var dilatedBlue = new Mat();
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
        Cv2.Dilate(blueMask, dilatedBlue, kernel, iterations: 1);

        // Replace blue pixels with white
        using var cleaned = src.Clone();
        cleaned.SetTo(new Scalar(255, 255, 255), dilatedBlue);

        // Convert to grayscale
        using var gray = new Mat();
        Cv2.CvtColor(cleaned, gray, ColorConversionCodes.BGR2GRAY);

        // Upscale 6x for better OCR on small text
        using var upscaled = new Mat();
        Cv2.Resize(gray, upscaled, new Size(gray.Width * 6, gray.Height * 6), interpolation: InterpolationFlags.Cubic);

        // Use adaptive thresholding for better handling of varying contrast
        using var binary = new Mat();
        Cv2.AdaptiveThreshold(upscaled, binary, 255, AdaptiveThresholdTypes.GaussianC,
            ThresholdTypes.Binary, 31, 10);

        // Add white border padding (Tesseract needs some whitespace around text)
        var padded = new Mat();
        Cv2.CopyMakeBorder(binary, padded, 20, 20, 20, 20, BorderTypes.Constant, new Scalar(255));

        return padded;
    }

    /// <summary>
    /// Run Tesseract OCR on a preprocessed grayscale image.
    /// Configured for digits only.
    /// </summary>
    private string RunOcr(Mat processed)
    {
        // Convert Mat to byte array (PNG)
        Cv2.ImEncode(".png", processed, out var buf);

        using var img = TesseractOCR.Pix.Image.LoadFromMemory(buf);
        using var page = _engine.Process(img, PageSegMode.SingleWord);

        // Try to set digit whitelist via the engine's SetVariable
        // TesseractOCR wraps this differently - we handle filtering in ParseBubbleNumber
        return page.Text?.Trim() ?? "";
    }

    /// <summary>
    /// Parse OCR output to extract a bubble number (1-51).
    /// Handles common OCR artifacts like '#', 'O'→'0', 'l'→'1', etc.
    /// </summary>
    private static int? ParseBubbleNumber(string ocrText)
    {
        if (string.IsNullOrWhiteSpace(ocrText)) return null;

        // Common OCR substitutions
        var cleaned = ocrText
            .Replace("#", "")
            .Replace("O", "0")
            .Replace("o", "0")
            .Replace("l", "1")
            .Replace("I", "1")
            .Replace("S", "5")
            .Replace("B", "8")
            .Replace(" ", "")
            .Trim();

        // Extract just digits
        var digits = new string(cleaned.Where(char.IsDigit).ToArray());

        if (int.TryParse(digits, out var number) && number >= 1 && number <= 99)
            return number;

        return null;
    }

    public void Dispose()
    {
        _engine.Dispose();
    }
}
