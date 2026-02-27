using OpenCvSharp;

namespace EngVision.Services;

/// <summary>Interface for bubble number OCR services.</summary>
public interface IBubbleOcrService : IDisposable
{
    int? ExtractBubbleNumber(string cropImagePath);
    int? ExtractBubbleNumber(Mat src);
    Dictionary<string, int?> ExtractAll(string cropDirectory);
}

/// <summary>Interface for table OCR services.</summary>
public interface ITableOcrService : IDisposable
{
    Dictionary<int, string> ExtractBalloonDimensions(Mat pageImage);
}
