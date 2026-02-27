namespace EngVision.Models;

/// <summary>
/// Represents a detected region on a PDF page with its bounding box and metadata.
/// </summary>
public record DetectedRegion
{
    public required int Id { get; init; }
    public required int PageNumber { get; init; }
    public required RegionType Type { get; init; }
    public required BoundingBox BoundingBox { get; init; }
    public int? BubbleNumber { get; init; }
    public string? Label { get; init; }
    public string? CroppedImagePath { get; init; }
    /// <summary>
    /// Focused capture box placed along the leader line direction.
    /// Used by the LLM validation step to crop the drawing region to inspect.
    /// </summary>
    public BoundingBox? CaptureBox { get; init; }
    /// <summary>
    /// Unit direction vector from bubble centre toward the leader line.
    /// </summary>
    public (double Dx, double Dy)? LeaderDirection { get; init; }
}

public record BoundingBox(int X, int Y, int Width, int Height);

public enum RegionType
{
    Bubble,
    BubbleWithFigure,
    TableRegion,
    FullPage
}
