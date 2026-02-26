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
}

public record BoundingBox(int X, int Y, int Width, int Height);

public enum RegionType
{
    Bubble,
    BubbleWithFigure,
    TableRegion,
    FullPage
}
