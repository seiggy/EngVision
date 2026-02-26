namespace EngVision.Models;

/// <summary>
/// A 2D point (x, y) used for bubble center positions.
/// </summary>
public record AnnotationPoint
{
    public int X { get; init; }
    public int Y { get; init; }
}

/// <summary>
/// Ground truth annotation from the manual annotation tool.
/// </summary>
public record GroundTruthAnnotation
{
    public string Id { get; init; } = "";
    public int? BubbleNumber { get; init; }
    public AnnotationPoint? BubbleCenter { get; init; }
    public BoundingBox BoundingBox { get; init; } = new(0, 0, 0, 0);
    public string? Label { get; init; }
    public string? Notes { get; init; }
}
