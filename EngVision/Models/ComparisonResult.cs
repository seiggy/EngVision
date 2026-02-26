namespace EngVision.Models;

/// <summary>
/// Result of comparing a bubble measurement against its corresponding table entry.
/// </summary>
public record ComparisonResult
{
    public required int BubbleNumber { get; init; }
    public MeasurementData? BubbleMeasurement { get; init; }
    public MeasurementData? TableMeasurement { get; init; }
    public bool Match { get; init; }
    public string? Discrepancy { get; init; }
}
