namespace EngVision.Models;

/// <summary>
/// Structured measurement data extracted by the vision LLM from a segment.
/// </summary>
public record MeasurementData
{
    public int BubbleNumber { get; init; }
    public string? DimensionName { get; init; }
    public string? NominalValue { get; init; }
    public string? Unit { get; init; }
    public string? UpperTolerance { get; init; }
    public string? LowerTolerance { get; init; }
    public string? ActualValue { get; init; }
    public string? RawText { get; init; }
    public int SourcePage { get; init; }
    public RegionType SourceType { get; init; }
}
