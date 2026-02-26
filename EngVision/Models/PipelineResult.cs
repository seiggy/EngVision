namespace EngVision.Models;

/// <summary>
/// Result of a full pipeline run on a PDF document.
/// </summary>
public record PipelineResult
{
    public required string RunId { get; init; }
    public required string PdfFilename { get; init; }
    public int PageCount { get; init; }
    public int ImageWidth { get; init; }
    public int ImageHeight { get; init; }
    public required List<BubbleResult> Bubbles { get; init; }
    public required Dictionary<int, DimensionMatch> DimensionMap { get; init; }
    public int TotalBubbles => Bubbles.Count;
    public int MatchedBubbles => DimensionMap.Values.Count(d => d.Dimension is not null);
    public int UnmatchedBubbles => TotalBubbles - MatchedBubbles;
    public int Warnings => DimensionMap.Values.Count(d => d.Confidence > 0 && d.Confidence < 0.8);
    public ProcessingMetrics? Metrics { get; init; }
    public LlmTokenUsage? TokenUsage { get; init; }
    public string Status { get; init; } = "complete";
    public string? Error { get; init; }
}

public record BubbleResult
{
    public int BubbleNumber { get; init; }
    public int Cx { get; init; }
    public int Cy { get; init; }
    public int Radius { get; init; }
    public BoundingBox BoundingBox { get; init; } = null!;
}

public record DimensionMatch
{
    public int BalloonNo { get; init; }
    public string? Dimension { get; init; }
    public string Source { get; init; } = "None"; // "Tesseract", "LLM", "Both", "None"
    public string? TesseractValue { get; init; }
    public string? LlmValue { get; init; }
    public bool HasConflict { get; init; }
    /// <summary>
    /// Confidence score 0-1. 1.0 = exact match, 0.0 = missing/no match.
    /// </summary>
    public double Confidence { get; init; }
}

/// <summary>
/// Processing performance metrics for a pipeline run.
/// </summary>
public record ProcessingMetrics
{
    public long TotalDurationMs { get; init; }
    public long RenderDurationMs { get; init; }
    public long DetectDurationMs { get; init; }
    public long OcrDurationMs { get; init; }
    public long LlmDurationMs { get; init; }
    public long MergeDurationMs { get; init; }
    public double PeakMemoryMb { get; init; }
}

/// <summary>
/// Token usage across all LLM calls in a pipeline run.
/// </summary>
public record LlmTokenUsage
{
    public int InputTokens { get; init; }
    public int OutputTokens { get; init; }
    public int TotalTokens { get; init; }
    public int LlmCalls { get; init; }
}
