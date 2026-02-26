namespace EngVision.Models;

/// <summary>
/// Configuration for the application, loaded from appsettings or environment variables.
/// </summary>
public class EngVisionConfig
{
    public string OpenAIApiKey { get; set; } = string.Empty;
    public string OpenAIModel { get; set; } = "gpt-4o";
    public string? OpenAIEndpoint { get; set; }
    public int PdfRenderDpi { get; set; } = 300;
    public string OutputDirectory { get; set; } = "Output";

    // Bubble detection parameters
    public int HoughMinRadius { get; set; } = 12;
    public int HoughMaxRadius { get; set; } = 50;
    public double HoughParam1 { get; set; } = 120;
    public double HoughParam2 { get; set; } = 25;
    public int BubbleContextPadding { get; set; } = 150;

    // Table detection parameters
    public int TableMinWidth { get; set; } = 200;
    public int TableMinHeight { get; set; } = 100;
}
