using System.Diagnostics;
using EngVision.Models;
using OpenAI;
using OpenAI.Responses;
using OpenCvSharp;

namespace EngVision.Services;

/// <summary>
/// Orchestrates the full analysis pipeline: render → detect → OCR → LLM → merge.
/// Designed to be called from the API for a given PDF file.
/// </summary>
public class PipelineService
{
    private readonly EngVisionConfig _config;
    private readonly string _tessDataPath;

    public PipelineService(EngVisionConfig config, string tessDataPath)
    {
        _config = config;
        _tessDataPath = tessDataPath;
    }

    /// <summary>
    /// Runs the full pipeline on a PDF file. Saves page images and overlay to outputDir.
    /// </summary>
    public async Task<PipelineResult> RunAsync(string pdfPath, string runId, string outputDir,
        Action<string>? onProgress = null)
    {
        void Progress(string msg) => onProgress?.Invoke(msg);
        Directory.CreateDirectory(outputDir);
        var pagesDir = Path.Combine(outputDir, "pages");
        var overlayDir = Path.Combine(outputDir, "overlays");
        Directory.CreateDirectory(pagesDir);
        Directory.CreateDirectory(overlayDir);

        var filename = Path.GetFileName(pdfPath);

        try
        {
            var totalSw = Stopwatch.StartNew();
            var stepSw = new Stopwatch();
            long renderMs = 0, detectMs = 0, ocrMs = 0, llmMs = 0, mergeMs = 0;
            int llmInputTokens = 0, llmOutputTokens = 0, llmTotalTokens = 0, llmCalls = 0;

            // Step 1: Render pages
            Progress("Rendering PDF pages...");
            stepSw.Restart();
            using var renderer = new PdfRendererService(_config.PdfRenderDpi);
            var pageImages = renderer.RenderAllPages(pdfPath);
            renderMs = stepSw.ElapsedMilliseconds;

            // Save page images as PNG
            for (int i = 0; i < pageImages.Count; i++)
                PdfRendererService.SaveImage(pageImages[i], Path.Combine(pagesDir, $"page_{i + 1}.png"));

            int pageCount = pageImages.Count;
            int imgW = pageImages[0].Width, imgH = pageImages[0].Height;

            // Step 2: Detect bubbles on page 1
            Progress("Detecting bubbles...");
            stepSw.Restart();
            var bubbleDetector = new BubbleDetectionService(_config);
            var bubbles = bubbleDetector.DetectBubbles(pageImages[0], pageNumber: 1);
            detectMs = stepSw.ElapsedMilliseconds;

            // Step 2b: Save raw bubble crops for OCR
            var rawCropsDir = Path.Combine(outputDir, "bubble_crops");
            Directory.CreateDirectory(rawCropsDir);
            foreach (var b in bubbles)
            {
                var bb = b.BoundingBox;
                int cx = bb.X + bb.Width / 2, cy = bb.Y + bb.Height / 2, r = bb.Width / 2;
                int pad = 2;
                int x1 = Math.Max(0, cx - r - pad), y1 = Math.Max(0, cy - r - pad);
                int x2 = Math.Min(imgW, cx + r + pad), y2 = Math.Min(imgH, cy + r + pad);
                using var crop = new Mat(pageImages[0], new Rect(x1, y1, x2 - x1, y2 - y1));
                Cv2.ImWrite(Path.Combine(rawCropsDir, $"bubble_{b.BubbleNumber:D3}.png"), crop);
            }

            // Step 2c: OCR bubble numbers
            Progress("OCR-ing bubble numbers...");
            stepSw.Restart();
            using var ocrService = new BubbleOcrService(_tessDataPath);
            var ocrResults = ocrService.ExtractAll(rawCropsDir);

            // Step 3: Table OCR (pages 2-4)
            Progress("OCR-ing table data...");
            using var tableOcr = new TableOcrService(_tessDataPath);
            var tesseractDimensions = new Dictionary<int, string>();
            for (int i = 1; i < pageImages.Count; i++)
            {
                var pageDims = tableOcr.ExtractBalloonDimensions(pageImages[i]);
                foreach (var (num, dim) in pageDims)
                    tesseractDimensions.TryAdd(num, dim);
            }
            ocrMs = stepSw.ElapsedMilliseconds;

            // Step 4: Vision LLM extraction (if configured)
            var llmDimensions = new Dictionary<int, string>();
            var endpoint = Environment.GetEnvironmentVariable("AZURE_ENDPOINT");
            var key = Environment.GetEnvironmentVariable("AZURE_KEY");
            var model = Environment.GetEnvironmentVariable("AZURE_DEPLOYMENT_NAME") ?? "gpt-5.3-codex";

            stepSw.Restart();
            if (!string.IsNullOrEmpty(key) && !string.IsNullOrEmpty(endpoint))
            {
                Progress("Running Vision LLM table extraction...");
                var responsesClient = new ResponsesClient(
                    model,
                    new System.ClientModel.ApiKeyCredential(key),
                    new OpenAIClientOptions { Endpoint = new Uri($"{endpoint.TrimEnd('/')}/openai/v1/") });
                var visionService = new VisionLlmService(responsesClient);

                for (int i = 1; i < pageImages.Count; i++)
                {
                    var pageBytes = pageImages[i].ToBytes(".png");
                    var extraction = await visionService.ExtractBalloonDimensionsWithUsage(pageBytes, i + 1);
                    foreach (var (num, dim) in extraction.Dimensions)
                        llmDimensions.TryAdd(num, dim);
                    llmInputTokens += extraction.InputTokens;
                    llmOutputTokens += extraction.OutputTokens;
                    llmTotalTokens += extraction.TotalTokens;
                    llmCalls++;
                }
            }
            llmMs = stepSw.ElapsedMilliseconds;

            // Step 5: Merge results
            Progress("Merging OCR + LLM results...");
            stepSw.Restart();
            var dimensionMap = new Dictionary<int, DimensionMatch>();

            // Build bubble list from OCR results
            var bubbleResults = new List<BubbleResult>();
            foreach (var (file, number) in ocrResults)
            {
                if (!number.HasValue) continue;
                int cropIdx = int.Parse(Path.GetFileNameWithoutExtension(file).Replace("bubble_", "")) - 1;
                if (cropIdx < 0 || cropIdx >= bubbles.Count) continue;
                var bb = bubbles[cropIdx].BoundingBox;
                int cx = bb.X + bb.Width / 2, cy = bb.Y + bb.Height / 2, r = bb.Width / 2;

                bubbleResults.Add(new BubbleResult
                {
                    BubbleNumber = number.Value,
                    Cx = cx,
                    Cy = cy,
                    Radius = r,
                    BoundingBox = bb
                });

                // Merge dimension data
                var tessVal = tesseractDimensions.GetValueOrDefault(number.Value);
                var llmVal = llmDimensions.GetValueOrDefault(number.Value);
                string? dimension = tessVal ?? llmVal;
                string source = (tessVal, llmVal) switch
                {
                    (not null, not null) => "Both",
                    (not null, null) => "Tesseract",
                    (null, not null) => "LLM",
                    _ => "None"
                };

                // Compute confidence score — need both sources to confirm
                double confidence;
                if (tessVal is not null && llmVal is not null)
                    confidence = DimensionMatcher.ConfidenceScore(tessVal, llmVal);
                else
                    confidence = 0.0; // Missing from one or both sources

                bool hasConflict = tessVal is not null && llmVal is not null
                    && !DimensionMatcher.AreSimilar(tessVal, llmVal);

                dimensionMap[number.Value] = new DimensionMatch
                {
                    BalloonNo = number.Value,
                    Dimension = dimension,
                    Source = source,
                    TesseractValue = tessVal,
                    LlmValue = llmVal,
                    HasConflict = hasConflict,
                    Confidence = Math.Round(confidence, 4)
                };
            }
            mergeMs = stepSw.ElapsedMilliseconds;

            // Step 6: Generate overlay image
            Progress("Generating overlay images...");
            GenerateOverlay(pageImages[0], bubbleResults, dimensionMap,
                Path.Combine(overlayDir, "page_1_overlay.png"));

            // Cleanup page images
            foreach (var page in pageImages) page.Dispose();

            totalSw.Stop();
            var peakMemory = GC.GetTotalMemory(false) / (1024.0 * 1024.0);

            Progress("Complete!");
            return new PipelineResult
            {
                RunId = runId,
                PdfFilename = filename,
                PageCount = pageCount,
                ImageWidth = imgW,
                ImageHeight = imgH,
                Bubbles = bubbleResults.OrderBy(b => b.BubbleNumber).ToList(),
                DimensionMap = dimensionMap,
                Metrics = new ProcessingMetrics
                {
                    TotalDurationMs = totalSw.ElapsedMilliseconds,
                    RenderDurationMs = renderMs,
                    DetectDurationMs = detectMs,
                    OcrDurationMs = ocrMs,
                    LlmDurationMs = llmMs,
                    MergeDurationMs = mergeMs,
                    PeakMemoryMb = Math.Round(peakMemory, 1)
                },
                TokenUsage = llmCalls > 0 ? new LlmTokenUsage
                {
                    InputTokens = llmInputTokens,
                    OutputTokens = llmOutputTokens,
                    TotalTokens = llmTotalTokens,
                    LlmCalls = llmCalls
                } : null,
                Status = "complete"
            };
        }
        catch (Exception ex)
        {
            return new PipelineResult
            {
                RunId = runId,
                PdfFilename = filename,
                PageCount = 0,
                ImageWidth = 0,
                ImageHeight = 0,
                Bubbles = [],
                DimensionMap = new(),
                Status = "error",
                Error = ex.Message
            };
        }
    }

    private static void GenerateOverlay(Mat pageImage, List<BubbleResult> bubbles,
        Dictionary<int, DimensionMatch> dimensionMap, string outputPath)
    {
        using var overlay = pageImage.Clone();

        foreach (var bubble in bubbles)
        {
            var match = dimensionMap.GetValueOrDefault(bubble.BubbleNumber);
            bool hasMatch = match?.Dimension is not null;
            bool hasConflict = match?.HasConflict ?? false;

            // Color: green=matched, red=unmatched, amber=conflict
            Scalar circleColor = hasMatch
                ? (hasConflict ? new Scalar(0, 200, 255) : new Scalar(0, 200, 0))  // amber or green
                : new Scalar(0, 0, 255);  // red

            // Draw circle
            Cv2.Circle(overlay, new Point(bubble.Cx, bubble.Cy), bubble.Radius + 3, circleColor, 3);

            // Draw bubble number label
            var label = $"#{bubble.BubbleNumber}";
            Cv2.PutText(overlay, label,
                new Point(bubble.Cx - 12, bubble.Cy - bubble.Radius - 8),
                HersheyFonts.HersheySimplex, 0.5, circleColor, 2);

            // Draw dimension text if available
            if (match?.Dimension is not null)
            {
                var dimLabel = match.Dimension.Length > 20
                    ? match.Dimension[..20] + "…"
                    : match.Dimension;
                Cv2.PutText(overlay, dimLabel,
                    new Point(bubble.Cx + bubble.Radius + 8, bubble.Cy + 5),
                    HersheyFonts.HersheySimplex, 0.35, circleColor, 1);
            }
        }

        Cv2.ImWrite(outputPath, overlay);
    }
}
