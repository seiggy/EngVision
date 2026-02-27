using System.Diagnostics;
using System.Text.Json;
using System.Threading.Channels;
using EngVision.Models;
using OpenAI;
using OpenAI.Responses;
using OpenCvSharp;

namespace EngVision.Services;

/// <summary>
/// Orchestrates the full analysis pipeline: render → detect → OCR → trace → validate → merge.
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

    /// <summary>Create OCR services based on OCR_PROVIDER config.</summary>
    private (IBubbleOcrService bubbleOcr, ITableOcrService tableOcr) CreateOcrServices()
    {
        if (string.Equals(_config.OcrProvider, "Azure", StringComparison.OrdinalIgnoreCase))
        {
            var ep = _config.AzureDocIntEndpoint;
            var key = _config.AzureDocIntKey;
            if (string.IsNullOrEmpty(ep) || string.IsNullOrEmpty(key))
            {
                Console.WriteLine("  WARNING: OCR_PROVIDER=Azure but AZURE_DOCINT_ENDPOINT/KEY not set — falling back to Tesseract");
                return (new BubbleOcrService(_tessDataPath), new TableOcrService(_tessDataPath));
            }
            Console.WriteLine($"  Using Azure Document Intelligence for OCR ({ep})");
            return (new AzureBubbleOcrService(ep, key), new AzureTableOcrService(ep, key));
        }
        return (new BubbleOcrService(_tessDataPath), new TableOcrService(_tessDataPath));
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
            long renderMs = 0, detectMs = 0, traceMs = 0, ocrMs = 0, llmMs = 0, mergeMs = 0;
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
            var (bubbleOcr, tableOcrSvc) = CreateOcrServices();
            using var ocrDisp = bubbleOcr as IDisposable;
            using var tableDisp = tableOcrSvc as IDisposable;
            var ocrResults = bubbleOcr.ExtractAll(rawCropsDir);

            // Step 3: Table OCR (pages 2+)
            Progress("OCR-ing table data...");
            var tesseractDimensions = new Dictionary<int, string>();
            if (tableOcrSvc is AzureTableOcrService azureTable)
            {
                // Full-PDF mode: single API call for all pages
                var pdfBytes = File.ReadAllBytes(pdfPath);
                tesseractDimensions = azureTable.ExtractBalloonDimensionsFromPdf(pdfBytes);
            }
            else if (tableOcrSvc is TableOcrService localTable)
            {
                for (int i = 1; i < pageImages.Count; i++)
                {
                    var pageDims = localTable.ExtractBalloonDimensions(pageImages[i]);
                    foreach (var (num, dim) in pageDims)
                        tesseractDimensions.TryAdd(num, dim);
                }
            }
            ocrMs = stepSw.ElapsedMilliseconds;

            // Step 4: Trace leader lines from each bubble
            Progress("Tracing leader lines...");
            stepSw.Restart();
            var tracer = new LeaderLineTracerService();
            var expandedBubbles = tracer.TraceAndExpand(bubbles, pageImages[0]);
            traceMs = stepSw.ElapsedMilliseconds;

            // Step 5: Vision LLM validation with progressive capture expansion
            // For each bubble with a table dimension, try progressively larger
            // capture boxes along the leader line until the LLM confirms a match.
            // Sizes: 128×128 → 256×128 → 512×256 → 1024×512
            var llmValidations = new Dictionary<int, VisionLlmService.LlmValidationResult>();
            var llmCaptureSizes = new Dictionary<int, string>();
            var endpoint = Environment.GetEnvironmentVariable("AZURE_ENDPOINT");
            var key = Environment.GetEnvironmentVariable("AZURE_KEY");
            var model = Environment.GetEnvironmentVariable("AZURE_DEPLOYMENT_NAME") ?? "gpt-5.3-codex";

            // Debug: save capture box crops
            var debugDir = Path.Combine(outputDir, "debug");
            Directory.CreateDirectory(debugDir);

            stepSw.Restart();
            if (!string.IsNullOrEmpty(key) && !string.IsNullOrEmpty(endpoint))
            {
                Progress("Validating dimensions with Vision LLM...");
                var responsesClient = new ResponsesClient(
                    model,
                    new System.ClientModel.ApiKeyCredential(key),
                    new OpenAIClientOptions { Endpoint = new Uri($"{endpoint.TrimEnd('/')}/openai/v1/") });
                var visionService = new VisionLlmService(responsesClient);

                // Build lookup from bubble index → expanded region
                var expandedByIdx = new Dictionary<int, DetectedRegion>();
                for (int i = 0; i < expandedBubbles.Count; i++)
                    expandedByIdx[i] = expandedBubbles[i];

                foreach (var (file, number) in ocrResults)
                {
                    if (!number.HasValue) continue;
                    int cropIdx = int.Parse(Path.GetFileNameWithoutExtension(file).Replace("bubble_", "")) - 1;
                    if (cropIdx < 0 || cropIdx >= bubbles.Count) continue;

                    var tableDim = tesseractDimensions.GetValueOrDefault(number.Value);

                    if (!expandedByIdx.TryGetValue(cropIdx, out var eb)) continue;
                    if (eb.LeaderDirection is null) continue;
                    var (dx, dy) = (eb.LeaderDirection.Value.Dx, eb.LeaderDirection.Value.Dy);

                    // Original bubble geometry for box placement
                    var origBb = bubbles[cropIdx].BoundingBox;
                    int bcx = origBb.X + origBb.Width / 2;
                    int bcy = origBb.Y + origBb.Height / 2;
                    int bRadius = origBb.Width / 2;

                    if (tableDim is not null)
                    {
                        // Progressive expansion: try each size, stop on match
                        VisionLlmService.LlmValidationResult? lastValidation = null;
                        string? finalCaptureSize = null;
                        foreach (var (capW, capH) in LeaderLineTracerService.CaptureSteps)
                        {
                            var cap = LeaderLineTracerService.PlaceCaptureBox(
                                bcx, bcy, bRadius, dx, dy, capW, capH, imgW, imgH);
                            int x1 = Math.Max(0, cap.X);
                            int y1 = Math.Max(0, cap.Y);
                            int x2 = Math.Min(imgW, cap.X + cap.Width);
                            int y2 = Math.Min(imgH, cap.Y + cap.Height);
                            if (x2 - x1 < 4 || y2 - y1 < 4) continue;

                            using var regionCrop = new Mat(pageImages[0], new Rect(x1, y1, x2 - x1, y2 - y1));
                            var cropBytes = regionCrop.ToBytes(".png");

                            // Save debug capture at this step
                            Cv2.ImWrite(Path.Combine(debugDir,
                                $"capture_bubble_{number.Value:D3}_{capW}x{capH}.png"), regionCrop);

                            var validation = await visionService.ValidateDimension(cropBytes, number.Value, tableDim);
                            lastValidation = validation;
                            finalCaptureSize = $"{capW}x{capH}";
                            llmInputTokens += validation.InputTokens;
                            llmOutputTokens += validation.OutputTokens;
                            llmTotalTokens += validation.TotalTokens;
                            llmCalls++;

                            if (validation.Matches)
                            {
                                Progress($"  Bubble {number.Value}: matched at {capW}x{capH}");
                                break;
                            }
                            else
                            {
                                Progress($"  Bubble {number.Value}: no match at {capW}x{capH}" +
                                    $" (saw '{validation.ObservedDimension}'), expanding...");
                            }
                        }

                        // Use the last validation result (match or best guess at max size)
                        if (lastValidation is not null)
                        {
                            llmValidations[number.Value] = lastValidation;
                            if (finalCaptureSize is not null)
                                llmCaptureSizes[number.Value] = finalCaptureSize;
                        }
                    }
                    else
                    {
                        // Discovery mode: table OCR missed this entry
                        var (capW, capH) = LeaderLineTracerService.CaptureSteps[0];
                        var cap = LeaderLineTracerService.PlaceCaptureBox(
                            bcx, bcy, bRadius, dx, dy, capW, capH, imgW, imgH);
                        int x1 = Math.Max(0, cap.X);
                        int y1 = Math.Max(0, cap.Y);
                        int x2 = Math.Min(imgW, cap.X + cap.Width);
                        int y2 = Math.Min(imgH, cap.Y + cap.Height);
                        if (x2 - x1 >= 4 && y2 - y1 >= 4)
                        {
                            using var regionCrop = new Mat(pageImages[0], new Rect(x1, y1, x2 - x1, y2 - y1));
                            var cropBytes = regionCrop.ToBytes(".png");

                            Cv2.ImWrite(Path.Combine(debugDir,
                                $"capture_bubble_{number.Value:D3}_{capW}x{capH}.png"), regionCrop);

                            var discovery = await visionService.DiscoverDimension(cropBytes, number.Value);
                            llmInputTokens += discovery.InputTokens;
                            llmOutputTokens += discovery.OutputTokens;
                            llmTotalTokens += discovery.TotalTokens;
                            llmCalls++;
                            Progress($"  Bubble {number.Value}: discovered '{discovery.ObservedDimension}' (no table entry)");

                            llmValidations[number.Value] = discovery with
                            {
                                Notes = $"[Table OCR miss] {discovery.Notes}"
                            };
                            llmCaptureSizes[number.Value] = $"{capW}x{capH}";
                        }
                    }
                }
            }
            llmMs = stepSw.ElapsedMilliseconds;

            // Step 6: Merge results — table OCR + LLM validation
            Progress("Merging OCR + LLM validation results...");
            stepSw.Restart();
            var dimensionMap = new Dictionary<int, DimensionMatch>();

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

                var tessVal = tesseractDimensions.GetValueOrDefault(number.Value);
                var validation = llmValidations.GetValueOrDefault(number.Value);

                // The LLM validation tells us if the table dimension matches the drawing
                bool? llmMatches = validation?.Matches;
                string? llmObserved = validation?.ObservedDimension;
                double llmConfidence = validation?.Confidence ?? 0.0;
                string? llmNotes = validation?.Notes;

                // Determine conflict: table says one thing, drawing shows another
                bool hasConflict = validation is not null && !validation.Matches;

                // Confidence: when LLM confirms match, use its confidence directly.
                // Only use fuzzy string matching when there's a conflict.
                double confidence;
                if (validation is not null && validation.Matches)
                    confidence = llmConfidence;
                else if (tessVal is not null && llmObserved is not null && llmObserved.Length > 0)
                    confidence = DimensionMatcher.ConfidenceScore(tessVal, llmObserved);
                else if (validation is not null)
                    confidence = llmConfidence;
                else
                    confidence = 0.0;

                string source = (tessVal, validation) switch
                {
                    (not null, not null) => "Table+Validated",
                    (not null, null) => "TableOnly",
                    (null, not null) => "LLMOnly",
                    _ => "None"
                };

                dimensionMap[number.Value] = new DimensionMatch
                {
                    BalloonNo = number.Value,
                    Dimension = tessVal ?? llmObserved,
                    Source = source,
                    TesseractValue = tessVal,
                    LlmObservedValue = llmObserved,
                    LlmMatches = llmMatches,
                    LlmConfidence = Math.Round(llmConfidence, 4),
                    LlmNotes = llmNotes,
                    HasConflict = hasConflict,
                    Confidence = Math.Round(confidence, 4),
                    CaptureSize = llmCaptureSizes.GetValueOrDefault(number.Value),
                };
            }
            mergeMs = stepSw.ElapsedMilliseconds;

            // Step 7: Generate overlay image
            Progress("Generating overlay images...");
            stepSw.Restart();
            GenerateOverlay(pageImages[0], bubbleResults, dimensionMap,
                Path.Combine(overlayDir, "page_1_overlay.png"));
            var overlayMs = stepSw.ElapsedMilliseconds;

            // Cleanup page images
            foreach (var page in pageImages) page.Dispose();

            totalSw.Stop();
            var peakMemory = GC.GetTotalMemory(false) / (1024.0 * 1024.0);

            // Write benchmark.json
            WriteBenchmark(outputDir, runId, filename,
                [("render", renderMs), ("detect", detectMs), ("ocr", ocrMs),
                 ("trace", traceMs), ("validate", llmMs), ("merge", mergeMs), ("overlay", overlayMs)],
                totalSw.ElapsedMilliseconds, bubbleResults.Count,
                dimensionMap.Values.Count(d => d.Dimension is not null),
                llmCalls, llmInputTokens, llmOutputTokens, llmTotalTokens);

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
                    TraceDurationMs = traceMs,
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

    /// <summary>
    /// Streams pipeline events as an async enumerable of dictionaries (SSE-ready).
    /// Uses a Channel internally so that error handling works with async iteration.
    /// </summary>
    public IAsyncEnumerable<Dictionary<string, object>> RunStreamAsync(string pdfPath, string runId, string outputDir)
    {
        var channel = Channel.CreateUnbounded<Dictionary<string, object>>();
        _ = RunStreamCoreAsync(channel.Writer, pdfPath, runId, outputDir);
        return channel.Reader.ReadAllAsync();
    }

    private async Task RunStreamCoreAsync(ChannelWriter<Dictionary<string, object>> writer,
        string pdfPath, string runId, string outputDir)
    {
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
            long renderMs = 0, detectMs = 0, traceMs = 0, ocrMs = 0, llmMs = 0, mergeMs = 0, overlayMs = 0;
            int llmInputTokens = 0, llmOutputTokens = 0, llmTotalTokens = 0, llmCalls = 0;

            // Step 1: Render pages
            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "step", ["step"] = 1, ["totalSteps"] = 7,
                ["name"] = "render", ["message"] = "Rendering PDF pages..."
            });
            stepSw.Restart();
            using var renderer = new PdfRendererService(_config.PdfRenderDpi);
            var pageImages = renderer.RenderAllPages(pdfPath);
            renderMs = stepSw.ElapsedMilliseconds;

            for (int i = 0; i < pageImages.Count; i++)
                PdfRendererService.SaveImage(pageImages[i], Path.Combine(pagesDir, $"page_{i + 1}.png"));

            int pageCount = pageImages.Count;
            int imgW = pageImages[0].Width, imgH = pageImages[0].Height;

            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "stepComplete", ["step"] = 1, ["name"] = "render",
                ["durationMs"] = renderMs,
                ["detail"] = new Dictionary<string, object> { ["pageCount"] = pageCount, ["imageSize"] = $"{imgW}x{imgH}" }
            });

            // Step 2: Detect bubbles on page 1
            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "step", ["step"] = 2, ["totalSteps"] = 7,
                ["name"] = "detect", ["message"] = "Detecting bubbles and OCR-ing bubble numbers..."
            });
            stepSw.Restart();
            var bubbleDetector = new BubbleDetectionService(_config);
            var bubbles = bubbleDetector.DetectBubbles(pageImages[0], pageNumber: 1);

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

            var (bubbleOcr2, tableOcrSvc2) = CreateOcrServices();
            using var ocrDisp2 = bubbleOcr2 as IDisposable;
            using var tableDisp2 = tableOcrSvc2 as IDisposable;
            var ocrResults = bubbleOcr2.ExtractAll(rawCropsDir);
            detectMs = stepSw.ElapsedMilliseconds;

            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "stepComplete", ["step"] = 2, ["name"] = "detect",
                ["durationMs"] = detectMs,
                ["detail"] = new Dictionary<string, object> { ["bubbleCount"] = bubbles.Count }
            });

            // Step 3: Table OCR (pages 2+)
            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "step", ["step"] = 3, ["totalSteps"] = 7,
                ["name"] = "ocr", ["message"] = "OCR-ing table data..."
            });
            stepSw.Restart();
            var tesseractDimensions = new Dictionary<int, string>();
            if (tableOcrSvc2 is AzureTableOcrService azureTable2)
            {
                var pdfBytes = File.ReadAllBytes(pdfPath);
                tesseractDimensions = azureTable2.ExtractBalloonDimensionsFromPdf(pdfBytes);
            }
            else if (tableOcrSvc2 is TableOcrService localTable2)
            {
                for (int i = 1; i < pageImages.Count; i++)
                {
                    var pageDims = localTable2.ExtractBalloonDimensions(pageImages[i]);
                    foreach (var (num, dim) in pageDims)
                        tesseractDimensions.TryAdd(num, dim);
                }
            }
            ocrMs = stepSw.ElapsedMilliseconds;

            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "stepComplete", ["step"] = 3, ["name"] = "ocr",
                ["durationMs"] = ocrMs,
                ["detail"] = new Dictionary<string, object> { ["dimensionCount"] = tesseractDimensions.Count }
            });

            // Step 4: Trace leader lines from each bubble
            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "step", ["step"] = 4, ["totalSteps"] = 7,
                ["name"] = "trace", ["message"] = "Tracing leader lines..."
            });
            stepSw.Restart();
            var tracer = new LeaderLineTracerService();
            var expandedBubbles = tracer.TraceAndExpand(bubbles, pageImages[0]);
            traceMs = stepSw.ElapsedMilliseconds;

            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "stepComplete", ["step"] = 4, ["name"] = "trace",
                ["durationMs"] = traceMs,
                ["detail"] = new Dictionary<string, object> { ["tracedCount"] = expandedBubbles.Count }
            });

            // Step 5: Vision LLM validation with progressive capture expansion
            var llmValidations = new Dictionary<int, VisionLlmService.LlmValidationResult>();
            var llmCaptureSizes = new Dictionary<int, string>();
            var endpoint = Environment.GetEnvironmentVariable("AZURE_ENDPOINT");
            var key = Environment.GetEnvironmentVariable("AZURE_KEY");
            var model = Environment.GetEnvironmentVariable("AZURE_DEPLOYMENT_NAME") ?? "gpt-5.3-codex";

            var debugDir = Path.Combine(outputDir, "debug");
            Directory.CreateDirectory(debugDir);

            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "step", ["step"] = 5, ["totalSteps"] = 7,
                ["name"] = "validate", ["message"] = "Validating dimensions with Vision LLM..."
            });
            stepSw.Restart();
            if (!string.IsNullOrEmpty(key) && !string.IsNullOrEmpty(endpoint))
            {
                var responsesClient = new ResponsesClient(
                    model,
                    new System.ClientModel.ApiKeyCredential(key),
                    new OpenAIClientOptions { Endpoint = new Uri($"{endpoint.TrimEnd('/')}/openai/v1/") });
                var visionService = new VisionLlmService(responsesClient);

                var expandedByIdx = new Dictionary<int, DetectedRegion>();
                for (int i = 0; i < expandedBubbles.Count; i++)
                    expandedByIdx[i] = expandedBubbles[i];

                var captureSteps = LeaderLineTracerService.CaptureSteps;

                foreach (var (file, number) in ocrResults)
                {
                    if (!number.HasValue) continue;
                    int cropIdx = int.Parse(Path.GetFileNameWithoutExtension(file).Replace("bubble_", "")) - 1;
                    if (cropIdx < 0 || cropIdx >= bubbles.Count) continue;

                    var tableDim = tesseractDimensions.GetValueOrDefault(number.Value);

                    if (!expandedByIdx.TryGetValue(cropIdx, out var eb)) continue;
                    if (eb.LeaderDirection is null) continue;
                    var (dx, dy) = (eb.LeaderDirection.Value.Dx, eb.LeaderDirection.Value.Dy);

                    var origBb = bubbles[cropIdx].BoundingBox;
                    int bcx = origBb.X + origBb.Width / 2;
                    int bcy = origBb.Y + origBb.Height / 2;
                    int bRadius = origBb.Width / 2;

                    if (tableDim is not null)
                    {
                        VisionLlmService.LlmValidationResult? lastValidation = null;
                        string? finalCaptureSize = null;
                        foreach (var (capW, capH) in captureSteps)
                        {
                            var cap = LeaderLineTracerService.PlaceCaptureBox(
                                bcx, bcy, bRadius, dx, dy, capW, capH, imgW, imgH);
                            int x1 = Math.Max(0, cap.X);
                            int y1 = Math.Max(0, cap.Y);
                            int x2 = Math.Min(imgW, cap.X + cap.Width);
                            int y2 = Math.Min(imgH, cap.Y + cap.Height);
                            if (x2 - x1 < 4 || y2 - y1 < 4) continue;

                            using var regionCrop = new Mat(pageImages[0], new Rect(x1, y1, x2 - x1, y2 - y1));
                            var cropBytes = regionCrop.ToBytes(".png");

                            Cv2.ImWrite(Path.Combine(debugDir,
                                $"capture_bubble_{number.Value:D3}_{capW}x{capH}.png"), regionCrop);

                            var validation = await visionService.ValidateDimension(cropBytes, number.Value, tableDim);
                            lastValidation = validation;
                            finalCaptureSize = $"{capW}x{capH}";
                            llmInputTokens += validation.InputTokens;
                            llmOutputTokens += validation.OutputTokens;
                            llmTotalTokens += validation.TotalTokens;
                            llmCalls++;

                            if (validation.Matches)
                            {
                                await writer.WriteAsync(new Dictionary<string, object>
                                {
                                    ["type"] = "bubble", ["bubbleNumber"] = number.Value,
                                    ["captureSize"] = finalCaptureSize, ["status"] = "match",
                                    ["tableDim"] = tableDim,
                                    ["observed"] = validation.ObservedDimension,
                                    ["confidence"] = validation.Confidence
                                });
                                break;
                            }
                            else
                            {
                                bool isLast = (capW, capH) == captureSteps[^1];
                                await writer.WriteAsync(new Dictionary<string, object>
                                {
                                    ["type"] = "bubble", ["bubbleNumber"] = number.Value,
                                    ["captureSize"] = finalCaptureSize,
                                    ["status"] = isLast ? "bestGuess" : "expanding",
                                    ["tableDim"] = tableDim,
                                    ["observed"] = validation.ObservedDimension ?? "",
                                    ["confidence"] = validation.Confidence
                                });
                            }
                        }

                        if (lastValidation is not null)
                        {
                            llmValidations[number.Value] = lastValidation;
                            if (finalCaptureSize is not null)
                                llmCaptureSizes[number.Value] = finalCaptureSize;
                        }
                    }
                    else
                    {
                        // Discovery mode: table OCR missed this entry
                        var (capW, capH) = captureSteps[0];
                        var cap = LeaderLineTracerService.PlaceCaptureBox(
                            bcx, bcy, bRadius, dx, dy, capW, capH, imgW, imgH);
                        int x1 = Math.Max(0, cap.X);
                        int y1 = Math.Max(0, cap.Y);
                        int x2 = Math.Min(imgW, cap.X + cap.Width);
                        int y2 = Math.Min(imgH, cap.Y + cap.Height);
                        if (x2 - x1 >= 4 && y2 - y1 >= 4)
                        {
                            using var regionCrop = new Mat(pageImages[0], new Rect(x1, y1, x2 - x1, y2 - y1));
                            var cropBytes = regionCrop.ToBytes(".png");

                            Cv2.ImWrite(Path.Combine(debugDir,
                                $"capture_bubble_{number.Value:D3}_{capW}x{capH}.png"), regionCrop);

                            var discovery = await visionService.DiscoverDimension(cropBytes, number.Value);
                            llmInputTokens += discovery.InputTokens;
                            llmOutputTokens += discovery.OutputTokens;
                            llmTotalTokens += discovery.TotalTokens;
                            llmCalls++;

                            await writer.WriteAsync(new Dictionary<string, object>
                            {
                                ["type"] = "bubble", ["bubbleNumber"] = number.Value,
                                ["captureSize"] = $"{capW}x{capH}",
                                ["status"] = "discovered",
                                ["tableDim"] = "",
                                ["observed"] = discovery.ObservedDimension,
                                ["confidence"] = discovery.Confidence
                            });

                            llmValidations[number.Value] = discovery with
                            {
                                Notes = $"[Table OCR miss] {discovery.Notes}"
                            };
                            llmCaptureSizes[number.Value] = $"{capW}x{capH}";
                        }
                    }
                }
            }
            llmMs = stepSw.ElapsedMilliseconds;

            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "stepComplete", ["step"] = 5, ["name"] = "validate",
                ["durationMs"] = llmMs,
                ["detail"] = new Dictionary<string, object> { ["llmCalls"] = llmCalls, ["validatedCount"] = llmValidations.Count }
            });

            // Step 6: Merge results
            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "step", ["step"] = 6, ["totalSteps"] = 7,
                ["name"] = "merge", ["message"] = "Merging OCR + LLM validation results..."
            });
            stepSw.Restart();
            var dimensionMap = new Dictionary<int, DimensionMatch>();
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
                    Cx = cx, Cy = cy, Radius = r,
                    BoundingBox = bb
                });

                var tessVal = tesseractDimensions.GetValueOrDefault(number.Value);
                var validation = llmValidations.GetValueOrDefault(number.Value);

                bool? llmMatches = validation?.Matches;
                string? llmObserved = validation?.ObservedDimension;
                double llmConfidence = validation?.Confidence ?? 0.0;
                string? llmNotes = validation?.Notes;
                bool hasConflict = validation is not null && !validation.Matches;

                double confidence;
                if (validation is not null && validation.Matches)
                    confidence = llmConfidence;
                else if (tessVal is not null && llmObserved is not null && llmObserved.Length > 0)
                    confidence = DimensionMatcher.ConfidenceScore(tessVal, llmObserved);
                else if (validation is not null)
                    confidence = llmConfidence;
                else
                    confidence = 0.0;

                string source = (tessVal, validation) switch
                {
                    (not null, not null) => "Table+Validated",
                    (not null, null) => "TableOnly",
                    (null, not null) => "LLMOnly",
                    _ => "None"
                };

                dimensionMap[number.Value] = new DimensionMatch
                {
                    BalloonNo = number.Value,
                    Dimension = tessVal ?? llmObserved,
                    Source = source,
                    TesseractValue = tessVal,
                    LlmObservedValue = llmObserved,
                    LlmMatches = llmMatches,
                    LlmConfidence = Math.Round(llmConfidence, 4),
                    LlmNotes = llmNotes,
                    HasConflict = hasConflict,
                    Confidence = Math.Round(confidence, 4),
                    CaptureSize = llmCaptureSizes.GetValueOrDefault(number.Value),
                };
            }
            mergeMs = stepSw.ElapsedMilliseconds;

            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "stepComplete", ["step"] = 6, ["name"] = "merge",
                ["durationMs"] = mergeMs,
                ["detail"] = new Dictionary<string, object> { ["dimensionCount"] = dimensionMap.Count }
            });

            // Step 7: Generate overlay image
            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "step", ["step"] = 7, ["totalSteps"] = 7,
                ["name"] = "overlay", ["message"] = "Generating overlay images..."
            });
            stepSw.Restart();
            GenerateOverlay(pageImages[0], bubbleResults, dimensionMap,
                Path.Combine(overlayDir, "page_1_overlay.png"));
            overlayMs = stepSw.ElapsedMilliseconds;

            foreach (var page in pageImages) page.Dispose();

            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "stepComplete", ["step"] = 7, ["name"] = "overlay",
                ["durationMs"] = overlayMs, ["detail"] = new Dictionary<string, object>()
            });

            totalSw.Stop();
            var result = new PipelineResult
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
                    TraceDurationMs = traceMs,
                    OcrDurationMs = ocrMs,
                    LlmDurationMs = llmMs,
                    MergeDurationMs = mergeMs,
                    PeakMemoryMb = Math.Round(GC.GetTotalMemory(false) / (1024.0 * 1024.0), 1)
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

            WriteBenchmark(outputDir, runId, filename,
                [("render", renderMs), ("detect", detectMs), ("ocr", ocrMs),
                 ("trace", traceMs), ("validate", llmMs), ("merge", mergeMs), ("overlay", overlayMs)],
                totalSw.ElapsedMilliseconds, bubbleResults.Count,
                dimensionMap.Values.Count(d => d.Dimension is not null),
                llmCalls, llmInputTokens, llmOutputTokens, llmTotalTokens);

            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "complete", ["result"] = result
            });
        }
        catch (Exception ex)
        {
            await writer.WriteAsync(new Dictionary<string, object>
            {
                ["type"] = "error", ["message"] = ex.Message
            });
        }
        finally
        {
            writer.Complete();
        }
    }

    private static void WriteBenchmark(string outputDir, string runId, string pdfFilename,
        List<(string Name, long DurationMs)> steps, long totalMs,
        int bubbleCount, int matchedBubbles, int llmCalls,
        int llmInputTokens, int llmOutputTokens, int llmTotalTokens)
    {
        var benchmark = new
        {
            runId,
            backend = "dotnet",
            pdfFilename,
            timestamp = DateTime.UtcNow.ToString("o"),
            steps = steps.Select(s => new { name = s.Name, durationMs = s.DurationMs }).ToArray(),
            totalDurationMs = totalMs,
            bubbleCount,
            matchedBubbles,
            llmCalls,
            tokenUsage = new { input = llmInputTokens, output = llmOutputTokens, total = llmTotalTokens }
        };
        var json = JsonSerializer.Serialize(benchmark, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(outputDir, "benchmark.json"), json);
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
