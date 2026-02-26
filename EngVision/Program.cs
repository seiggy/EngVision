using System.Text.Json;
using EngVision.Models;
using EngVision.Services;
using OpenAI;
using OpenAI.Responses;
using OpenCvSharp;

// ── Load .env file if present ──────────────────────────────────────────────────
var envFile = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".env");
if (File.Exists(envFile))
{
    foreach (var line in File.ReadAllLines(envFile))
    {
        var trimmed = line.Trim();
        if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith('#')) continue;
        var eqIdx = trimmed.IndexOf('=');
        if (eqIdx <= 0) continue;
        var key = trimmed[..eqIdx].Trim();
        var val = trimmed[(eqIdx + 1)..].Trim();
        Environment.SetEnvironmentVariable(key, val);
    }
    Console.WriteLine($"  Loaded .env from {Path.GetFullPath(envFile)}");
}

// ── Check for --analyze mode ───────────────────────────────────────────────────
if (args.Contains("--analyze"))
{
    await RunBubbleAnalysis(args);
    return;
}

// ── Check for --diag-missed mode ──────────────────────────────────────────────
if (args.Contains("--diag-missed"))
{
    DiagnoseMissedNodes();
    return;
}

// ── Configuration ──────────────────────────────────────────────────────────────
var config = new EngVisionConfig
{
    OpenAIApiKey = Environment.GetEnvironmentVariable("AZURE_KEY")
        ?? Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "",
    OpenAIModel = Environment.GetEnvironmentVariable("AZURE_DEPLOYMENT_NAME")
        ?? Environment.GetEnvironmentVariable("OPENAI_MODEL") ?? "gpt-4o",
    OpenAIEndpoint = Environment.GetEnvironmentVariable("AZURE_ENDPOINT")
        ?? Environment.GetEnvironmentVariable("OPENAI_ENDPOINT"),
    PdfRenderDpi = 300,
    OutputDirectory = Path.Combine(AppContext.BaseDirectory, "Output"),
    HoughMinRadius = 12,
    HoughMaxRadius = 50,
    HoughParam1 = 120,
    HoughParam2 = 25,
    BubbleContextPadding = 150
};

// Resolve PDF path
string pdfPath;
if (args.Length > 0 && File.Exists(args[0]))
{
    pdfPath = args[0];
}
else
{
    // Default to sample doc
    var sampleDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "sample_docs");
    pdfPath = Directory.GetFiles(sampleDir, "*.pdf").FirstOrDefault()
        ?? throw new FileNotFoundException("No PDF found. Pass a PDF path as argument or place one in sample_docs/");
}

Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║          EngVision - CAD Dimensional Analysis               ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine($"  PDF: {pdfPath}");
Console.WriteLine($"  Output: {config.OutputDirectory}");
Console.WriteLine();

// ── Step 1: Render PDF to images ───────────────────────────────────────────────
Console.WriteLine("Step 1: Rendering PDF pages to images...");
using var renderer = new PdfRendererService(config.PdfRenderDpi);
var pageImages = renderer.RenderAllPages(pdfPath);
Console.WriteLine($"  Rendered {pageImages.Count} pages\n");

var exportService = new SegmentExportService(config.OutputDirectory);

// Save full page images
for (int i = 0; i < pageImages.Count; i++)
{
    exportService.SaveFullPage(pageImages[i], i + 1);
}

// ── Color diagnostic at ground truth locations ────────────────────────────────
if (args.Contains("--color-diag"))
{
    var diagGtPath = Path.Combine(Path.GetDirectoryName(pdfPath)!,
        Path.GetFileNameWithoutExtension(pdfPath) + "_ground_truth.json");
    if (File.Exists(diagGtPath))
    {
        var gtJson = File.ReadAllText(diagGtPath);
        var gtDict = JsonSerializer.Deserialize<Dictionary<string, List<GroundTruthAnnotation>>>(gtJson,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? [];
        var gtAnns = gtDict.Values.SelectMany(v => v).ToList();
        using var hsvImg = new Mat();
        Cv2.CvtColor(pageImages[0], hsvImg, ColorConversionCodes.BGR2HSV);
        using var bgrImg = pageImages[0];

        Console.WriteLine("\n=== COLOR DIAGNOSTIC AT GT BUBBLE CENTERS ===");
        // Save raw color crop for visual inspection
        var diagCropsDir = Path.Combine(config.OutputDirectory, "debug", "color_crops");
        Directory.CreateDirectory(diagCropsDir);
        foreach (var ann in gtAnns.Where(a => a.BubbleCenter != null).Take(5))
        {
            int cx = (int)ann.BubbleCenter!.X, cy = (int)ann.BubbleCenter.Y;
            int pad = 40;
            int x1 = Math.Max(0, cx - pad), y1 = Math.Max(0, cy - pad);
            int x2 = Math.Min(pageImages[0].Width, cx + pad), y2 = Math.Min(pageImages[0].Height, cy + pad);
            using var rawCrop = new Mat(pageImages[0], new Rect(x1, y1, x2 - x1, y2 - y1));
            Cv2.ImWrite(Path.Combine(diagCropsDir, $"raw_{ann.BubbleNumber}.png"), rawCrop);
        }
        Console.WriteLine($"  Saved raw color crops to {diagCropsDir}");
        
        // Sample a thin annulus at multiple radii to find the circle outline color
        foreach (var ann in gtAnns.Where(a => a.BubbleCenter != null).Take(5))
        {
            int cx = (int)ann.BubbleCenter!.X, cy = (int)ann.BubbleCenter.Y;
            Console.Write($"  Bubble #{ann.BubbleNumber} at ({cx},{cy}): ");
            for (int r = 14; r <= 20; r++)
            {
                var samples = new List<(int B, int G, int R)>();
                for (int angle = 0; angle < 360; angle += 5)
                {
                    int px = cx + (int)(r * Math.Cos(angle * Math.PI / 180));
                    int py = cy + (int)(r * Math.Sin(angle * Math.PI / 180));
                    if (px < 0 || px >= pageImages[0].Width || py < 0 || py >= pageImages[0].Height) continue;
                    var bgr = pageImages[0].At<Vec3b>(py, px);
                    samples.Add((bgr[0], bgr[1], bgr[2]));
                }
                double avgB = samples.Average(v => v.B);
                double avgG = samples.Average(v => v.G);
                double avgR = samples.Average(v => v.R);
                int darkPx = samples.Count(v => v.B < 128 && v.G < 128 && v.R < 128);
                Console.Write($"r={r}: BGR({avgB:F0},{avgG:F0},{avgR:F0}) dk={darkPx}  ");
            }
            Console.WriteLine();
        }
        Console.WriteLine();
    }
}

// ── Step 2: Detect bubbles on page 1 ──────────────────────────────────────────
Console.WriteLine("Step 2: Detecting bubbles on page 1...");
var bubbleDetector = new BubbleDetectionService(config);
bubbleDetector.Verbose = args.Contains("--verbose") || args.Contains("-v");
var bubbles = bubbleDetector.DetectBubbles(pageImages[0], pageNumber: 1);
Console.WriteLine($"  Found {bubbles.Count} bubbles\n");

// ── Step 2b: Save individual bubble crops for visual inspection ─────────────
Console.WriteLine("Step 2b: Saving bubble circle crops for inspection...");
var cropsDir = Path.Combine(config.OutputDirectory, "debug", "bubble_crops");
var rawCropsDir = Path.Combine(config.OutputDirectory, "debug", "bubble_crops_raw");
if (Directory.Exists(cropsDir)) Directory.Delete(cropsDir, true);
if (Directory.Exists(rawCropsDir)) Directory.Delete(rawCropsDir, true);
Directory.CreateDirectory(cropsDir);
Directory.CreateDirectory(rawCropsDir);

using var pageGray = new Mat();
Cv2.CvtColor(pageImages[0], pageGray, ColorConversionCodes.BGR2GRAY);

foreach (var b in bubbles)
{
    var bb = b.BoundingBox;
    int cx = bb.X + bb.Width / 2, cy = bb.Y + bb.Height / 2;
    int r = bb.Width / 2;
    // Crop tightly: just the bubble circle + minimal 2px margin (~30x30)
    int pad = 2;
    int x1 = Math.Max(0, cx - r - pad), y1 = Math.Max(0, cy - r - pad);
    int x2 = Math.Min(pageImages[0].Width, cx + r + pad), y2 = Math.Min(pageImages[0].Height, cy + r + pad);
    using var crop = new Mat(pageImages[0], new Rect(x1, y1, x2 - x1, y2 - y1));

    // Save raw crop (for OCR - no annotations)
    var rawPath = Path.Combine(rawCropsDir, $"bubble_{b.BubbleNumber:D3}.png");
    Cv2.ImWrite(rawPath, crop);

    // Draw circle overlay on crop for visual inspection
    using var annotated = crop.Clone();
    Cv2.Circle(annotated, new Point(cx - x1, cy - y1), r, new Scalar(0, 0, 255), 1);
    Cv2.PutText(annotated, $"#{b.BubbleNumber}", new Point(2, 12),
        HersheyFonts.HersheySimplex, 0.35, new Scalar(255, 0, 0), 1);
    var path = Path.Combine(cropsDir, $"bubble_{b.BubbleNumber:D3}.png");
    Cv2.ImWrite(path, annotated);
}
Console.WriteLine($"  Saved {bubbles.Count} crops to {cropsDir}\n");

// ── Step 2c: OCR bubble numbers ───────────────────────────────────────────────
Console.WriteLine("Step 2c: OCR-ing bubble numbers...");
var tessDataPath = Path.Combine(AppContext.BaseDirectory, "tessdata");
using var ocrService = new BubbleOcrService(tessDataPath);
var ocrResults = ocrService.ExtractAll(rawCropsDir);
int ocrSuccess = ocrResults.Count(r => r.Value.HasValue);
Console.WriteLine($"  OCR results: {ocrSuccess}/{ocrResults.Count} bubbles identified");
foreach (var (file, number) in ocrResults.OrderBy(r => r.Key))
{
    Console.WriteLine($"    {file} → {(number.HasValue ? $"Bubble #{number}" : "FAILED")}");
}
Console.WriteLine();

// ── Step 3: Trace leader lines and expand to associated dimension text ──────────
Console.WriteLine("Step 3: Tracing leader lines to find associated dimension text...");
var leaderTracer = new LeaderLineTracerService();
var bubbleRegions = leaderTracer.TraceAndExpand(bubbles, pageImages[0]);
var exportedBubbles = exportService.ExportRegions(bubbleRegions, pageImages[0]);

// Also save the raw bubble-only detections for the debug visualization
exportService.SaveDebugVisualization(bubbles, pageImages[0], 1);
// Save expanded regions as a second debug view
exportService.SaveDebugVisualization(bubbleRegions, pageImages[0], pageNumber: 1, suffix: "_expanded");
Console.WriteLine($"  Exported {exportedBubbles.Count} bubble+figure segments\n");

// ── Step 4: Detect tables on pages 2-4 ────────────────────────────────────────
Console.WriteLine("Step 4: Detecting table regions on subsequent pages...");
var tableDetector = new TableDetectionService(config);
var allTableRegions = new List<DetectedRegion>();

for (int i = 1; i < pageImages.Count; i++)
{
    var pageNum = i + 1;
    var tables = tableDetector.DetectTables(pageImages[i], pageNum);

    if (tables.Count == 0)
    {
        // Fall back to full page if no tables detected
        Console.WriteLine($"  Page {pageNum}: no table structure detected, using full page");
        tables = [tableDetector.GetFullPageRegion(pageImages[i], pageNum)];
    }

    var exportedTables = exportService.ExportRegions(tables, pageImages[i]);
    exportService.SaveDebugVisualization(tables, pageImages[i], pageNum);
    allTableRegions.AddRange(exportedTables);
}
Console.WriteLine($"  Total table regions: {allTableRegions.Count}\n");

// ── Step 4b: OCR table data — extract balloon# → dimension mapping ─────────
Console.WriteLine("Step 4b: OCR-ing table data (balloon# → dimension)...");
using var tableOcr = new TableOcrService(tessDataPath);
var balloonDimensions = new Dictionary<int, string>();
for (int i = 1; i < pageImages.Count; i++)
{
    var pageNum = i + 1;
    Console.WriteLine($"  Page {pageNum}:");
    var pageDims = tableOcr.ExtractBalloonDimensions(pageImages[i]);
    foreach (var (num, dim) in pageDims.OrderBy(kv => kv.Key))
    {
        balloonDimensions[num] = dim;
        Console.WriteLine($"    Balloon #{num} → \"{dim}\"");
    }
}
Console.WriteLine($"  Total: {balloonDimensions.Count} balloon→dimension mappings\n");

// ── Step 4c: Combine bubble detections + OCR numbers + table dimensions ──────
Console.WriteLine("Step 4c: Building bubble→location→dimension map...");
var bubbleMap = new Dictionary<int, (int Cx, int Cy, int Radius, string? Dimension)>();
foreach (var (file, number) in ocrResults)
{
    if (!number.HasValue) continue;
    // Find the bubble detection that corresponds to this crop file
    int cropIdx = int.Parse(Path.GetFileNameWithoutExtension(file).Replace("bubble_", "")) - 1;
    if (cropIdx < 0 || cropIdx >= bubbles.Count) continue;
    var bb = bubbles[cropIdx].BoundingBox;
    int cx = bb.X + bb.Width / 2, cy = bb.Y + bb.Height / 2;
    int r = bb.Width / 2;
    string? dim = balloonDimensions.GetValueOrDefault(number.Value);
    bubbleMap[number.Value] = (cx, cy, r, dim);
}
Console.WriteLine($"  Mapped {bubbleMap.Count} bubbles:");
int withDim = 0, withoutDim = 0;
foreach (var (num, info) in bubbleMap.OrderBy(kv => kv.Key))
{
    if (info.Dimension is not null)
    {
        Console.WriteLine($"    #{num} at ({info.Cx},{info.Cy}) → \"{info.Dimension}\"");
        withDim++;
    }
    else
    {
        Console.WriteLine($"    #{num} at ({info.Cx},{info.Cy}) → [no table match]");
        withoutDim++;
    }
}
Console.WriteLine($"  {withDim} with dimension, {withoutDim} without\n");

// ── Step 5: Export metadata ────────────────────────────────────────────────────
Console.WriteLine("Step 5: Exporting detection metadata...");
var allRegions = exportedBubbles.Concat(allTableRegions).ToList();
var metadataPath = Path.Combine(config.OutputDirectory, "detections.json");
var jsonOptions = new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
await File.WriteAllTextAsync(metadataPath, JsonSerializer.Serialize(allRegions, jsonOptions));
Console.WriteLine($"  Metadata saved to: {metadataPath}\n");

// ── Step 5b: Evaluate against ground truth if available ────────────────────────
var gtPath = Path.ChangeExtension(pdfPath, null) + "_ground_truth.json";
if (File.Exists(gtPath))
{
    Console.WriteLine("Step 5b: Evaluating detection against ground truth...");
    var gtJson = await File.ReadAllTextAsync(gtPath);
    var gtData = JsonSerializer.Deserialize<Dictionary<string, List<GroundTruthAnnotation>>>(gtJson, jsonOptions);

    // Find the page 1 ground truth key
    var gtKey = gtData?.Keys.FirstOrDefault(k => k.EndsWith(":1"));
    if (gtKey is not null && gtData![gtKey] is { Count: > 0 } gtAnnotations)
    {
        Console.WriteLine($"  Ground truth: {gtAnnotations.Count} annotations");
        Console.WriteLine($"  Detected bubbles: {bubbles.Count}");

        // Match: a detection matches a GT annotation if the detected bubble center
        // falls inside the GT bounding box (expanded by a tolerance margin),
        // OR the GT center is within the detected bubble radius + margin.
        const int margin = 40;
        int matched = 0, missed = 0;
        var usedDetections = new HashSet<int>();

        foreach (var gt in gtAnnotations)
        {
            var gtBb = gt.BoundingBox;
            // Use explicit bubble center if annotated, otherwise fall back to box center
            int gtCx = gt.BubbleCenter?.X ?? (gtBb.X + gtBb.Width / 2);
            int gtCy = gt.BubbleCenter?.Y ?? (gtBb.Y + gtBb.Height / 2);
            bool hasBubbleCenter = gt.BubbleCenter is not null;

            // Expanded GT region for matching
            int gx1 = gtBb.X - margin, gy1 = gtBb.Y - margin;
            int gx2 = gtBb.X + gtBb.Width + margin, gy2 = gtBb.Y + gtBb.Height + margin;

            double bestDist = double.MaxValue;
            int bestIdx = -1;

            for (int i = 0; i < bubbles.Count; i++)
            {
                if (usedDetections.Contains(i)) continue;
                var db = bubbles[i].BoundingBox;
                var dCx = db.X + db.Width / 2;
                var dCy = db.Y + db.Height / 2;

                if (hasBubbleCenter)
                {
                    // With explicit bubble center: match if detection is within 50px of the bubble
                    var dist = Math.Sqrt(Math.Pow(gtCx - dCx, 2) + Math.Pow(gtCy - dCy, 2));
                    if (dist < 50 && dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx = i;
                    }
                }
                else
                {
                    // Without bubble center: use box-based matching
                    bool insideGt = dCx >= gx1 && dCx <= gx2 && dCy >= gy1 && dCy <= gy2;
                    var dist = Math.Sqrt(Math.Pow(gtCx - dCx, 2) + Math.Pow(gtCy - dCy, 2));
                    if (insideGt || dist < Math.Max(gtBb.Width, gtBb.Height) / 2.0 + margin)
                    {
                        if (dist < bestDist) { bestDist = dist; bestIdx = i; }
                    }
                }
            }

            if (bestIdx >= 0)
            {
                matched++;
                usedDetections.Add(bestIdx);
                Console.WriteLine($"    ✓ Bubble #{gt.BubbleNumber}: matched (dist={bestDist:F0}px)");
            }
            else
            {
                // Find nearest detection for diagnostic
                double nearestDist = double.MaxValue;
                for (int i = 0; i < bubbles.Count; i++)
                {
                    var db = bubbles[i].BoundingBox;
                    var d = Math.Sqrt(Math.Pow(gtCx - db.X - db.Width / 2, 2) + Math.Pow(gtCy - db.Y - db.Height / 2, 2));
                    if (d < nearestDist) nearestDist = d;
                }
                missed++;
                Console.WriteLine($"    ✗ Bubble #{gt.BubbleNumber}: MISSED (size={gtBb.Width}x{gtBb.Height}, nearest det={nearestDist:F0}px)");
            }
        }

        var falsePositives = bubbles.Count - usedDetections.Count;
        var precision = bubbles.Count > 0 ? (double)matched / bubbles.Count : 0;
        var recall = gtAnnotations.Count > 0 ? (double)matched / gtAnnotations.Count : 0;
        var f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

        Console.WriteLine();
        Console.WriteLine("  ╔════════════════════════════════════════╗");
        Console.WriteLine("  ║       DETECTION ACCURACY REPORT       ║");
        Console.WriteLine("  ╠════════════════════════════════════════╣");
        Console.WriteLine($"  ║  Ground Truth:     {gtAnnotations.Count,4}                ║");
        Console.WriteLine($"  ║  Detected:         {bubbles.Count,4}                ║");
        Console.WriteLine($"  ║  Matched:          {matched,4}                ║");
        Console.WriteLine($"  ║  Missed:           {missed,4}                ║");
        Console.WriteLine($"  ║  False Positives:  {falsePositives,4}                ║");
        Console.WriteLine($"  ║  Precision:        {precision,7:P1}            ║");
        Console.WriteLine($"  ║  Recall:           {recall,7:P1}            ║");
        Console.WriteLine($"  ║  F1 Score:         {f1,7:P1}            ║");
        Console.WriteLine("  ╚════════════════════════════════════════╝");
        Console.WriteLine();
    }
}

// ── Step 6: Vision LLM table extraction (if API key is configured) ─────────────
if (string.IsNullOrEmpty(config.OpenAIApiKey))
{
    Console.WriteLine("Step 6: SKIPPED - Set AZURE_KEY environment variable to enable vision LLM extraction");
    Console.WriteLine("  Using Tesseract OCR data only for validation.\n");

    // ── Step 7: Validate with OCR-only data ──────────────────────────────────────
    Console.WriteLine("Step 7: Validation report (Tesseract OCR only)...");
    PrintValidationReport(bubbleMap);
}
else
{
    Console.WriteLine($"Step 6: Vision LLM table extraction via {config.OpenAIModel}...");

    ResponsesClient responsesClient;
    if (!string.IsNullOrEmpty(config.OpenAIEndpoint))
    {
        var endpoint = config.OpenAIEndpoint.TrimEnd('/');
        responsesClient = new ResponsesClient(
            config.OpenAIModel,
            new System.ClientModel.ApiKeyCredential(config.OpenAIApiKey),
            new OpenAIClientOptions { Endpoint = new Uri($"{endpoint}/openai/v1/") });
    }
    else
    {
        responsesClient = new ResponsesClient(config.OpenAIModel, config.OpenAIApiKey);
    }

    var visionService = new VisionLlmService(responsesClient);

    // Send each table page image to the LLM for balloon→dimension extraction
    var llmDimensions = new Dictionary<int, string>();
    for (int i = 1; i < pageImages.Count; i++)
    {
        var pageNum = i + 1;
        Console.Write($"  Page {pageNum}...");

        // Encode page image to PNG bytes
        var pageBytes = pageImages[i].ToBytes(".png");
        var pageDims = await visionService.ExtractBalloonDimensions(pageBytes, pageNum);

        foreach (var (num, dim) in pageDims.OrderBy(kv => kv.Key))
        {
            llmDimensions.TryAdd(num, dim);
        }
        Console.WriteLine($" {pageDims.Count} balloon→dimension pairs");
    }
    Console.WriteLine($"  LLM total: {llmDimensions.Count} balloon→dimension mappings\n");

    // ── Step 6b: Merge Tesseract OCR + LLM results ───────────────────────────────
    Console.WriteLine("Step 6b: Merging Tesseract OCR + Vision LLM results...");
    var mergedDimensions = new Dictionary<int, string>(balloonDimensions); // start with Tesseract
    int llmFills = 0, llmConflicts = 0;
    foreach (var (num, dim) in llmDimensions)
    {
        if (!mergedDimensions.ContainsKey(num))
        {
            mergedDimensions[num] = dim;
            llmFills++;
            Console.WriteLine($"  + LLM filled gap: #{num} → \"{dim}\"");
        }
        else if (!DimensionMatcher.AreSimilar(mergedDimensions[num], dim))
        {
            llmConflicts++;
            Console.WriteLine($"  ~ Conflict #{num}: Tesseract=\"{mergedDimensions[num]}\" vs LLM=\"{dim}\" (keeping Tesseract)");
        }
    }
    Console.WriteLine($"  Merged: {mergedDimensions.Count} total ({llmFills} LLM fills, {llmConflicts} conflicts)\n");

    // Update bubbleMap with merged dimensions
    var mergedBubbleMap = new Dictionary<int, (int Cx, int Cy, int Radius, string? Dimension)>();
    foreach (var (num, info) in bubbleMap)
    {
        var dim = mergedDimensions.GetValueOrDefault(num);
        mergedBubbleMap[num] = (info.Cx, info.Cy, info.Radius, dim ?? info.Dimension);
    }

    // Save merged data
    var mergedPath = Path.Combine(config.OutputDirectory, "merged_balloon_dimensions.json");
    var mergedData = mergedBubbleMap.OrderBy(kv => kv.Key).Select(kv => new
    {
        BalloonNo = kv.Key,
        kv.Value.Cx,
        kv.Value.Cy,
        kv.Value.Radius,
        kv.Value.Dimension,
        Source = balloonDimensions.ContainsKey(kv.Key)
            ? (llmDimensions.ContainsKey(kv.Key) ? "Both" : "Tesseract")
            : (llmDimensions.ContainsKey(kv.Key) ? "LLM" : "None")
    });
    await File.WriteAllTextAsync(mergedPath, JsonSerializer.Serialize(mergedData, jsonOptions));
    Console.WriteLine($"  Merged data saved to: {mergedPath}\n");

    // ── Step 7: Validation report ────────────────────────────────────────────────
    Console.WriteLine("Step 7: Validation report (merged OCR + LLM)...");
    PrintValidationReport(mergedBubbleMap);
}

// Cleanup
foreach (var page in pageImages) page.Dispose();

Console.WriteLine("\nDone! Check the Output directory for results.");

// ── Validation Report Helper ───────────────────────────────────────────────────
static void PrintValidationReport(Dictionary<int, (int Cx, int Cy, int Radius, string? Dimension)> map)
{
    int withDim = map.Values.Count(v => v.Dimension is not null);
    int withoutDim = map.Values.Count(v => v.Dimension is null);

    Console.WriteLine();
    Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
    Console.WriteLine("║          BALLOON → DIMENSION VALIDATION REPORT              ║");
    Console.WriteLine("╠══════════════════════════════════════════════════════════════╣");

    foreach (var (num, info) in map.OrderBy(kv => kv.Key))
    {
        var status = info.Dimension is not null ? "✓" : "✗";
        var dimText = info.Dimension ?? "NO TABLE MATCH";
        Console.WriteLine($"║ {status} Balloon #{num,-3} at ({info.Cx,4},{info.Cy,4}) → {dimText,-28}║");
    }

    Console.WriteLine("╠══════════════════════════════════════════════════════════════╣");
    var coverage = map.Count > 0 ? (double)withDim / map.Count : 0;
    Console.WriteLine($"║ Coverage: {withDim}/{map.Count} balloons have dimensions ({coverage:P0})      ║");
    if (withoutDim > 0)
    {
        var missing = string.Join(", ", map.Where(kv => kv.Value.Dimension is null).Select(kv => $"#{kv.Key}"));
        Console.WriteLine($"║ Missing:  {missing,-50}║");
    }
    Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
    Console.WriteLine();
}

// ── Bubble Analysis Mode ───────────────────────────────────────────────────────
static async Task RunBubbleAnalysis(string[] args)
{
    var sampleDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "sample_docs"));
    var pdfPath = Directory.GetFiles(sampleDir, "*.pdf").First();
    var jsonOpts = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    // Fetch annotations from API or file
    List<GroundTruthAnnotation> annotations;
    var apiUrl = args.SkipWhile(a => a != "--api").Skip(1).FirstOrDefault() ?? "http://localhost:5250";
    try
    {
        using var http = new HttpClient();
        var resp = await http.GetStringAsync($"{apiUrl}/api/pdfs/{Uri.EscapeDataString(Path.GetFileName(pdfPath))}/pages/1/annotations");
        var data = JsonSerializer.Deserialize<JsonElement>(resp);
        annotations = JsonSerializer.Deserialize<List<GroundTruthAnnotation>>(
            data.GetProperty("manual").GetRawText(), jsonOpts) ?? [];
    }
    catch
    {
        Console.WriteLine("Could not fetch from API, looking for local file...");
        var gtPath = Path.ChangeExtension(pdfPath, null) + "_ground_truth.json";
        var gtJson = await File.ReadAllTextAsync(gtPath);
        var gtData = JsonSerializer.Deserialize<Dictionary<string, List<GroundTruthAnnotation>>>(gtJson, jsonOpts);
        var key = gtData?.Keys.FirstOrDefault(k => k.EndsWith(":1")) ?? "";
        annotations = gtData?[key] ?? [];
    }

    var withCenters = annotations.Where(a => a.BubbleCenter is not null).ToList();
    Console.WriteLine($"Loaded {annotations.Count} annotations ({withCenters.Count} with bubble centers)");

    // Render page 1
    var renderer = new PdfRendererService(300);
    using var page = renderer.RenderPage(pdfPath, 0);
    using var gray = new Mat();
    Cv2.CvtColor(page, gray, ColorConversionCodes.BGR2GRAY);
    Console.WriteLine($"Page size: {page.Width}x{page.Height}\n");

    Console.WriteLine("=== PIXEL ANALYSIS AT BUBBLE CENTERS ===");
    Console.WriteLine($"{"#",-5} {"Center",-14} {"Px",-4} {"R10",-7} {"R15",-7} {"R20",-7} {"Ring10",-7} {"Ring15",-7} {"Ring20",-7} {"Ctrst",-7} {"Dark%",-6}");

    foreach (var ann in withCenters)
    {
        int cx = ann.BubbleCenter!.X, cy = ann.BubbleCenter.Y;
        byte centerPx = gray.At<byte>(cy, cx);

        double[] innerMeans = new double[3];
        double[] ringMeans = new double[3];
        int[] radii = [10, 15, 20];

        for (int ri = 0; ri < 3; ri++)
        {
            int r = radii[ri];
            using var mask = new Mat(gray.Size(), MatType.CV_8UC1, Scalar.Black);
            Cv2.Circle(mask, new Point(cx, cy), r, Scalar.White, -1);
            innerMeans[ri] = Cv2.Mean(gray, mask).Val0;

            using var ringMask = new Mat(gray.Size(), MatType.CV_8UC1, Scalar.Black);
            Cv2.Circle(ringMask, new Point(cx, cy), r + 2, Scalar.White, 3);
            Cv2.Circle(ringMask, new Point(cx, cy), Math.Max(1, r - 1), Scalar.Black, -1);
            ringMeans[ri] = Cv2.Mean(gray, ringMask).Val0;
        }

        using var innerMask15 = new Mat(gray.Size(), MatType.CV_8UC1, Scalar.Black);
        Cv2.Circle(innerMask15, new Point(cx, cy), 15, Scalar.White, -1);
        using var bin = new Mat();
        Cv2.Threshold(gray, bin, 128, 255, ThresholdTypes.BinaryInv);
        using var darkInner = new Mat();
        Cv2.BitwiseAnd(bin, innerMask15, darkInner);
        int darkPx = Cv2.CountNonZero(darkInner);
        int totalPx = Cv2.CountNonZero(innerMask15);
        double darkRatio = totalPx > 0 ? (double)darkPx / totalPx : 0;
        double contrast = innerMeans[1] - ringMeans[1];

        Console.WriteLine($"{ann.BubbleNumber ?? 0,-5} ({cx,4},{cy,4})  {centerPx,-4} {innerMeans[0],-7:F1} {innerMeans[1],-7:F1} {innerMeans[2],-7:F1} {ringMeans[0],-7:F1} {ringMeans[1],-7:F1} {ringMeans[2],-7:F1} {contrast,-7:F1} {darkRatio,-6:F3}");
    }

    // HoughCircles coverage check
    Console.WriteLine("\n=== HOUGHCIRCLES COVERAGE CHECK ===");
    using var blur = new Mat();
    Cv2.GaussianBlur(gray, blur, new Size(9, 9), 2);

    var circles = Cv2.HoughCircles(blur, HoughModes.Gradient,
        dp: 1.2, minDist: 22, param1: 120, param2: 23, minRadius: 8, maxRadius: 50);
    Console.WriteLine($"HoughCircles found {circles.Length} candidates (before verification)\n");

    int found = 0, near = 0, miss = 0;
    foreach (var ann in withCenters)
    {
        int cx = ann.BubbleCenter!.X, cy = ann.BubbleCenter.Y;
        double minDist = double.MaxValue;
        CircleSegment? nearest = null;
        foreach (var c in circles)
        {
            double d = Math.Sqrt(Math.Pow(cx - c.Center.X, 2) + Math.Pow(cy - c.Center.Y, 2));
            if (d < minDist) { minDist = d; nearest = c; }
        }

        string status;
        if (minDist < 30) { status = "✓ FOUND"; found++; }
        else if (minDist < 60) { status = "~ NEAR"; near++; }
        else { status = "✗ MISS"; miss++; }

        Console.WriteLine($"  Bubble #{ann.BubbleNumber ?? 0,-3}: dist={minDist,5:F0}px  r={nearest?.Radius ?? 0,3:F0}  {status}");
    }

    Console.WriteLine($"\nSummary: {found} found, {near} near, {miss} missed out of {withCenters.Count}");
}

void DiagnoseMissedNodes()
{
    var sampleDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "sample_docs");
    var missedPath = Path.Combine(sampleDir, "missed_nodes.json");
    if (!File.Exists(missedPath)) { Console.WriteLine("missed_nodes.json not found"); return; }

    var json = File.ReadAllText(missedPath);
    var missedDict = JsonSerializer.Deserialize<Dictionary<string, List<GroundTruthAnnotation>>>(json,
        new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? [];
    var missed = missedDict.Values.SelectMany(v => v).Where(a => a.BubbleCenter != null).ToList();
    Console.WriteLine($"Loaded {missed.Count} missed nodes\n");

    var pdfPath = Directory.GetFiles(sampleDir, "*.pdf").First();
    var renderer = new PdfRendererService(300);
    using var page = renderer.RenderPage(pdfPath, 0);
    using var hsv = new Mat();
    Cv2.CvtColor(page, hsv, ColorConversionCodes.BGR2HSV);
    using var gray = new Mat();
    Cv2.CvtColor(page, gray, ColorConversionCodes.BGR2GRAY);

    // Run HoughCircles to get all candidates
    using var blur = new Mat();
    Cv2.GaussianBlur(gray, blur, new Size(9, 9), 2);
    var allCircles = new List<CircleSegment>();
    foreach (var pass in new[] {
        Cv2.HoughCircles(blur, HoughModes.Gradient, 1.2, 22, 120, 23, 12, 50),
        Cv2.HoughCircles(blur, HoughModes.Gradient, 1.0, 18, 100, 20, 8, 25),
        Cv2.HoughCircles(blur, HoughModes.Gradient, 1.5, 22, 80, 18, 10, 45)
    }) allCircles.AddRange(pass);

    var cropsDir = Path.Combine(AppContext.BaseDirectory, "Output", "debug", "missed_crops");
    if (Directory.Exists(cropsDir)) Directory.Delete(cropsDir, true);
    Directory.CreateDirectory(cropsDir);

    Console.WriteLine("=== MISSED NODE DIAGNOSTICS ===\n");

    for (int i = 0; i < missed.Count; i++)
    {
        var m = missed[i];
        int cx = (int)m.BubbleCenter!.X, cy = (int)m.BubbleCenter.Y;
        Console.WriteLine($"--- Missed #{i + 1}: center=({cx},{cy}) ---");

        // 1. Save crops at multiple sizes
        foreach (int pad in new[] { 25, 50 })
        {
            int x1 = Math.Max(0, cx - pad), y1 = Math.Max(0, cy - pad);
            int x2 = Math.Min(page.Width, cx + pad), y2 = Math.Min(page.Height, cy + pad);
            using var crop = new Mat(page, new Rect(x1, y1, x2 - x1, y2 - y1));
            // Mark center
            using var marked = crop.Clone();
            Cv2.DrawMarker(marked, new Point(cx - x1, cy - y1), new Scalar(0, 0, 255),
                MarkerTypes.Cross, 10, 2);
            Cv2.ImWrite(Path.Combine(cropsDir, $"missed_{i + 1}_pad{pad}.png"), marked);
        }

        // 2. Find nearest HoughCircles candidate
        double nearestDist = double.MaxValue;
        CircleSegment? nearest = null;
        foreach (var c in allCircles)
        {
            double d = Math.Sqrt(Math.Pow(cx - c.Center.X, 2) + Math.Pow(cy - c.Center.Y, 2));
            if (d < nearestDist) { nearestDist = d; nearest = c; }
        }
        Console.WriteLine($"  Nearest HoughCircle: dist={nearestDist:F1}px r={nearest?.Radius ?? 0:F0}");

        // 3. Sample BGR/HSV at perimeter radii 10-22
        for (int r = 10; r <= 22; r += 2)
        {
            int blueCount = 0, totalSamples = 0;
            var bgrSamples = new List<(int B, int G, int R)>();
            for (int angle = 0; angle < 360; angle += 5)
            {
                int px = cx + (int)(r * Math.Cos(angle * Math.PI / 180));
                int py = cy + (int)(r * Math.Sin(angle * Math.PI / 180));
                if (px < 0 || px >= page.Width || py < 0 || py >= page.Height) continue;
                totalSamples++;
                var bgr = page.At<Vec3b>(py, px);
                bgrSamples.Add((bgr[0], bgr[1], bgr[2]));
                var hsvPx = hsv.At<Vec3b>(py, px);
                // Blue check: H 85-125, S>=25, V>=50
                if (hsvPx[0] >= 85 && hsvPx[0] <= 125 && hsvPx[1] >= 25 && hsvPx[2] >= 50)
                    blueCount++;
            }
            double blueRatio = totalSamples > 0 ? (double)blueCount / totalSamples : 0;
            double avgB = bgrSamples.Average(v => v.B);
            double avgG = bgrSamples.Average(v => v.G);
            double avgR = bgrSamples.Average(v => v.R);
            Console.WriteLine($"  r={r,2}: BGR({avgB:F0},{avgG:F0},{avgR:F0}) blue={blueRatio:P1} ({blueCount}/{totalSamples})");
        }

        // 4. Interior brightness check
        using var innerMask = new Mat(gray.Size(), MatType.CV_8UC1, Scalar.Black);
        Cv2.Circle(innerMask, new Point(cx, cy), 10, Scalar.White, -1);
        double brightness = Cv2.Mean(gray, innerMask).Val0;
        Console.WriteLine($"  Interior brightness (r=10): {brightness:F1}");

        Console.WriteLine();
    }
}
