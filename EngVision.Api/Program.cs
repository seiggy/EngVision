using System.Text.Json;
using System.Collections.Concurrent;
using EngVision.Models;
using EngVision.Services;

var builder = WebApplication.CreateBuilder(args);

builder.AddServiceDefaults();

// Load .env file from repo root
var envFile = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".env"));
if (File.Exists(envFile))
{
    foreach (var line in File.ReadAllLines(envFile))
    {
        var trimmed = line.Trim();
        if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith('#')) continue;
        var eqIdx = trimmed.IndexOf('=');
        if (eqIdx <= 0) continue;
        Environment.SetEnvironmentVariable(trimmed[..eqIdx].Trim(), trimmed[(eqIdx + 1)..].Trim());
    }
}

builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
        policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
});

builder.Services.AddSingleton<EngVisionConfig>(_ => new EngVisionConfig
{
    PdfRenderDpi = 300,
    OutputDirectory = Path.Combine(AppContext.BaseDirectory, "Output")
});
builder.Services.AddSingleton<PdfRendererService>(sp =>
    new PdfRendererService(sp.GetRequiredService<EngVisionConfig>().PdfRenderDpi));
builder.Services.AddSingleton<BubbleDetectionService>(sp =>
    new BubbleDetectionService(sp.GetRequiredService<EngVisionConfig>()));
builder.Services.AddSingleton<LeaderLineTracerService>();
builder.Services.AddSingleton<TableDetectionService>(sp =>
    new TableDetectionService(sp.GetRequiredService<EngVisionConfig>()));
builder.Services.AddSingleton<AnnotationStore>();

// Pipeline service + run store
var tessDataPath = Path.Combine(AppContext.BaseDirectory, "tessdata");
var uploadsDir = Path.Combine(AppContext.BaseDirectory, "uploads");
var pipelineOutputDir = Path.Combine(AppContext.BaseDirectory, "pipeline_runs");
Directory.CreateDirectory(uploadsDir);
Directory.CreateDirectory(pipelineOutputDir);

builder.Services.AddSingleton<PipelineService>(sp =>
    new PipelineService(sp.GetRequiredService<EngVisionConfig>(), tessDataPath));

// In-memory store for pipeline run results
var pipelineRuns = new ConcurrentDictionary<string, PipelineResult>();
var pipelineProgress = new ConcurrentDictionary<string, string>();

var app = builder.Build();

app.MapDefaultEndpoints();
app.UseCors();

var jsonOpts = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true };
var sampleDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "sample_docs"));

// ── PDF listing ────────────────────────────────────────────────────────────────
app.MapGet("/api/pdfs", () =>
{
    if (!Directory.Exists(sampleDir))
        return Results.Ok(Array.Empty<string>());
    var files = Directory.GetFiles(sampleDir, "*.pdf")
        .Select(Path.GetFileName)
        .ToArray();
    return Results.Ok(files);
});

// ── Render page as image ───────────────────────────────────────────────────────
app.MapGet("/api/pdfs/{filename}/pages/{pageNum}/image", (
    string filename, int pageNum,
    PdfRendererService renderer, EngVisionConfig config) =>
{
    var pdfPath = Path.Combine(sampleDir, filename);
    if (!File.Exists(pdfPath)) return Results.NotFound("PDF not found");

    var outputDir = Path.Combine(config.OutputDirectory, SanitizeName(filename));
    Directory.CreateDirectory(outputDir);
    var imgPath = Path.Combine(outputDir, $"page_{pageNum}.png");

    if (!File.Exists(imgPath))
    {
        using var mat = renderer.RenderPage(pdfPath, pageNum - 1);
        PdfRendererService.SaveImage(mat, imgPath);
    }

    return Results.File(imgPath, "image/png");
});

// ── Page count ─────────────────────────────────────────────────────────────────
app.MapGet("/api/pdfs/{filename}/info", (string filename, PdfRendererService renderer) =>
{
    var pdfPath = Path.Combine(sampleDir, filename);
    if (!File.Exists(pdfPath)) return Results.NotFound("PDF not found");

    var pdfBytes = File.ReadAllBytes(pdfPath);
    var pageCount = PDFtoImage.Conversion.GetPageCount(pdfBytes);

    // Get page dimensions from first page render
    using var mat = renderer.RenderPage(pdfPath, 0);
    return Results.Ok(new { pageCount, width = mat.Width, height = mat.Height });
});

// ── Auto-detect bubbles ────────────────────────────────────────────────────────
app.MapPost("/api/pdfs/{filename}/pages/{pageNum}/detect", (
    string filename, int pageNum,
    BubbleDetectionService bubbleDetector,
    PdfRendererService renderer,
    AnnotationStore store) =>
{
    var pdfPath = Path.Combine(sampleDir, filename);
    if (!File.Exists(pdfPath)) return Results.NotFound("PDF not found");

    using var mat = renderer.RenderPage(pdfPath, pageNum - 1);

    List<DetectedRegion> regions;
    if (pageNum == 1)
    {
        // Return raw bubble detections (tight circle bounding boxes)
        regions = bubbleDetector.DetectBubbles(mat, pageNum);
    }
    else
    {
        var tableDetector = app.Services.GetRequiredService<TableDetectionService>();
        regions = tableDetector.DetectTables(mat, pageNum);
        if (regions.Count == 0)
            regions = [tableDetector.GetFullPageRegion(mat, pageNum)];
    }

    // Store as auto-detected
    var docKey = SanitizeName(filename);
    store.SetAutoDetections(docKey, pageNum, regions);

    return Results.Ok(regions);
});

// ── Get annotations (manual + auto) ───────────────────────────────────────────
app.MapGet("/api/pdfs/{filename}/pages/{pageNum}/annotations", (
    string filename, int pageNum, AnnotationStore store) =>
{
    var docKey = SanitizeName(filename);
    var manual = store.GetManualAnnotations(docKey, pageNum);
    var auto = store.GetAutoDetections(docKey, pageNum);
    return Results.Ok(new { manual, auto });
});

// ── Save manual annotation ────────────────────────────────────────────────────
app.MapPost("/api/pdfs/{filename}/pages/{pageNum}/annotations", (
    string filename, int pageNum, Annotation annotation, AnnotationStore store) =>
{
    var docKey = SanitizeName(filename);
    store.AddManualAnnotation(docKey, pageNum, annotation);
    return Results.Ok(annotation);
});

// ── Update manual annotation ──────────────────────────────────────────────────
app.MapPut("/api/pdfs/{filename}/pages/{pageNum}/annotations/{id}", (
    string filename, int pageNum, string id, Annotation annotation, AnnotationStore store) =>
{
    var docKey = SanitizeName(filename);
    store.UpdateManualAnnotation(docKey, pageNum, id, annotation);
    return Results.Ok(annotation);
});

// ── Clear all manual annotations for a page ───────────────────────────────────
app.MapDelete("/api/pdfs/{filename}/pages/{pageNum}/annotations", (
    string filename, int pageNum, AnnotationStore store) =>
{
    var docKey = SanitizeName(filename);
    store.ClearManualAnnotations(docKey, pageNum);
    return Results.NoContent();
});

// ── Delete manual annotation ──────────────────────────────────────────────────
app.MapDelete("/api/pdfs/{filename}/pages/{pageNum}/annotations/{id}", (
    string filename, int pageNum, string id, AnnotationStore store) =>
{
    var docKey = SanitizeName(filename);
    store.DeleteManualAnnotation(docKey, pageNum, id);
    return Results.NoContent();
});

// ── Export all annotations as ground truth JSON ────────────────────────────────
app.MapGet("/api/pdfs/{filename}/annotations/export", (
    string filename, AnnotationStore store, EngVisionConfig config) =>
{
    var docKey = SanitizeName(filename);
    var allAnnotations = store.ExportAll(docKey);
    var outputDir = Path.Combine(config.OutputDirectory, docKey);
    Directory.CreateDirectory(outputDir);
    var path = Path.Combine(outputDir, "ground_truth.json");
    File.WriteAllText(path, JsonSerializer.Serialize(allAnnotations, jsonOpts));
    return Results.Ok(allAnnotations);
});

// ── Compute detection accuracy vs ground truth ────────────────────────────────
app.MapGet("/api/pdfs/{filename}/pages/{pageNum}/accuracy", (
    string filename, int pageNum, AnnotationStore store) =>
{
    var docKey = SanitizeName(filename);
    var manual = store.GetManualAnnotations(docKey, pageNum);
    var auto = store.GetAutoDetections(docKey, pageNum);

    if (manual.Count == 0)
        return Results.Ok(new { message = "No manual annotations to compare against" });

    // Compute IoU-based matching
    int matched = 0, missed = 0, falsePositives = 0;
    var usedAuto = new HashSet<int>();

    foreach (var gt in manual)
    {
        double bestIoU = 0;
        int bestIdx = -1;
        for (int i = 0; i < auto.Count; i++)
        {
            if (usedAuto.Contains(i)) continue;
            var iou = ComputeIoU(gt.BoundingBox, auto[i].BoundingBox);
            if (iou > bestIoU) { bestIoU = iou; bestIdx = i; }
        }
        if (bestIoU >= 0.3 && bestIdx >= 0)
        {
            matched++;
            usedAuto.Add(bestIdx);
        }
        else missed++;
    }
    falsePositives = auto.Count - usedAuto.Count;

    double precision = auto.Count > 0 ? (double)matched / auto.Count : 0;
    double recall = manual.Count > 0 ? (double)matched / manual.Count : 0;
    double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

    return Results.Ok(new
    {
        groundTruth = manual.Count,
        detected = auto.Count,
        matched,
        missed,
        falsePositives,
        precision = Math.Round(precision, 3),
        recall = Math.Round(recall, 3),
        f1 = Math.Round(f1, 3)
    });
});

// ── Pipeline: Upload and run ────────────────────────────────────────────────────
app.MapPost("/api/pipeline/run", async (HttpRequest request, PipelineService pipeline) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest("Expected multipart/form-data with a PDF file");

    var form = await request.ReadFormAsync();
    var file = form.Files.GetFile("pdf");
    if (file is null || file.Length == 0)
        return Results.BadRequest("No PDF file provided");

    // Save uploaded file
    var runId = Guid.NewGuid().ToString("N")[..8];
    var pdfPath = Path.Combine(uploadsDir, $"{runId}_{file.FileName}");
    await using (var stream = File.Create(pdfPath))
        await file.CopyToAsync(stream);

    var runOutputDir = Path.Combine(pipelineOutputDir, runId);

    // Run pipeline
    pipelineProgress[runId] = "Starting...";
    var result = await pipeline.RunAsync(pdfPath, runId, runOutputDir,
        msg => pipelineProgress[runId] = msg);

    pipelineRuns[runId] = result;
    pipelineProgress.TryRemove(runId, out _);

    return Results.Ok(result);
}).DisableAntiforgery();

// ── Pipeline: Run on existing sample PDF ───────────────────────────────────────
app.MapPost("/api/pipeline/run-sample/{filename}", async (string filename, PipelineService pipeline) =>
{
    var pdfPath = Path.Combine(sampleDir, filename);
    if (!File.Exists(pdfPath)) return Results.NotFound("PDF not found");

    var runId = Guid.NewGuid().ToString("N")[..8];
    var runOutputDir = Path.Combine(pipelineOutputDir, runId);

    pipelineProgress[runId] = "Starting...";
    var result = await pipeline.RunAsync(pdfPath, runId, runOutputDir,
        msg => pipelineProgress[runId] = msg);

    pipelineRuns[runId] = result;
    pipelineProgress.TryRemove(runId, out _);

    return Results.Ok(result);
});

// ── Pipeline: Get run result ───────────────────────────────────────────────────
app.MapGet("/api/pipeline/{runId}/results", (string runId) =>
{
    if (pipelineRuns.TryGetValue(runId, out var result))
        return Results.Ok(result);
    if (pipelineProgress.TryGetValue(runId, out var progress))
        return Results.Ok(new { status = "running", progress });
    return Results.NotFound();
});

// ── Pipeline: Get page image ───────────────────────────────────────────────────
app.MapGet("/api/pipeline/{runId}/pages/{pageNum}/image", (string runId, int pageNum) =>
{
    var imgPath = Path.Combine(pipelineOutputDir, runId, "pages", $"page_{pageNum}.png");
    return File.Exists(imgPath) ? Results.File(imgPath, "image/png") : Results.NotFound();
});

// ── Pipeline: Get overlay image ────────────────────────────────────────────────
app.MapGet("/api/pipeline/{runId}/pages/{pageNum}/overlay", (string runId, int pageNum) =>
{
    var imgPath = Path.Combine(pipelineOutputDir, runId, "overlays", $"page_{pageNum}_overlay.png");
    return File.Exists(imgPath) ? Results.File(imgPath, "image/png") : Results.NotFound();
});

// ── Pipeline: Get capture crop for a bubble ───────────────────────────────────
app.MapGet("/api/pipeline/{runId}/bubbles/{bubbleNo}/capture", (string runId, int bubbleNo, string? size) =>
{
    var debugDir = Path.Combine(pipelineOutputDir, runId, "debug");
    if (!string.IsNullOrEmpty(size))
    {
        var imgPath = Path.Combine(debugDir, $"capture_bubble_{bubbleNo:D3}_{size}.png");
        return File.Exists(imgPath) ? Results.File(imgPath, "image/png") : Results.NotFound();
    }
    // Serve the largest available capture
    var sizes = new[] { (1024, 512), (512, 256), (256, 128), (128, 128) };
    foreach (var (w, h) in sizes)
    {
        var candidate = Path.Combine(debugDir, $"capture_bubble_{bubbleNo:D3}_{w}x{h}.png");
        if (File.Exists(candidate))
            return Results.File(candidate, "image/png");
    }
    return Results.NotFound();
});

// ── Pipeline: List all capture sizes for a bubble ─────────────────────────────
app.MapGet("/api/pipeline/{runId}/bubbles/{bubbleNo}/captures", (string runId, int bubbleNo) =>
{
    var debugDir = Path.Combine(pipelineOutputDir, runId, "debug");
    var sizes = new (int W, int H)[] { (128, 128), (256, 128), (512, 256), (1024, 512) };
    var available = sizes
        .Where(s => File.Exists(Path.Combine(debugDir, $"capture_bubble_{bubbleNo:D3}_{s.W}x{s.H}.png")))
        .Select(s => new { size = $"{s.W}x{s.H}", width = s.W, height = s.H });
    return Results.Ok(available);
});

// ── Pipeline: List all runs ────────────────────────────────────────────────────
app.MapGet("/api/pipeline/runs", () =>
{
    var runs = pipelineRuns.Values
        .OrderByDescending(r => r.RunId)
        .Select(r => new
        {
            r.RunId,
            r.PdfFilename,
            r.TotalBubbles,
            r.MatchedBubbles,
            r.UnmatchedBubbles,
            r.Status
        });
    return Results.Ok(runs);
});

// ── Pipeline: SSE streaming run (uploaded PDF) ─────────────────────────────────
app.MapPost("/api/pipeline/run-stream", async (HttpRequest request, PipelineService pipeline, HttpContext context) =>
{
    if (!request.HasFormContentType)
    {
        context.Response.StatusCode = 400;
        await context.Response.WriteAsync("Expected multipart/form-data with a PDF file");
        return;
    }

    var form = await request.ReadFormAsync();
    var file = form.Files.GetFile("pdf");
    if (file is null || file.Length == 0)
    {
        context.Response.StatusCode = 400;
        await context.Response.WriteAsync("No PDF file provided");
        return;
    }

    var runId = Guid.NewGuid().ToString("N")[..8];
    var pdfPath = Path.Combine(uploadsDir, $"{runId}_{file.FileName}");
    await using (var stream = File.Create(pdfPath))
        await file.CopyToAsync(stream);

    var runOutputDir = Path.Combine(pipelineOutputDir, runId);

    context.Response.ContentType = "text/event-stream";
    context.Response.Headers["Cache-Control"] = "no-cache";
    context.Response.Headers["Connection"] = "keep-alive";

    var sseJsonOpts = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    await foreach (var evt in pipeline.RunStreamAsync(pdfPath, runId, runOutputDir))
    {
        var eventType = evt["type"]?.ToString() ?? "message";
        var json = JsonSerializer.Serialize(evt, sseJsonOpts);
        await context.Response.WriteAsync($"event: {eventType}\ndata: {json}\n\n");
        await context.Response.Body.FlushAsync();

        // Store result on completion
        if (eventType == "complete" && evt.TryGetValue("result", out var resultObj) && resultObj is PipelineResult result)
            pipelineRuns[runId] = result;
    }
}).DisableAntiforgery();

// ── Pipeline: SSE streaming run (sample PDF) ───────────────────────────────────
app.MapPost("/api/pipeline/run-stream-sample/{filename}", async (string filename, PipelineService pipeline, HttpContext context) =>
{
    var pdfPath = Path.Combine(sampleDir, filename);
    if (!File.Exists(pdfPath))
    {
        context.Response.StatusCode = 404;
        await context.Response.WriteAsync("PDF not found");
        return;
    }

    var runId = Guid.NewGuid().ToString("N")[..8];
    var runOutputDir = Path.Combine(pipelineOutputDir, runId);

    context.Response.ContentType = "text/event-stream";
    context.Response.Headers["Cache-Control"] = "no-cache";
    context.Response.Headers["Connection"] = "keep-alive";

    var sseJsonOpts = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    await foreach (var evt in pipeline.RunStreamAsync(pdfPath, runId, runOutputDir))
    {
        var eventType = evt["type"]?.ToString() ?? "message";
        var json = JsonSerializer.Serialize(evt, sseJsonOpts);
        await context.Response.WriteAsync($"event: {eventType}\ndata: {json}\n\n");
        await context.Response.Body.FlushAsync();

        if (eventType == "complete" && evt.TryGetValue("result", out var resultObj) && resultObj is PipelineResult result)
            pipelineRuns[runId] = result;
    }
});

app.Run();

// ── Pipeline API endpoints ─────────────────────────────────────────────────────
// These are mapped above via app.Map* calls before app.Run()

static string SanitizeName(string name) => Path.GetFileNameWithoutExtension(name)
    .Replace(" ", "_").Replace(".", "_");

static double ComputeIoU(BoundingBox a, BoundingBox b)
{
    int x1 = Math.Max(a.X, b.X);
    int y1 = Math.Max(a.Y, b.Y);
    int x2 = Math.Min(a.X + a.Width, b.X + b.Width);
    int y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);
    if (x2 <= x1 || y2 <= y1) return 0;
    double intersection = (x2 - x1) * (y2 - y1);
    double union = (double)a.Width * a.Height + (double)b.Width * b.Height - intersection;
    return union > 0 ? intersection / union : 0;
}

// ── In-memory annotation store ─────────────────────────────────────────────────
public record AnnotationPoint
{
    public int X { get; init; }
    public int Y { get; init; }
}

public record Annotation
{
    public string Id { get; init; } = Guid.NewGuid().ToString("N")[..8];
    public int? BubbleNumber { get; init; }
    public AnnotationPoint? BubbleCenter { get; init; }
    public required BoundingBox BoundingBox { get; init; }
    public string? Label { get; init; }
    public string? Notes { get; init; }
}

public class AnnotationStore
{
    // Key: "{docKey}:{pageNum}"
    private readonly Dictionary<string, List<Annotation>> _manual = new();
    private readonly Dictionary<string, List<DetectedRegion>> _auto = new();
    private readonly string _persistDir;

    public AnnotationStore(EngVisionConfig config)
    {
        _persistDir = Path.Combine(config.OutputDirectory, "annotations");
        Directory.CreateDirectory(_persistDir);
        LoadFromDisk();
    }

    public List<Annotation> GetManualAnnotations(string docKey, int pageNum)
        => _manual.TryGetValue($"{docKey}:{pageNum}", out var list) ? list : [];

    public List<DetectedRegion> GetAutoDetections(string docKey, int pageNum)
        => _auto.TryGetValue($"{docKey}:{pageNum}", out var list) ? list : [];

    public void SetAutoDetections(string docKey, int pageNum, List<DetectedRegion> regions)
        => _auto[$"{docKey}:{pageNum}"] = regions;

    public void AddManualAnnotation(string docKey, int pageNum, Annotation annotation)
    {
        var key = $"{docKey}:{pageNum}";
        if (!_manual.ContainsKey(key)) _manual[key] = [];
        _manual[key].Add(annotation);
        SaveToDisk();
    }

    public void UpdateManualAnnotation(string docKey, int pageNum, string id, Annotation annotation)
    {
        var key = $"{docKey}:{pageNum}";
        if (!_manual.ContainsKey(key)) return;
        var idx = _manual[key].FindIndex(a => a.Id == id);
        if (idx >= 0) _manual[key][idx] = annotation;
        SaveToDisk();
    }

    public void ClearManualAnnotations(string docKey, int pageNum)
    {
        var key = $"{docKey}:{pageNum}";
        _manual.Remove(key);
        SaveToDisk();
    }

    public void DeleteManualAnnotation(string docKey, int pageNum, string id)
    {
        var key = $"{docKey}:{pageNum}";
        if (!_manual.ContainsKey(key)) return;
        _manual[key].RemoveAll(a => a.Id == id);
        SaveToDisk();
    }

    public Dictionary<string, List<Annotation>> ExportAll(string docKey)
    {
        return _manual
            .Where(kv => kv.Key.StartsWith(docKey + ":"))
            .ToDictionary(kv => kv.Key, kv => kv.Value);
    }

    private void SaveToDisk()
    {
        var path = Path.Combine(_persistDir, "manual_annotations.json");
        var json = JsonSerializer.Serialize(_manual, new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
        File.WriteAllText(path, json);
    }

    private void LoadFromDisk()
    {
        var path = Path.Combine(_persistDir, "manual_annotations.json");
        if (!File.Exists(path)) return;
        try
        {
            var json = File.ReadAllText(path);
            var data = JsonSerializer.Deserialize<Dictionary<string, List<Annotation>>>(json,
                new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
            if (data is not null)
                foreach (var (k, v) in data) _manual[k] = v;
        }
        catch { /* ignore corrupt file */ }
    }
}
