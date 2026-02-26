using System.Text.Json;
using EngVision.Models;
using EngVision.Services;
using OpenCvSharp;

// Analyze pixel characteristics at annotated bubble centers
var sampleDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "sample_docs"));
var pdfPath = Directory.GetFiles(sampleDir, "*.pdf").First();
var annotPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "Output", "new_annotations.json"));

var jsonOpts = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

// Load annotations
var annJson = File.ReadAllText(annotPath);
var annotations = JsonSerializer.Deserialize<List<GroundTruthAnnotation>>(annJson, jsonOpts)!;
Console.WriteLine($"Loaded {annotations.Count} annotations with bubble centers");

// Render page 1
var renderer = new PdfRendererService(300);
using var page = renderer.RenderPage(pdfPath, 0);
using var gray = new Mat();
Cv2.CvtColor(page, gray, ColorConversionCodes.BGR2GRAY);

Console.WriteLine($"Page size: {page.Width}x{page.Height}");
Console.WriteLine();
Console.WriteLine("=== PIXEL ANALYSIS AT BUBBLE CENTERS ===");
Console.WriteLine($"{"#",-4} {"Center",-14} {"Pixel",-6} {"R10mean",-8} {"R15mean",-8} {"R20mean",-8} {"Ring10",-8} {"Ring15",-8} {"Ring20",-8} {"Contrast",-9} {"DarkRatio",-10}");

foreach (var ann in annotations)
{
    if (ann.BubbleCenter is null) continue;
    int cx = ann.BubbleCenter.X, cy = ann.BubbleCenter.Y;
    
    // Center pixel value
    byte centerPx = gray.At<byte>(cy, cx);
    
    // Mean brightness in circles of various radii
    double[] innerMeans = new double[3];
    double[] ringMeans = new double[3];
    int[] radii = [10, 15, 20];
    
    for (int ri = 0; ri < 3; ri++)
    {
        int r = radii[ri];
        // Inner mean
        using var mask = new Mat(gray.Size(), MatType.CV_8UC1, Scalar.Black);
        Cv2.Circle(mask, new Point(cx, cy), r, Scalar.White, -1);
        innerMeans[ri] = Cv2.Mean(gray, mask).Val0;
        
        // Ring mean (annular region)
        using var ringMask = new Mat(gray.Size(), MatType.CV_8UC1, Scalar.Black);
        Cv2.Circle(ringMask, new Point(cx, cy), r + 2, Scalar.White, 3);
        Cv2.Circle(ringMask, new Point(cx, cy), Math.Max(1, r - 1), Scalar.Black, -1);
        ringMeans[ri] = Cv2.Mean(gray, ringMask).Val0;
    }
    
    // Dark pixel ratio within r=15
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
    
    Console.WriteLine($"{ann.BubbleNumber ?? 0,-4} ({cx,4},{cy,4})  {centerPx,-6} {innerMeans[0],-8:F1} {innerMeans[1],-8:F1} {innerMeans[2],-8:F1} {ringMeans[0],-8:F1} {ringMeans[1],-8:F1} {ringMeans[2],-8:F1} {contrast,-9:F1} {darkRatio,-10:F3}");
}

// Now run HoughCircles and check which bubble centers have a detection nearby
Console.WriteLine();
Console.WriteLine("=== HOUGHCIRCLES COVERAGE CHECK ===");

using var blur = new Mat();
Cv2.GaussianBlur(gray, blur, new Size(9, 9), 2);

var circles = Cv2.HoughCircles(blur, HoughModes.Gradient,
    dp: 1.2, minDist: 22, param1: 120, param2: 23, minRadius: 8, maxRadius: 50);
Console.WriteLine($"HoughCircles found {circles.Length} candidates (before verification)");

foreach (var ann in annotations)
{
    if (ann.BubbleCenter is null) continue;
    int cx = ann.BubbleCenter.X, cy = ann.BubbleCenter.Y;
    
    double minDist = double.MaxValue;
    CircleSegment? nearest = null;
    foreach (var c in circles)
    {
        double d = Math.Sqrt(Math.Pow(cx - c.Center.X, 2) + Math.Pow(cy - c.Center.Y, 2));
        if (d < minDist) { minDist = d; nearest = c; }
    }
    
    string status = minDist < 30 ? "✓ FOUND" : minDist < 60 ? "~ NEAR" : "✗ MISS";
    Console.WriteLine($"  Bubble #{ann.BubbleNumber ?? 0}: nearest HoughCircle at dist={minDist:F0}px (r={nearest?.Radius:F0}) {status}");
}
