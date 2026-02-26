using OpenCvSharp;
using PDFtoImage;
using SkiaSharp;

namespace EngVision.Services;

/// <summary>
/// Renders PDF pages to OpenCV Mat images using PDFium via PDFtoImage + SkiaSharp.
/// </summary>
public class PdfRendererService : IDisposable
{
    private readonly int _dpi;

    public PdfRendererService(int dpi = 300)
    {
        _dpi = dpi;
    }

    /// <summary>
    /// Renders all pages of a PDF to a list of Mat images.
    /// </summary>
    public List<Mat> RenderAllPages(string pdfPath)
    {
        var pages = new List<Mat>();
        var pdfBytes = File.ReadAllBytes(pdfPath);
        var pageCount = Conversion.GetPageCount(pdfBytes);
        Console.WriteLine($"PDF has {pageCount} page(s), rendering at {_dpi} DPI...");

        for (int i = 0; i < pageCount; i++)
        {
            var mat = RenderPageInternal(pdfBytes, i);
            pages.Add(mat);
            Console.WriteLine($"  Page {i + 1}: {mat.Width}x{mat.Height}");
        }

        return pages;
    }

    /// <summary>
    /// Renders a single page to a Mat image.
    /// </summary>
    public Mat RenderPage(string pdfPath, int pageIndex)
    {
        var pdfBytes = File.ReadAllBytes(pdfPath);
        return RenderPageInternal(pdfBytes, pageIndex);
    }

    private Mat RenderPageInternal(byte[] pdfBytes, int pageIndex)
    {
        var options = new RenderOptions(Dpi: _dpi);
        using var bitmap = Conversion.ToImage(pdfBytes, page: pageIndex, options: options);

        // Encode as PNG to transfer to OpenCV
        using var data = bitmap.Encode(SKEncodedImageFormat.Png, 100);
        var pngBytes = data.ToArray();

        // Decode PNG bytes into OpenCV Mat
        var mat = Cv2.ImDecode(pngBytes, ImreadModes.Color);
        return mat;
    }

    /// <summary>
    /// Saves a Mat image to disk as PNG.
    /// </summary>
    public static string SaveImage(Mat image, string outputPath)
    {
        Cv2.ImWrite(outputPath, image);
        return outputPath;
    }

    public void Dispose() { }
}
