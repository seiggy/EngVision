using System.Text.RegularExpressions;
using Azure;
using Azure.AI.DocumentIntelligence;
using OpenCvSharp;

namespace EngVision.Services;

/// <summary>
/// Azure Document Intelligence table OCR — drop-in replacement for TableOcrService.
/// Uses the prebuilt-layout model to detect tables and extract balloon→dimension mappings.
/// </summary>
public class AzureTableOcrService : ITableOcrService
{
    private readonly DocumentIntelligenceClient _client;

    public AzureTableOcrService(string endpoint, string key)
    {
        _client = new DocumentIntelligenceClient(new Uri(endpoint), new AzureKeyCredential(key));
    }

    /// <summary>Extract balloon→dimension from a single page image.</summary>
    public Dictionary<int, string> ExtractBalloonDimensions(Mat pageImage)
    {
        Cv2.ImEncode(".png", pageImage, out var buf);
        return ExtractFromBytes(buf);
    }

    /// <summary>Extract balloon→dimension from all pages of a PDF in a single API call.</summary>
    public Dictionary<int, string> ExtractBalloonDimensionsFromPdf(byte[] pdfBytes)
    {
        return ExtractFromBytes(pdfBytes);
    }

    private Dictionary<int, string> ExtractFromBytes(byte[] documentBytes)
    {
        var binaryData = BinaryData.FromBytes(documentBytes);
        var operation = _client.AnalyzeDocument(WaitUntil.Completed, "prebuilt-layout", binaryData);
        var result = operation.Value;

        var dimensions = new Dictionary<int, string>();

        if (result.Tables == null || result.Tables.Count == 0)
        {
            Console.WriteLine("    Azure Doc Intelligence: no tables found");
            return dimensions;
        }

        foreach (var table in result.Tables)
        {
            var (balloonCol, dimCol) = FindColumns(table);
            if (balloonCol is null || dimCol is null) continue;

            foreach (var cell in table.Cells)
            {
                if (cell.RowIndex == 0) continue; // skip header
                if (cell.ColumnIndex != balloonCol) continue;

                var balloonNo = ParseBalloonNumber(cell.Content);
                if (balloonNo is null) continue;

                var dimVal = GetCellContent(table, cell.RowIndex, dimCol.Value);
                if (!string.IsNullOrWhiteSpace(dimVal))
                    dimensions.TryAdd(balloonNo.Value, dimVal);
            }
        }

        Console.WriteLine($"    Azure Doc Intelligence: extracted {dimensions.Count} balloon→dimension mappings");
        return dimensions;
    }

    private static (int? balloonCol, int? dimCol) FindColumns(DocumentTable table)
    {
        int? balloonCol = null, dimCol = null;

        foreach (var cell in table.Cells)
        {
            if (cell.RowIndex != 0) continue;
            var text = (cell.Content ?? "").ToUpperInvariant().Trim();
            if (text.Contains("BALLOON") || text.Contains("BALL") || text.Contains("ITEM"))
                balloonCol = cell.ColumnIndex;
            else if (text.Contains("DIM") || text.Contains("NOMINAL") || text.Contains("VALUE") || text.Contains("SIZE"))
                dimCol = cell.ColumnIndex;
        }

        return (balloonCol, dimCol);
    }

    private static string GetCellContent(DocumentTable table, int rowIndex, int colIndex)
    {
        foreach (var cell in table.Cells)
            if (cell.RowIndex == rowIndex && cell.ColumnIndex == colIndex)
                return (cell.Content ?? "").Trim();
        return "";
    }

    private static int? ParseBalloonNumber(string text)
    {
        var digits = Regex.Replace(text.Trim(), @"[^0-9]", "");
        if (int.TryParse(digits, out var num) && num is >= 1 and <= 99)
            return num;
        return null;
    }

    public void Dispose() { }
}
