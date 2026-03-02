using System.Text.RegularExpressions;
using Azure;
using Azure.AI.DocumentIntelligence;
using OpenCvSharp;

namespace EngVision.Services;

/// <summary>
/// Azure Document Intelligence table OCR — drop-in replacement for TableOcrService.
/// Uses the prebuilt-layout model to detect tables and extract balloon→dimension mappings.
/// Builds a full cell matrix (handling row_span/column_span) for accurate extraction.
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

        Console.WriteLine($"    Azure Doc Intelligence: found {result.Tables.Count} table(s)");

        foreach (var table in result.Tables)
        {
            // Build a full matrix handling row_span and column_span
            var matrix = BuildMatrix(table);

            // Debug: dump first few rows
            var dumpRows = Math.Min(5, table.RowCount);
            for (int r = 0; r < dumpRows; r++)
            {
                var rowCells = string.Join(" | ", Enumerable.Range(0, table.ColumnCount)
                    .Select(c => matrix[r, c].Length > 30 ? matrix[r, c][..30] + "…" : matrix[r, c]));
                Console.WriteLine($"      Row {r}: {rowCells}");
            }

            // Find balloon and dimension columns from the matrix
            var (balloonCol, dimCol, headerRow) = FindColumnsFromMatrix(matrix, table.RowCount, table.ColumnCount);
            Console.WriteLine($"      → balloonCol={balloonCol}, dimCol={dimCol}, headerRow={headerRow}");

            if (balloonCol is null || dimCol is null) continue;

            // Extract balloon→dimension pairs from rows below the header
            for (int r = headerRow + 1; r < table.RowCount; r++)
            {
                var balloonText = matrix[r, balloonCol.Value];
                var balloonNo = ParseBalloonNumber(balloonText);
                if (balloonNo is null) continue;

                var dimVal = matrix[r, dimCol.Value].Trim();
                if (!string.IsNullOrWhiteSpace(dimVal))
                {
                    dimensions.TryAdd(balloonNo.Value, dimVal);
                    Console.WriteLine($"      Balloon {balloonNo} → {dimVal}");
                }
            }
        }

        Console.WriteLine($"    Azure Doc Intelligence: extracted {dimensions.Count} balloon→dimension mappings");
        return dimensions;
    }

    /// <summary>
    /// Build a 2D string matrix from table cells, properly handling row_span and column_span.
    /// This ensures merged cells fill all their spanned positions.
    /// </summary>
    private static string[,] BuildMatrix(DocumentTable table)
    {
        var matrix = new string[table.RowCount, table.ColumnCount];
        for (int r = 0; r < table.RowCount; r++)
            for (int c = 0; c < table.ColumnCount; c++)
                matrix[r, c] = "";

        foreach (var cell in table.Cells)
        {
            var text = (cell.Content ?? "").Trim();
            var rowSpan = cell.RowSpan ?? 1;
            var colSpan = cell.ColumnSpan ?? 1;

            for (int rr = cell.RowIndex; rr < cell.RowIndex + rowSpan && rr < table.RowCount; rr++)
                for (int cc = cell.ColumnIndex; cc < cell.ColumnIndex + colSpan && cc < table.ColumnCount; cc++)
                    matrix[rr, cc] = text;
        }

        return matrix;
    }

    /// <summary>
    /// Find balloon and dimension columns by scanning the matrix rows for header keywords.
    /// </summary>
    private static (int? balloonCol, int? dimCol, int headerRow) FindColumnsFromMatrix(
        string[,] matrix, int rows, int cols)
    {
        var scanRows = Math.Min(5, rows);
        for (int r = 0; r < scanRows; r++)
        {
            int? balloonCol = null, dimCol = null;
            for (int c = 0; c < cols; c++)
            {
                var text = matrix[r, c].ToUpperInvariant();
                if (balloonCol is null && (text.Contains("BALLOON") || text.Contains("BALL") || text.Contains("ITEM")))
                    balloonCol = c;
                else if (dimCol is null && (text.Contains("DIM") || text.Contains("NOMINAL") || text.Contains("VALUE") || text.Contains("SIZE")))
                    dimCol = c;
            }
            if (balloonCol is not null && dimCol is not null)
                return (balloonCol, dimCol, r);
        }
        return (null, null, 0);
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
