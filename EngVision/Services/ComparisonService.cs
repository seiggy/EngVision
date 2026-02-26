using EngVision.Models;

namespace EngVision.Services;

/// <summary>
/// Compares measurement data extracted from page 1 bubbles against
/// table values from pages 2-4.
/// </summary>
public class ComparisonService
{
    /// <summary>
    /// Compares bubble measurements against table measurements, matching by bubble number.
    /// </summary>
    public List<ComparisonResult> Compare(
        List<MeasurementData> bubbleMeasurements,
        List<MeasurementData> tableMeasurements)
    {
        var results = new List<ComparisonResult>();

        // Group table measurements by bubble number
        var tableByBubble = tableMeasurements
            .Where(m => m.BubbleNumber > 0)
            .GroupBy(m => m.BubbleNumber)
            .ToDictionary(g => g.Key, g => g.First());

        // Get all bubble numbers from both sources
        var allBubbleNumbers = bubbleMeasurements
            .Select(m => m.BubbleNumber)
            .Union(tableMeasurements.Select(m => m.BubbleNumber))
            .Where(n => n > 0)
            .Distinct()
            .OrderBy(n => n);

        foreach (var bubbleNum in allBubbleNumbers)
        {
            var bubbleMeasurement = bubbleMeasurements.FirstOrDefault(m => m.BubbleNumber == bubbleNum);
            var tableMeasurement = tableByBubble.GetValueOrDefault(bubbleNum);

            var (match, discrepancy) = CompareMeasurements(bubbleMeasurement, tableMeasurement);

            results.Add(new ComparisonResult
            {
                BubbleNumber = bubbleNum,
                BubbleMeasurement = bubbleMeasurement,
                TableMeasurement = tableMeasurement,
                Match = match,
                Discrepancy = discrepancy
            });
        }

        return results;
    }

    /// <summary>
    /// Prints a comparison report to the console.
    /// </summary>
    public void PrintReport(List<ComparisonResult> results)
    {
        Console.WriteLine();
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║              MEASUREMENT COMPARISON REPORT                  ║");
        Console.WriteLine("╠══════════════════════════════════════════════════════════════╣");

        foreach (var result in results.OrderBy(r => r.BubbleNumber))
        {
            var status = result.Match ? "✓ MATCH" : "✗ MISMATCH";
            Console.WriteLine($"║ Bubble #{result.BubbleNumber,-4} {status,-12}                              ║");

            if (result.BubbleMeasurement is not null)
            {
                Console.WriteLine($"║   Drawing: {FormatMeasurement(result.BubbleMeasurement),-49}║");
            }
            else
            {
                Console.WriteLine($"║   Drawing: NOT FOUND                                        ║");
            }

            if (result.TableMeasurement is not null)
            {
                Console.WriteLine($"║   Table:   {FormatMeasurement(result.TableMeasurement),-49}║");
            }
            else
            {
                Console.WriteLine($"║   Table:   NOT FOUND                                        ║");
            }

            if (result.Discrepancy is not null)
            {
                Console.WriteLine($"║   Note:    {result.Discrepancy,-49}║");
            }

            Console.WriteLine("╠══════════════════════════════════════════════════════════════╣");
        }

        var matchCount = results.Count(r => r.Match);
        var total = results.Count;
        Console.WriteLine($"║ Summary: {matchCount}/{total} measurements match                          ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
    }

    private static (bool Match, string? Discrepancy) CompareMeasurements(
        MeasurementData? bubble, MeasurementData? table)
    {
        if (bubble is null && table is null)
            return (true, null);

        if (bubble is null)
            return (false, "Bubble measurement not found on drawing");

        if (table is null)
            return (false, "Table entry not found");

        // Compare nominal values
        if (!string.IsNullOrEmpty(bubble.NominalValue) && !string.IsNullOrEmpty(table.NominalValue))
        {
            var bubbleVal = NormalizeValue(bubble.NominalValue);
            var tableVal = NormalizeValue(table.NominalValue);

            if (bubbleVal != tableVal)
            {
                return (false, $"Nominal value mismatch: drawing={bubble.NominalValue}, table={table.NominalValue}");
            }
        }

        // Compare tolerances if both have them
        if (!string.IsNullOrEmpty(bubble.UpperTolerance) && !string.IsNullOrEmpty(table.UpperTolerance))
        {
            if (NormalizeValue(bubble.UpperTolerance) != NormalizeValue(table.UpperTolerance))
            {
                return (false, $"Upper tolerance mismatch: drawing={bubble.UpperTolerance}, table={table.UpperTolerance}");
            }
        }

        if (!string.IsNullOrEmpty(bubble.LowerTolerance) && !string.IsNullOrEmpty(table.LowerTolerance))
        {
            if (NormalizeValue(bubble.LowerTolerance) != NormalizeValue(table.LowerTolerance))
            {
                return (false, $"Lower tolerance mismatch: drawing={bubble.LowerTolerance}, table={table.LowerTolerance}");
            }
        }

        return (true, null);
    }

    private static string NormalizeValue(string value)
    {
        // Remove whitespace, convert to lowercase for comparison
        var normalized = value.Trim().ToLowerInvariant();

        // Try to parse as a number and normalize decimal representation
        if (double.TryParse(normalized, out var numericVal))
        {
            return numericVal.ToString("G");
        }

        return normalized;
    }

    private static string FormatMeasurement(MeasurementData m)
    {
        var parts = new List<string>();
        if (!string.IsNullOrEmpty(m.NominalValue))
            parts.Add($"{m.NominalValue}{(string.IsNullOrEmpty(m.Unit) ? "" : " " + m.Unit)}");
        if (!string.IsNullOrEmpty(m.UpperTolerance))
            parts.Add($"+{m.UpperTolerance}");
        if (!string.IsNullOrEmpty(m.LowerTolerance))
            parts.Add($"-{m.LowerTolerance}");
        if (!string.IsNullOrEmpty(m.DimensionName))
            parts.Add($"({m.DimensionName})");

        return parts.Count > 0 ? string.Join(" ", parts) : "(no data)";
    }
}
