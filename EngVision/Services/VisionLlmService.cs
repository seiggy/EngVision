using System.Text.Json;
using EngVision.Models;
using OpenAI.Responses;

namespace EngVision.Services;

/// <summary>
/// Uses the OpenAI Responses API to extract structured measurement data
/// from cropped image segments via vision capabilities.
/// </summary>
public class VisionLlmService
{
    private readonly ResponsesClient _client;
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };

    public VisionLlmService(ResponsesClient client)
    {
        _client = client;
    }

    /// <summary>
    /// Extracts measurement data from a bubble/figure segment image.
    /// </summary>
    public async Task<MeasurementData?> ExtractBubbleMeasurement(DetectedRegion region)
    {
        if (region.CroppedImagePath is null || !File.Exists(region.CroppedImagePath))
        {
            Console.WriteLine($"  Warning: No cropped image for region {region.Id}");
            return null;
        }

        var imageBytes = await File.ReadAllBytesAsync(region.CroppedImagePath);

        var options = new CreateResponseOptions
        {
            Instructions = """
                You are an expert at reading engineering CAD drawings and dimensional analysis documents.
                Extract measurement data from the provided image segment showing a numbered bubble annotation
                on a CAD drawing. The bubble contains a number that references a specific measurement/dimension.

                Return a JSON object with these fields:
                {
                    "bubbleNumber": <integer - the number inside the bubble>,
                    "dimensionName": "<name or description of the dimension being measured>",
                    "nominalValue": "<the nominal/target value shown>",
                    "unit": "<unit of measurement (mm, in, deg, etc.)>",
                    "upperTolerance": "<upper tolerance if shown>",
                    "lowerTolerance": "<lower tolerance if shown>",
                    "rawText": "<all text visible near this bubble>"
                }

                If a field cannot be determined, use null.
                Return ONLY the JSON object, no other text.
                """
        };

        var contentParts = new List<ResponseContentPart>
        {
            ResponseContentPart.CreateInputTextPart("Extract the measurement data from this CAD drawing segment:"),
            ResponseContentPart.CreateInputImagePart(BinaryData.FromBytes(imageBytes), "image/png")
        };
        options.InputItems.Add(ResponseItem.CreateUserMessageItem(contentParts));

        try
        {
            var response = await _client.CreateResponseAsync(options);
            var content = GetOutputText(response.Value);

            if (string.IsNullOrEmpty(content)) return null;

            content = StripCodeFences(content);

            var data = JsonSerializer.Deserialize<MeasurementData>(content, _jsonOptions);
            return data is null ? null : data with
            {
                SourcePage = region.PageNumber,
                SourceType = region.Type
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Error extracting bubble {region.Id}: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Extracts measurement data from a table region image.
    /// Returns multiple measurements (one per table row).
    /// </summary>
    public async Task<List<MeasurementData>> ExtractTableMeasurements(DetectedRegion region)
    {
        if (region.CroppedImagePath is null || !File.Exists(region.CroppedImagePath))
        {
            Console.WriteLine($"  Warning: No cropped image for table region {region.Id}");
            return [];
        }

        var imageBytes = await File.ReadAllBytesAsync(region.CroppedImagePath);

        var options = new CreateResponseOptions
        {
            Instructions = """
                You are an expert at reading engineering dimensional analysis tables.
                Extract ALL measurement rows from the provided table image.

                Return a JSON array of objects, one per measurement row:
                [
                    {
                        "bubbleNumber": <integer - the bubble/item number>,
                        "dimensionName": "<dimension name or description>",
                        "nominalValue": "<nominal/target value>",
                        "unit": "<unit of measurement>",
                        "upperTolerance": "<upper tolerance>",
                        "lowerTolerance": "<lower tolerance>",
                        "actualValue": "<actual measured value if present>",
                        "rawText": "<full row text>"
                    }
                ]

                If a field cannot be determined, use null.
                Return ONLY the JSON array, no other text.
                """
        };

        var contentParts = new List<ResponseContentPart>
        {
            ResponseContentPart.CreateInputTextPart($"Extract all measurement data from this table (page {region.PageNumber}):"),
            ResponseContentPart.CreateInputImagePart(BinaryData.FromBytes(imageBytes), "image/png")
        };
        options.InputItems.Add(ResponseItem.CreateUserMessageItem(contentParts));

        try
        {
            var response = await _client.CreateResponseAsync(options);
            var content = GetOutputText(response.Value);

            if (string.IsNullOrEmpty(content)) return [];

            content = StripCodeFences(content);

            var measurements = JsonSerializer.Deserialize<List<MeasurementData>>(content, _jsonOptions);
            return measurements?
                .Select(m => m with { SourcePage = region.PageNumber, SourceType = region.Type })
                .ToList() ?? [];
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Error extracting table on page {region.PageNumber}: {ex.Message}");
            return [];
        }
    }

    /// <summary>
    /// Result of an LLM extraction call, including token usage.
    /// </summary>
    public record LlmExtractionResult(Dictionary<int, string> Dimensions, int InputTokens, int OutputTokens, int TotalTokens);

    /// <summary>
    /// Extracts balloon number → dimension text mapping from a full table page image.
    /// Focused extraction: just the two key columns. Returns token usage data.
    /// </summary>
    public async Task<LlmExtractionResult> ExtractBalloonDimensionsWithUsage(byte[] pageImageBytes, int pageNumber)
    {
        var options = new CreateResponseOptions
        {
            Instructions = """
                You are an expert at reading engineering dimensional analysis tables.
                This image shows a page from a dimensional analysis report.
                
                The table has columns including "BALLOON NO." (or "SN") and "DIMENSION".
                Extract EVERY row's balloon number and dimension value.
                
                Return a JSON array of objects:
                [
                    { "balloonNo": <integer>, "dimension": "<dimension text exactly as shown>" }
                ]
                
                Rules:
                - Include ALL rows, even if some cells are hard to read
                - The balloon number is always an integer (2 through 51)
                - The dimension text should be copied exactly as shown (e.g., "0.81", "18°", "Ø.500", "MATERIAL")
                - If a row has sub-rows or multiple dimension values, include each as a separate entry
                - Do NOT skip any rows
                - Return ONLY the JSON array, no other text
                """
        };

        var contentParts = new List<ResponseContentPart>
        {
            ResponseContentPart.CreateInputTextPart($"Extract all balloon number and dimension pairs from this table page (page {pageNumber}):"),
            ResponseContentPart.CreateInputImagePart(BinaryData.FromBytes(pageImageBytes), "image/png")
        };
        options.InputItems.Add(ResponseItem.CreateUserMessageItem(contentParts));

        try
        {
            var response = await _client.CreateResponseAsync(options);
            var content = GetOutputText(response.Value);
            var usage = response.Value.Usage;
            int inputTokens = usage?.InputTokenCount ?? 0;
            int outputTokens = usage?.OutputTokenCount ?? 0;
            int totalTokens = usage?.TotalTokenCount ?? 0;

            if (string.IsNullOrEmpty(content))
                return new LlmExtractionResult(new(), inputTokens, outputTokens, totalTokens);

            content = StripCodeFences(content);

            var rows = JsonSerializer.Deserialize<List<BalloonDimensionRow>>(content, _jsonOptions);
            if (rows is null)
                return new LlmExtractionResult(new(), inputTokens, outputTokens, totalTokens);

            var result = new Dictionary<int, string>();
            foreach (var row in rows)
            {
                if (row.BalloonNo > 0 && !string.IsNullOrWhiteSpace(row.Dimension))
                    result.TryAdd(row.BalloonNo, row.Dimension.Trim());
            }
            return new LlmExtractionResult(result, inputTokens, outputTokens, totalTokens);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Error extracting balloon dimensions from page {pageNumber}: {ex.Message}");
            return new LlmExtractionResult(new(), 0, 0, 0);
        }
    }

    /// <summary>
    /// Backwards-compatible wrapper that returns just the dimensions dictionary.
    /// </summary>
    public async Task<Dictionary<int, string>> ExtractBalloonDimensions(byte[] pageImageBytes, int pageNumber)
    {
        var result = await ExtractBalloonDimensionsWithUsage(pageImageBytes, pageNumber);
        return result.Dimensions;
    }

    private record BalloonDimensionRow(int BalloonNo, string Dimension);

    private static string GetOutputText(ResponseResult response)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var item in response.OutputItems)
        {
            if (item is MessageResponseItem msg)
            {
                foreach (var part in msg.Content)
                {
                    if (part.Text is not null)
                        sb.Append(part.Text);
                }
            }
        }
        return sb.ToString().Trim();
    }

    private static string StripCodeFences(string content)
    {
        if (content.StartsWith("```"))
        {
            var lines = content.Split('\n');
            var startIdx = lines[0].StartsWith("```") ? 1 : 0;
            var endIdx = lines[^1].Trim() == "```" ? lines.Length - 1 : lines.Length;
            content = string.Join('\n', lines[startIdx..endIdx]);
        }
        return content.Trim();
    }
}
