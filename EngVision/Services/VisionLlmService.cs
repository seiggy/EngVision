using System.Text.Json;
using EngVision.Models;
using OpenAI.Responses;

namespace EngVision.Services;

/// <summary>
/// Uses the OpenAI Responses API to validate that dimension annotations on the
/// engineering drawing match the table values extracted by Tesseract OCR.
/// For each bubble, the pipeline crops the drawing region the leader line points to
/// and sends that crop along with the table dimension text for validation.
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
    /// Result of validating a single bubble's dimension against the drawing.
    /// </summary>
    public record LlmValidationResult(
        int BalloonNo,
        string TableDimension,
        string ObservedDimension,
        bool Matches,
        double Confidence,
        string Notes,
        int InputTokens,
        int OutputTokens,
        int TotalTokens);

    /// <summary>
    /// Validates that the dimension on the drawing matches the table value.
    /// </summary>
    /// <param name="cropImageBytes">PNG bytes of the drawing region the bubble points to.</param>
    /// <param name="balloonNo">The bubble/balloon number.</param>
    /// <param name="tableDimension">The dimension string extracted from the table by Tesseract.</param>
    public async Task<LlmValidationResult> ValidateDimension(byte[] cropImageBytes, int balloonNo, string tableDimension)
    {
        var options = new CreateResponseOptions
        {
            Instructions = """
                You are an expert at reading engineering drawings and dimensional annotations.

                You will be given:
                1. A cropped region from an engineering drawing where a numbered balloon/bubble
                   points via its leader line.
                2. A dimension value extracted from the inspection table for that balloon number.

                Your job is to:
                - Examine the cropped drawing region and find the dimension annotation visible there.
                - Compare it to the table dimension value provided.
                - Determine if they match (accounting for formatting differences like leading zeros,
                  degree symbols, diameter symbols, etc.).

                Return a JSON object:
                {
                    "observedDimension": "<the dimension text you see on the drawing, or empty string if none visible>",
                    "matches": <true if the drawing dimension matches the table value, false otherwise>,
                    "confidence": <0.0 to 1.0 confidence in your assessment>,
                    "notes": "<brief explanation, e.g. 'exact match', 'formatting difference only', 'dimension not visible in crop', etc.>"
                }

                Rules:
                - If you cannot see any dimension annotation in the crop, set observedDimension to ""
                  and matches to false with a low confidence.
                - Treat formatting variations as matches (e.g. '0.81' vs '.81', '18°' vs '18 DEG',
                  'Ø.500' vs 'DIA .500').
                - Return ONLY the JSON object, no other text.
                """
        };

        var contentParts = new List<ResponseContentPart>
        {
            ResponseContentPart.CreateInputTextPart(
                $"Balloon #{balloonNo}\nTable dimension value: \"{tableDimension}\"\n\n" +
                "Examine the drawing region below and validate whether the dimension annotation matches the table value:"),
            ResponseContentPart.CreateInputImagePart(BinaryData.FromBytes(cropImageBytes), "image/png")
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
                return new LlmValidationResult(balloonNo, tableDimension, "", false, 0.0, "", inputTokens, outputTokens, totalTokens);

            content = StripCodeFences(content);
            var parsed = JsonSerializer.Deserialize<ValidationResponse>(content, _jsonOptions);
            if (parsed is null)
                return new LlmValidationResult(balloonNo, tableDimension, "", false, 0.0, "", inputTokens, outputTokens, totalTokens);

            return new LlmValidationResult(
                balloonNo,
                tableDimension,
                parsed.ObservedDimension ?? "",
                parsed.Matches,
                parsed.Confidence,
                parsed.Notes ?? "",
                inputTokens,
                outputTokens,
                totalTokens);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Error validating balloon #{balloonNo}: {ex.Message}");
            return new LlmValidationResult(balloonNo, tableDimension, "", false, 0.0, "", 0, 0, 0);
        }
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

    private record ValidationResponse(string? ObservedDimension, bool Matches, double Confidence, string? Notes);
    private record DiscoveryResponse(string? ObservedDimension, double Confidence, string? Notes);

    /// <summary>
    /// Discovers the dimension annotation in a crop when no table value exists.
    /// Used as a fallback when the table OCR fails to extract a dimension.
    /// </summary>
    public async Task<LlmValidationResult> DiscoverDimension(byte[] cropImageBytes, int balloonNo)
    {
        var options = new CreateResponseOptions
        {
            Instructions = """
                You are an expert at reading engineering drawings and dimensional annotations.

                You will be given a cropped region from an engineering drawing where a numbered
                balloon/bubble points via its leader line.

                Your job is to find and read the dimension annotation visible in the crop.

                Return a JSON object:
                {
                    "observedDimension": "<the dimension text you see on the drawing, or empty string if none visible>",
                    "confidence": <0.0 to 1.0 confidence in your reading>,
                    "notes": "<brief description of what you see>"
                }

                Rules:
                - If you cannot see any dimension annotation, set observedDimension to ""
                  with a low confidence.
                - Include the full dimension text with any symbols (Ø, °, ±, etc.).
                - Return ONLY the JSON object, no other text.
                """
        };

        var contentParts = new List<ResponseContentPart>
        {
            ResponseContentPart.CreateInputTextPart(
                $"Balloon #{balloonNo}\n\nRead the dimension annotation visible in this drawing region:"),
            ResponseContentPart.CreateInputImagePart(BinaryData.FromBytes(cropImageBytes), "image/png")
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
                return new LlmValidationResult(balloonNo, "", "", false, 0.0, "", inputTokens, outputTokens, totalTokens);

            content = StripCodeFences(content);
            var parsed = JsonSerializer.Deserialize<DiscoveryResponse>(content, _jsonOptions);
            if (parsed is null)
                return new LlmValidationResult(balloonNo, "", "", false, 0.0, "", inputTokens, outputTokens, totalTokens);

            var observed = parsed.ObservedDimension ?? "";
            return new LlmValidationResult(
                balloonNo,
                "",
                observed,
                !string.IsNullOrEmpty(observed), // treat as "match" if we see something
                parsed.Confidence,
                parsed.Notes ?? "",
                inputTokens,
                outputTokens,
                totalTokens);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Error discovering dimension for balloon #{balloonNo}: {ex.Message}");
            return new LlmValidationResult(balloonNo, "", "", false, 0.0, "", 0, 0, 0);
        }
    }

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
