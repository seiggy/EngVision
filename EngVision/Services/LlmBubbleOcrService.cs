using OpenAI;
using OpenAI.Responses;
using OpenCvSharp;
using System.Text.RegularExpressions;

namespace EngVision.Services;

/// <summary>
/// Uses the LLM vision model (e.g. GPT-5.3-Codex) to read bubble numbers from crop images.
/// Much faster than Doc Intelligence when run in parallel, and more accurate for simple digits.
/// </summary>
public class LlmBubbleOcrService : IBubbleOcrService
{
    private readonly ResponsesClient _client;
    private const int MaxParallelism = 5;

    public LlmBubbleOcrService(ResponsesClient client)
    {
        _client = client;
    }

    public int? ExtractBubbleNumber(string cropImagePath)
    {
        var bytes = File.ReadAllBytes(cropImagePath);
        return ExtractBubbleNumberAsync(bytes).GetAwaiter().GetResult();
    }

    public int? ExtractBubbleNumber(Mat src)
    {
        Cv2.ImEncode(".png", src, out var buf);
        return ExtractBubbleNumberAsync(buf).GetAwaiter().GetResult();
    }

    public async Task<int?> ExtractBubbleNumberAsync(byte[] imageBytes)
    {
        var options = new CreateResponseOptions
        {
            Instructions = """
                You are reading a small cropped image from an engineering drawing.
                The image contains a circle (balloon/bubble) with a number inside it.
                The number is between 1 and 99.

                Return ONLY the integer number you see. Nothing else.
                If you cannot read a number, return "null".
                """
        };

        var contentParts = new List<ResponseContentPart>
        {
            ResponseContentPart.CreateInputTextPart("What number is in this bubble?"),
            ResponseContentPart.CreateInputImagePart(BinaryData.FromBytes(imageBytes), "image/png")
        };
        options.InputItems.Add(ResponseItem.CreateUserMessageItem(contentParts));

        try
        {
            var response = await _client.CreateResponseAsync(options);
            var text = GetOutputText(response.Value).Trim();

            // Parse the response — should be just a number
            var digits = Regex.Replace(text, @"[^0-9]", "");
            if (int.TryParse(digits, out var num) && num is >= 1 and <= 99)
                return num;
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [LLM-OCR] Error: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Extracts all bubble numbers in parallel batches.
    /// Returns results plus calls onProgress after each completion.
    /// </summary>
    public async Task<Dictionary<string, int?>> ExtractAllAsync(
        string cropDirectory,
        Func<int, string, int?, Task>? onProgress = null)
    {
        var files = Directory.GetFiles(cropDirectory, "bubble_*.png")
            .OrderBy(f => f)
            .ToArray();

        var results = new Dictionary<string, int?>();
        var semaphore = new SemaphoreSlim(MaxParallelism);
        var completed = 0;

        var tasks = files.Select(async path =>
        {
            await semaphore.WaitAsync();
            try
            {
                var filename = Path.GetFileName(path);
                var bytes = await File.ReadAllBytesAsync(path);
                var number = await ExtractBubbleNumberAsync(bytes);

                lock (results)
                {
                    results[filename] = number;
                }

                var idx = Interlocked.Increment(ref completed);
                Console.WriteLine($"  [LLM-OCR] Bubble {idx}/{files.Length}: {filename} → {number?.ToString() ?? "null"}");

                if (onProgress != null)
                    await onProgress(idx, filename, number);
            }
            finally
            {
                semaphore.Release();
            }
        }).ToArray();

        await Task.WhenAll(tasks);
        return results;
    }

    public Dictionary<string, int?> ExtractAll(string cropDirectory)
    {
        return ExtractAllAsync(cropDirectory).GetAwaiter().GetResult();
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

    public void Dispose() { }
}
