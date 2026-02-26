using EngVision.Models;
using EngVision.Services;
using OpenAI;
using OpenAI.Responses;

namespace EngVision.Tests;

/// <summary>
/// Integration tests for VisionLlmService against Azure OpenAI Responses API.
/// Requires .env with AZURE_ENDPOINT, AZURE_KEY, AZURE_DEPLOYMENT_NAME.
/// </summary>
public class VisionLlmTests
{
    private readonly ResponsesClient _client;
    private readonly VisionLlmService _service;
    private readonly string _testDataDir;

    public VisionLlmTests()
    {
        // Load .env from repo root
        var envFile = FindEnvFile();
        if (envFile != null)
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

        var endpoint = Environment.GetEnvironmentVariable("AZURE_ENDPOINT")
            ?? throw new InvalidOperationException("AZURE_ENDPOINT not set");
        var key = Environment.GetEnvironmentVariable("AZURE_KEY")
            ?? throw new InvalidOperationException("AZURE_KEY not set");
        var model = Environment.GetEnvironmentVariable("AZURE_DEPLOYMENT_NAME") ?? "gpt-5.3-codex";

        _client = new ResponsesClient(
            model,
            new System.ClientModel.ApiKeyCredential(key),
            new OpenAIClientOptions { Endpoint = new Uri($"{endpoint.TrimEnd('/')}/openai/v1/") });

        _service = new VisionLlmService(_client);
        _testDataDir = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "TestData");
    }

    [Fact]
    public async Task ResponsesClient_CanSendTextPrompt()
    {
        // Simplest test: send a text-only prompt and get a response
        var options = new CreateResponseOptions
        {
            Instructions = "You are a helpful assistant. Reply with exactly one word."
        };
        options.InputItems.Add(ResponseItem.CreateUserMessageItem("Say hello."));

        var response = await _client.CreateResponseAsync(options);
        var text = GetOutputText(response.Value);

        Assert.False(string.IsNullOrWhiteSpace(text), "Expected non-empty response from model");
        Console.WriteLine($"Text prompt response: {text}");
    }

    [Fact]
    public async Task ResponsesClient_CanSendImagePrompt()
    {
        // Send a bubble crop image and ask the model to describe it
        var imagePath = Path.Combine(_testDataDir, "bubble_sample.png");
        Assert.True(File.Exists(imagePath), $"Test image not found: {imagePath}");

        var imageBytes = await File.ReadAllBytesAsync(imagePath);

        var options = new CreateResponseOptions
        {
            Instructions = "You are an expert at reading engineering drawings. Be concise."
        };

        var contentParts = new List<ResponseContentPart>
        {
            ResponseContentPart.CreateInputTextPart("What number is inside this circle? Reply with just the number."),
            ResponseContentPart.CreateInputImagePart(BinaryData.FromBytes(imageBytes), "image/png")
        };
        options.InputItems.Add(ResponseItem.CreateUserMessageItem(contentParts));

        var response = await _client.CreateResponseAsync(options);
        var text = GetOutputText(response.Value);

        Assert.False(string.IsNullOrWhiteSpace(text), "Expected non-empty response for image prompt");
        Console.WriteLine($"Image prompt response: {text}");

        // bubble_007.png was OCR'd as Bubble #2
        Assert.Contains("2", text);
    }

    [Fact]
    public async Task ExtractBubbleMeasurement_ReturnsStructuredData()
    {
        var imagePath = Path.Combine(_testDataDir, "bubble_sample.png");
        Assert.True(File.Exists(imagePath), $"Test image not found: {imagePath}");

        var region = new DetectedRegion
        {
            Id = 1,
            PageNumber = 1,
            Type = RegionType.Bubble,
            BoundingBox = new BoundingBox(0, 0, 50, 50),
            BubbleNumber = 2,
            CroppedImagePath = imagePath
        };

        var result = await _service.ExtractBubbleMeasurement(region);

        // The crop is just the bubble circle, so we mainly check the service doesn't crash
        // and returns some structured data
        Assert.NotNull(result);
        Console.WriteLine($"Bubble measurement result: BubbleNumber={result.BubbleNumber}, " +
                         $"DimensionName={result.DimensionName}, RawText={result.RawText}");
    }

    [Fact]
    public async Task ExtractTableMeasurements_ReturnsMultipleRows()
    {
        var imagePath = Path.Combine(_testDataDir, "table_page.png");
        Assert.True(File.Exists(imagePath), $"Test image not found: {imagePath}");

        var region = new DetectedRegion
        {
            Id = 100,
            PageNumber = 2,
            Type = RegionType.TableRegion,
            BoundingBox = new BoundingBox(0, 0, 2550, 3299),
            CroppedImagePath = imagePath
        };

        var results = await _service.ExtractTableMeasurements(region);

        Assert.NotNull(results);
        Assert.NotEmpty(results);
        Console.WriteLine($"Table extraction returned {results.Count} rows");
        foreach (var row in results.Take(5))
        {
            Console.WriteLine($"  Balloon #{row.BubbleNumber}: {row.DimensionName} = {row.NominalValue}");
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

    private static string? FindEnvFile()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var envPath = Path.Combine(dir.FullName, ".env");
            if (File.Exists(envPath)) return envPath;
            dir = dir.Parent;
        }
        return null;
    }
}
