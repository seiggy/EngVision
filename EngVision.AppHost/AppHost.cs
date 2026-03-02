var builder = DistributedApplication.CreateBuilder(args);

// ── Load .env file from repo root if present ────────────────────────────────
var envFile = Path.Combine(builder.AppHostDirectory, "..", ".env");
if (File.Exists(envFile))
{
    foreach (var line in File.ReadAllLines(envFile))
    {
        var trimmed = line.Trim();
        if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith('#')) continue;
        var eqIdx = trimmed.IndexOf('=');
        if (eqIdx <= 0) continue;
        var key = trimmed[..eqIdx].Trim();
        var val = trimmed[(eqIdx + 1)..].Trim();
        Environment.SetEnvironmentVariable(key, val);
    }
}

// ── Helper: read env var (from .env or system) ──────────────────────────────
string? EnvOrNull(string name) => Environment.GetEnvironmentVariable(name);

// ── API backend toggle ─────────────────────────────────────────────────────────
// Choose which API backend to run: "dotnet" (default) or "python".
//
// Ways to set:
//   1. appsettings.json      →  "ApiBackend": "python"
//   2. Environment variable  →  set ApiBackend=python
//   3. CLI argument           →  aspire run -- --ApiBackend python
var usePython = string.Equals(
    builder.Configuration["ApiBackend"], "python", StringComparison.OrdinalIgnoreCase);

if (usePython)
{
    var api = builder.AddUvicornApp("api", "../engvision-py", "engvision.app:app")
        .WithUv();

    // Pass OCR provider config if set in AppHost configuration;
    // otherwise let the Python process read from its own .env file
    var ocrProvider = builder.Configuration["OCR_PROVIDER"] ?? EnvOrNull("OCR_PROVIDER");
    if (!string.IsNullOrEmpty(ocrProvider))
        api = api.WithEnvironment("OCR_PROVIDER", ocrProvider);
    var docIntEndpoint = builder.Configuration["AZURE_DOCINT_ENDPOINT"] ?? EnvOrNull("AZURE_DOCINT_ENDPOINT");
    if (!string.IsNullOrEmpty(docIntEndpoint))
        api = api.WithEnvironment("AZURE_DOCINT_ENDPOINT", docIntEndpoint);
    var docIntKey = builder.Configuration["AZURE_DOCINT_KEY"] ?? EnvOrNull("AZURE_DOCINT_KEY");
    if (!string.IsNullOrEmpty(docIntKey))
        api = api.WithEnvironment("AZURE_DOCINT_KEY", docIntKey);

    builder.AddNpmApp("frontend", "../EngVision.Web", "dev")
        .WithReference(api).WaitFor(api)
        .WithHttpEndpoint(env: "PORT")
        .WithExternalHttpEndpoints();
}
else
{
    var api = builder.AddProject<Projects.EngVision_Api>("api");

    // Pass OCR + Azure OpenAI env vars to the .NET project
    foreach (var envName in new[] {
        "OCR_PROVIDER", "AZURE_DOCINT_ENDPOINT", "AZURE_DOCINT_KEY",
        "AZURE_ENDPOINT", "AZURE_KEY", "AZURE_DEPLOYMENT_NAME" })
    {
        var val = builder.Configuration[envName] ?? EnvOrNull(envName);
        if (!string.IsNullOrEmpty(val))
            api = api.WithEnvironment(envName, val);
    }

    builder.AddNpmApp("frontend", "../EngVision.Web", "dev")
        .WithReference(api).WaitFor(api)
        .WithHttpEndpoint(env: "PORT")
        .WithExternalHttpEndpoints();
}

builder.Build().Run();
