var builder = DistributedApplication.CreateBuilder(args);

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

    builder.AddNpmApp("frontend", "../EngVision.Web", "dev")
        .WithReference(api).WaitFor(api)
        .WithHttpEndpoint(env: "PORT")
        .WithExternalHttpEndpoints();
}
else
{
    var api = builder.AddProject<Projects.EngVision_Api>("api");

    builder.AddNpmApp("frontend", "../EngVision.Web", "dev")
        .WithReference(api).WaitFor(api)
        .WithHttpEndpoint(env: "PORT")
        .WithExternalHttpEndpoints();
}

builder.Build().Run();
