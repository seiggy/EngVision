var builder = DistributedApplication.CreateBuilder(args);

var api = builder.AddProject<Projects.EngVision_Api>("api");

var frontend = builder.AddNpmApp("frontend", "../EngVision.Web", "dev")
    .WithReference(api)
    .WaitFor(api)
    .WithHttpEndpoint(env: "PORT")
    .WithExternalHttpEndpoints();

builder.Build().Run();
