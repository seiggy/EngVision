using EngVision.Services;

namespace EngVision.Tests;

public class DimensionMatcherTests
{
    [Theory]
    // Should match: OCR char differences
    [InlineData("FILLET RADDI 0.03", "FILLET RADI 0.03", true)]
    [InlineData("RO.06", "R0.06", true)]
    [InlineData("RO.25", "R0.25", true)]
    [InlineData("CARBURIZE-RC45 MIN.", "CARBURIZE Rc45 MIN.", true)]
    [InlineData("CARBURIZE-RC45 MIN.", "CARBURIZE:RC45 MIN.", true)]
    [InlineData("FILLET RADDI 0.03", "FILLET RADII 0.03", true)]
    // Should match: exact
    [InlineData("0.81", "0.81", true)]
    [InlineData("18°", "18°", true)]
    [InlineData("45°", "45°", true)]
    // Should NOT match: too different (LLM misread)
    [InlineData("1.5730-16NS-3", "12.5736-15N-3", false)]
    // Should NOT match: totally different values
    [InlineData("MATERIAL", "LQ", false)]
    [InlineData("0.81", "1.500", false)]
    [InlineData("45°", "0.06", false)]
    // Should match: diameter symbol variation
    [InlineData("0.250 / 0.280", "Ø.250 / Ø.280", true)]
    public void AreSimilar_MatchesCorrectly(string a, string b, bool expected)
    {
        var result = DimensionMatcher.AreSimilar(a, b);
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData("0.81", "0.81", 1.0)]
    [InlineData(null, "0.81", 0.0)]
    [InlineData("0.81", null, 0.0)]
    [InlineData(null, null, 0.0)]
    [InlineData("FILLET RADDI 0.03", "FILLET RADI 0.03", 0.93, 1.0)]   // 1 char diff over 17
    [InlineData("CARBURIZE-RC45 MIN.", "CARBURIZE:RC45 MIN.", 0.94, 0.96)] // 1 char diff over 19
    [InlineData("RO.06", "R0.06", 0.79, 0.81)]                          // 1 char diff over 5
    [InlineData("1.5730-16NS-3", "12.5736-15N-3", 0.0, 0.74)]          // too different
    [InlineData("MATERIAL", "LQ", 0.0, 0.3)]                            // totally different
    public void ConfidenceScore_InExpectedRange(string? a, string? b, double minExpected, double maxExpected = 1.0)
    {
        var score = DimensionMatcher.ConfidenceScore(a, b);
        Assert.InRange(score, minExpected, maxExpected);
    }
}
