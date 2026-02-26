namespace EngVision.Services;

/// <summary>
/// Fuzzy matching for dimension text extracted via OCR/LLM.
/// Uses different strategies for numeric tokens (strict) vs word tokens (lenient).
/// </summary>
public static class DimensionMatcher
{
    /// <summary>
    /// Returns true if two dimension strings are "essentially the same"
    /// accounting for OCR noise.
    /// </summary>
    public static bool AreSimilar(string a, string b) => ConfidenceScore(a, b) >= 0.75;

    /// <summary>
    /// Computes a confidence score between 0 and 1 indicating how similar two
    /// dimension strings are. 1.0 = exact match, 0.0 = completely different/missing.
    /// Uses Levenshtein distance on uppercased, whitespace-trimmed strings.
    /// </summary>
    public static double ConfidenceScore(string? a, string? b)
    {
        if (a is null || b is null)
            return 0.0;

        var na = NormalizeWhitespace(a).ToUpperInvariant();
        var nb = NormalizeWhitespace(b).ToUpperInvariant();

        if (na == nb)
            return 1.0;

        int maxLen = Math.Max(na.Length, nb.Length);
        if (maxLen == 0) return 1.0;

        int distance = LevenshteinDistance(na, nb);
        return Math.Round(1.0 - (double)distance / maxLen, 4);
    }

    /// <summary>
    /// Compares two individual tokens using type-appropriate rules.
    /// Numeric tokens: strict (exact after OCR char normalization).
    /// Word tokens: case-insensitive with Levenshtein tolerance.
    /// Mixed tokens (e.g. "RC45"): normalize OCR chars then compare.
    /// </summary>
    private static bool TokensMatch(string a, string b) => TokenConfidence(a, b) >= 0.75;

    /// <summary>
    /// Returns a 0-1 confidence score for how well two individual tokens match.
    /// </summary>
    private static double TokenConfidence(string a, string b)
    {
        if (string.Equals(a, b, StringComparison.OrdinalIgnoreCase))
            return 1.0;

        bool aIsNumeric = IsNumericToken(a);
        bool bIsNumeric = IsNumericToken(b);

        if (aIsNumeric && bIsNumeric)
        {
            var na = NormalizeNumeric(a);
            var nb = NormalizeNumeric(b);
            if (na == nb) return 1.0;
            int maxLen = Math.Max(na.Length, nb.Length);
            if (maxLen == 0) return 1.0;
            return 1.0 - (double)LevenshteinDistance(na, nb) / maxLen;
        }

        if (!aIsNumeric && !bIsNumeric)
        {
            var ua = FixCommonWords(a.ToUpperInvariant());
            var ub = FixCommonWords(b.ToUpperInvariant());
            if (ua == ub) return 1.0;
            int maxLen = Math.Max(ua.Length, ub.Length);
            if (maxLen == 0) return 1.0;
            return 1.0 - (double)LevenshteinDistance(ua, ub) / maxLen;
        }

        // Mixed tokens
        var ma = NormalizeMixed(a).ToUpperInvariant();
        var mb = NormalizeMixed(b).ToUpperInvariant();
        if (ma == mb) return 1.0;
        int maxLenM = Math.Max(ma.Length, mb.Length);
        if (maxLenM == 0) return 1.0;
        return 1.0 - (double)LevenshteinDistance(ma, mb) / maxLenM;
    }

    /// <summary>
    /// A token is "numeric" if it's primarily digits, decimal points, slashes, and
    /// common OCR-for-digit characters.
    /// </summary>
    private static bool IsNumericToken(string s)
    {
        int digitLike = 0;
        foreach (var c in s)
        {
            if (char.IsDigit(c) || c == '.' || c == '/' || c == '-' || c == '°'
                || c == 'O' || c == 'o' || c == 'Ø' || c == '⌀')
                digitLike++;
        }
        return s.Length > 0 && (double)digitLike / s.Length > 0.6;
    }

    /// <summary>
    /// Normalize a numeric token: fix OCR char substitutions.
    /// </summary>
    private static string NormalizeNumeric(string s)
    {
        var chars = s.ToCharArray();
        for (int i = 0; i < chars.Length; i++)
        {
            chars[i] = chars[i] switch
            {
                'O' or 'o' => '0',
                'I' or 'l' => '1',
                'S' when IsDigitContext(chars, i) => '5',
                'B' when IsDigitContext(chars, i) => '8',
                'Ø' or '⌀' => '0',
                '°' or 'º' => '°',
                _ => chars[i]
            };
        }
        return new string(chars);
    }

    /// <summary>
    /// Normalize a mixed token (e.g. "RC45", "RO.06") for comparison.
    /// </summary>
    private static string NormalizeMixed(string s)
    {
        var result = s.ToUpperInvariant();
        // In mixed tokens like "RO.06", the O adjacent to digits/dots is likely 0
        var chars = result.ToCharArray();
        for (int i = 0; i < chars.Length; i++)
        {
            bool adjDigit = (i > 0 && (char.IsDigit(chars[i - 1]) || chars[i - 1] == '.'))
                         || (i < chars.Length - 1 && (char.IsDigit(chars[i + 1]) || chars[i + 1] == '.'));
            if (chars[i] == 'O' && adjDigit) chars[i] = '0';
        }
        return new string(chars);
    }

    private static bool IsDigitContext(char[] chars, int i)
    {
        bool prev = i > 0 && (char.IsDigit(chars[i - 1]) || chars[i - 1] == '.');
        bool next = i < chars.Length - 1 && (char.IsDigit(chars[i + 1]) || chars[i + 1] == '.');
        return prev && next;
    }

    private static string FixCommonWords(string s)
    {
        return s
            .Replace("RADDI", "RADII")
            .Replace("RADI", "RADII")
            .Replace("DIAMATER", "DIAMETER")
            .Replace("DIMENTIONAL", "DIMENSIONAL");
    }

    private static string NormalizeWhitespace(string s)
    {
        return System.Text.RegularExpressions.Regex.Replace(s.Trim(), @"\s+", " ");
    }

    private static string[] Tokenize(string s)
    {
        // Split on whitespace, hyphens, and colons (OCR often misreads - as :)
        return System.Text.RegularExpressions.Regex.Split(s, @"[\s\-:]+")
            .Where(t => t.Length > 0)
            .ToArray();
    }

    private static int LevenshteinDistance(string s, string t)
    {
        int n = s.Length, m = t.Length;
        if (n == 0) return m;
        if (m == 0) return n;

        var d = new int[n + 1, m + 1];
        for (int i = 0; i <= n; i++) d[i, 0] = i;
        for (int j = 0; j <= m; j++) d[0, j] = j;

        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int cost = s[i - 1] == t[j - 1] ? 0 : 1;
                d[i, j] = Math.Min(
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                    d[i - 1, j - 1] + cost);
            }
        }

        return d[n, m];
    }
}
