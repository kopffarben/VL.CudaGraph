using System;
using System.Collections.Generic;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// Result of graph validation. Contains errors and warnings.
/// </summary>
public sealed class ValidationResult
{
    private readonly List<string> _errors = new();
    private readonly List<string> _warnings = new();

    public IReadOnlyList<string> Errors => _errors;
    public IReadOnlyList<string> Warnings => _warnings;
    public bool IsValid => _errors.Count == 0;

    internal void AddError(string message) => _errors.Add(message);
    internal void AddWarning(string message) => _warnings.Add(message);

    public override string ToString()
    {
        if (IsValid) return "Valid";
        return $"Invalid: {string.Join("; ", _errors)}";
    }
}
