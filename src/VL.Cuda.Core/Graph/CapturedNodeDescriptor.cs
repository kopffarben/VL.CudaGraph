using System;
using System.Collections.Generic;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// Describes the input/output/scalar parameters of a captured library operation.
/// Analogous to KernelDescriptor for KernelNodes, but for stream-captured operations.
/// </summary>
public sealed class CapturedNodeDescriptor
{
    public string DebugName { get; }
    public IReadOnlyList<CapturedParam> Inputs { get; }
    public IReadOnlyList<CapturedParam> Outputs { get; }
    public IReadOnlyList<CapturedParam> Scalars { get; }

    public CapturedNodeDescriptor(
        string debugName,
        IReadOnlyList<CapturedParam>? inputs = null,
        IReadOnlyList<CapturedParam>? outputs = null,
        IReadOnlyList<CapturedParam>? scalars = null)
    {
        DebugName = debugName;
        Inputs = inputs ?? Array.Empty<CapturedParam>();
        Outputs = outputs ?? Array.Empty<CapturedParam>();
        Scalars = scalars ?? Array.Empty<CapturedParam>();
    }

    /// <summary>
    /// Total parameter count across all categories.
    /// </summary>
    public int TotalParamCount => Inputs.Count + Outputs.Count + Scalars.Count;
}

/// <summary>
/// Describes a single parameter of a captured library operation.
/// </summary>
public sealed class CapturedParam
{
    public string Name { get; }
    public string Type { get; }
    public bool IsPointer { get; }

    public CapturedParam(string name, string type, bool isPointer = true)
    {
        Name = name;
        Type = type;
        IsPointer = isPointer;
    }

    /// <summary>
    /// Create a pointer parameter (buffer input/output).
    /// </summary>
    public static CapturedParam Pointer(string name, string type) => new(name, type, isPointer: true);

    /// <summary>
    /// Create a scalar parameter (e.g., alpha/beta for BLAS).
    /// </summary>
    public static CapturedParam Scalar(string name, string type) => new(name, type, isPointer: false);
}
