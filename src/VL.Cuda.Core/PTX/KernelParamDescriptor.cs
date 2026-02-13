namespace VL.Cuda.Core.PTX;

/// <summary>
/// Direction of a kernel parameter (for documentation/validation).
/// </summary>
public enum ParamDirection
{
    In,
    Out,
    InOut,
}

/// <summary>
/// Describes a single kernel parameter from JSON metadata.
/// </summary>
public sealed class KernelParamDescriptor
{
    public required string Name { get; init; }
    public required string Type { get; init; }
    public required int Index { get; init; }
    public ParamDirection Direction { get; init; } = ParamDirection.In;

    /// <summary>
    /// Whether this parameter is a pointer (buffer) vs a scalar value.
    /// </summary>
    public bool IsPointer { get; init; }

    /// <summary>
    /// Explicit size in bytes for this parameter slot. When 0, the size is
    /// computed automatically (IntPtr.Size for pointers, type-based for scalars).
    /// Used for ILGPU ArrayView struct params (16 bytes: pointer + length packed).
    /// </summary>
    public int SizeBytes { get; init; }
}
