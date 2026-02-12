using System;

namespace VL.Cuda.Core.Blocks.Builder;

/// <summary>
/// Reference to a specific parameter on a captured handle within a block.
/// Analogous to KernelPin for KernelHandle.
/// </summary>
public readonly struct CapturedPin
{
    public Guid CapturedHandleId { get; }
    public int ParamIndex { get; }

    /// <summary>
    /// Which category this pin references: Input, Output, or Scalar.
    /// </summary>
    public CapturedPinCategory Category { get; }

    public CapturedPin(Guid capturedHandleId, int paramIndex, CapturedPinCategory category)
    {
        CapturedHandleId = capturedHandleId;
        ParamIndex = paramIndex;
        Category = category;
    }
}

/// <summary>
/// Identifies which parameter list a CapturedPin references.
/// </summary>
public enum CapturedPinCategory
{
    Input,
    Output,
    Scalar,
}
