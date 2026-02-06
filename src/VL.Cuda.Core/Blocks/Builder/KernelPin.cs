using System;

namespace VL.Cuda.Core.Blocks.Builder;

/// <summary>
/// Reference to a specific parameter on a kernel handle within a block.
/// Used by BlockBuilder.Input/Output/InputScalar to map block ports to kernel params.
/// </summary>
public readonly struct KernelPin
{
    public Guid KernelHandleId { get; }
    public int ParamIndex { get; }

    public KernelPin(Guid kernelHandleId, int paramIndex)
    {
        KernelHandleId = kernelHandleId;
        ParamIndex = paramIndex;
    }
}
