using System;
using System.Collections.Generic;
using ManagedCuda.BasicTypes;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// Describes a memset operation to be inserted into the CUDA graph.
/// Used primarily for resetting AppendBuffer counters to 0 before kernel execution.
/// </summary>
internal sealed class MemsetDescriptor
{
    public Guid Id { get; } = Guid.NewGuid();
    public CUdeviceptr Destination { get; }
    public uint Value { get; }
    public uint ElementSize { get; }
    public ulong Width { get; }
    public string DebugName { get; }

    /// <summary>
    /// Kernel node IDs that depend on this memset (must complete before kernels run).
    /// </summary>
    public HashSet<Guid> DependentKernelNodeIds { get; } = new();

    public MemsetDescriptor(CUdeviceptr destination, uint value, uint elementSize, ulong width, string debugName)
    {
        Destination = destination;
        Value = value;
        ElementSize = elementSize;
        Width = width;
        DebugName = debugName;
    }
}
