using System;
using System.Collections.Generic;
using System.Linq;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Graph;

namespace VL.Cuda.Core.Blocks.Builder;

/// <summary>
/// A kernel entry in a block description. Stores everything the engine
/// needs to recreate the KernelNode: PTX path, handle ID (for port/param
/// resolution), and grid dimensions.
/// </summary>
public sealed class KernelEntry
{
    public Guid HandleId { get; }
    public string PtxPath { get; }
    public string EntryPoint { get; }
    public uint GridDimX { get; }
    public uint GridDimY { get; }
    public uint GridDimZ { get; }

    public KernelEntry(Guid handleId, string ptxPath, string entryPoint,
        uint gridDimX, uint gridDimY, uint gridDimZ)
    {
        HandleId = handleId;
        PtxPath = ptxPath;
        EntryPoint = entryPoint;
        GridDimX = gridDimX;
        GridDimY = gridDimY;
        GridDimZ = gridDimZ;
    }

    /// <summary>
    /// Structural equality ignores HandleId (changes every construction).
    /// Compares PTX path, entry point, and grid dimensions.
    /// </summary>
    public bool StructuralEquals(KernelEntry? other)
    {
        if (other is null) return false;
        return PtxPath == other.PtxPath &&
               EntryPoint == other.EntryPoint &&
               GridDimX == other.GridDimX &&
               GridDimY == other.GridDimY &&
               GridDimZ == other.GridDimZ;
    }
}

/// <summary>
/// Describes an AppendBuffer output within a block: which kernel params hold
/// the data pointer and counter pointer, and the maximum capacity.
/// </summary>
public sealed class AppendBufferInfo
{
    public Guid BlockId { get; }
    public string PortName { get; }
    public string CountPortName => $"{PortName} Count";
    public Guid DataKernelHandleId { get; }
    public int DataParamIndex { get; }
    public Guid CounterKernelHandleId { get; }
    public int CounterParamIndex { get; }
    public int MaxCapacity { get; }
    public int ElementSize { get; }

    public AppendBufferInfo(Guid blockId, string portName,
        Guid dataKernelHandleId, int dataParamIndex,
        Guid counterKernelHandleId, int counterParamIndex,
        int maxCapacity, int elementSize)
    {
        BlockId = blockId;
        PortName = portName;
        DataKernelHandleId = dataKernelHandleId;
        DataParamIndex = dataParamIndex;
        CounterKernelHandleId = counterKernelHandleId;
        CounterParamIndex = counterParamIndex;
        MaxCapacity = maxCapacity;
        ElementSize = elementSize;
    }

    public bool StructuralEquals(AppendBufferInfo? other)
    {
        if (other is null) return false;
        return PortName == other.PortName &&
               DataParamIndex == other.DataParamIndex &&
               CounterParamIndex == other.CounterParamIndex &&
               MaxCapacity == other.MaxCapacity &&
               ElementSize == other.ElementSize;
    }
}

/// <summary>
/// Returned by BlockBuilder.AppendOutput to give block authors access to the data port
/// and the auto-generated count port name.
/// </summary>
public sealed class AppendOutputPort
{
    public BlockPort DataPort { get; }
    public AppendBufferInfo Info { get; }
    public string CountPortName => Info.CountPortName;

    internal AppendOutputPort(BlockPort dataPort, AppendBufferInfo info)
    {
        DataPort = dataPort;
        Info = info;
    }
}

/// <summary>
/// A captured library operation entry in a block description.
/// Stores the handle ID, descriptor, and capture action needed to recreate the CapturedNode.
/// </summary>
public sealed class CapturedEntry
{
    public Guid HandleId { get; }
    public CapturedNodeDescriptor Descriptor { get; }
    public Action<CUstream, CUdeviceptr[]> CaptureAction { get; }

    public CapturedEntry(Guid handleId, CapturedNodeDescriptor descriptor, Action<CUstream, CUdeviceptr[]> captureAction)
    {
        HandleId = handleId;
        Descriptor = descriptor;
        CaptureAction = captureAction;
    }

    /// <summary>
    /// Structural equality ignores HandleId (changes every construction).
    /// Compares descriptor debug name and parameter counts.
    /// </summary>
    public bool StructuralEquals(CapturedEntry? other)
    {
        if (other is null) return false;
        return Descriptor.DebugName == other.Descriptor.DebugName &&
               Descriptor.Inputs.Count == other.Descriptor.Inputs.Count &&
               Descriptor.Outputs.Count == other.Descriptor.Outputs.Count &&
               Descriptor.Scalars.Count == other.Descriptor.Scalars.Count;
    }
}

/// <summary>
/// Immutable snapshot of a block's structural description.
/// Used for change detection and graph rebuilding.
/// </summary>
public sealed class BlockDescription
{
    /// <summary>
    /// Kernel entries in AddKernel order. Each stores HandleId, PTX path, grid dims.
    /// </summary>
    public IReadOnlyList<KernelEntry> KernelEntries { get; }

    /// <summary>
    /// Captured library operation entries in AddCaptured order.
    /// </summary>
    public IReadOnlyList<CapturedEntry> CapturedEntries { get; }

    /// <summary>
    /// Port entries: (Name, Direction, PinType) in order.
    /// </summary>
    public IReadOnlyList<(string Name, PortDirection Direction, PinType Type)> Ports { get; }

    /// <summary>
    /// Internal connections: (SrcKernelIndex, SrcParam, TgtKernelIndex, TgtParam).
    /// Uses kernel index (not GUID) for stable comparison across hot-swap.
    /// </summary>
    public IReadOnlyList<(int SrcKernelIndex, int SrcParam, int TgtKernelIndex, int TgtParam)> InternalConnections { get; }

    /// <summary>
    /// AppendBuffer descriptions for variable-length outputs.
    /// </summary>
    public IReadOnlyList<AppendBufferInfo> AppendBuffers { get; }

    public BlockDescription(
        IReadOnlyList<KernelEntry> kernelEntries,
        IReadOnlyList<(string Name, PortDirection Direction, PinType Type)> ports,
        IReadOnlyList<(int SrcKernelIndex, int SrcParam, int TgtKernelIndex, int TgtParam)> internalConnections,
        IReadOnlyList<AppendBufferInfo>? appendBuffers = null,
        IReadOnlyList<CapturedEntry>? capturedEntries = null)
    {
        KernelEntries = kernelEntries;
        Ports = ports;
        InternalConnections = internalConnections;
        AppendBuffers = appendBuffers ?? Array.Empty<AppendBufferInfo>();
        CapturedEntries = capturedEntries ?? Array.Empty<CapturedEntry>();
    }

    /// <summary>
    /// Structural equality: same kernels (path + grid), captured ops, ports, and connections.
    /// HandleIds are ignored — they change every construction.
    /// </summary>
    public bool StructuralEquals(BlockDescription? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;

        if (KernelEntries.Count != other.KernelEntries.Count) return false;
        if (CapturedEntries.Count != other.CapturedEntries.Count) return false;
        if (Ports.Count != other.Ports.Count) return false;
        if (InternalConnections.Count != other.InternalConnections.Count) return false;
        if (AppendBuffers.Count != other.AppendBuffers.Count) return false;

        for (int i = 0; i < KernelEntries.Count; i++)
        {
            if (!KernelEntries[i].StructuralEquals(other.KernelEntries[i]))
                return false;
        }

        for (int i = 0; i < CapturedEntries.Count; i++)
        {
            if (!CapturedEntries[i].StructuralEquals(other.CapturedEntries[i]))
                return false;
        }

        for (int i = 0; i < Ports.Count; i++)
        {
            if (Ports[i].Name != other.Ports[i].Name ||
                Ports[i].Direction != other.Ports[i].Direction ||
                !Ports[i].Type.Equals(other.Ports[i].Type))
                return false;
        }

        for (int i = 0; i < InternalConnections.Count; i++)
        {
            var a = InternalConnections[i];
            var b = other.InternalConnections[i];
            if (a.SrcKernelIndex != b.SrcKernelIndex || a.SrcParam != b.SrcParam ||
                a.TgtKernelIndex != b.TgtKernelIndex || a.TgtParam != b.TgtParam)
                return false;
        }

        for (int i = 0; i < AppendBuffers.Count; i++)
        {
            if (!AppendBuffers[i].StructuralEquals(other.AppendBuffers[i]))
                return false;
        }

        return true;
    }
}
