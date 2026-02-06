using System;
using System.Collections.Generic;

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
    /// Port entries: (Name, Direction, PinType) in order.
    /// </summary>
    public IReadOnlyList<(string Name, PortDirection Direction, PinType Type)> Ports { get; }

    /// <summary>
    /// Internal connections: (SrcKernelIndex, SrcParam, TgtKernelIndex, TgtParam).
    /// Uses kernel index (not GUID) for stable comparison across hot-swap.
    /// </summary>
    public IReadOnlyList<(int SrcKernelIndex, int SrcParam, int TgtKernelIndex, int TgtParam)> InternalConnections { get; }

    public BlockDescription(
        IReadOnlyList<KernelEntry> kernelEntries,
        IReadOnlyList<(string Name, PortDirection Direction, PinType Type)> ports,
        IReadOnlyList<(int SrcKernelIndex, int SrcParam, int TgtKernelIndex, int TgtParam)> internalConnections)
    {
        KernelEntries = kernelEntries;
        Ports = ports;
        InternalConnections = internalConnections;
    }

    /// <summary>
    /// Structural equality: same kernels (path + grid), ports, and connections.
    /// HandleIds are ignored — they change every construction.
    /// </summary>
    public bool StructuralEquals(BlockDescription? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;

        if (KernelEntries.Count != other.KernelEntries.Count) return false;
        if (Ports.Count != other.Ports.Count) return false;
        if (InternalConnections.Count != other.InternalConnections.Count) return false;

        for (int i = 0; i < KernelEntries.Count; i++)
        {
            if (!KernelEntries[i].StructuralEquals(other.KernelEntries[i]))
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

        return true;
    }
}
