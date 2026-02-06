using System;
using VL.Cuda.Core.PTX;

namespace VL.Cuda.Core.Blocks.Builder;

/// <summary>
/// Represents a kernel added to a block via BlockBuilder.AddKernel().
/// Provides In()/Out() to reference specific parameters for port binding.
/// </summary>
public sealed class KernelHandle
{
    public Guid Id { get; }
    public string PtxPath { get; }
    public KernelDescriptor Descriptor { get; }

    public uint GridDimX { get; set; } = 1;
    public uint GridDimY { get; set; } = 1;
    public uint GridDimZ { get; set; } = 1;

    public KernelHandle(string ptxPath, KernelDescriptor descriptor)
    {
        Id = Guid.NewGuid();
        PtxPath = ptxPath;
        Descriptor = descriptor;
    }

    /// <summary>
    /// Reference to an input parameter by index.
    /// </summary>
    public KernelPin In(int index) => new(Id, index);

    /// <summary>
    /// Reference to an output parameter by index.
    /// </summary>
    public KernelPin Out(int index) => new(Id, index);
}
