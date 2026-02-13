using System;
using VL.Cuda.Core.PTX;

namespace VL.Cuda.Core.Blocks.Builder;

/// <summary>
/// Represents a kernel added to a block via BlockBuilder.AddKernel().
/// Provides In()/Out() to reference specific parameters for port binding.
/// For ILGPU kernels, an index remap translates user-facing parameter indices
/// to expanded PTX indices (accounting for ArrayView length params).
/// </summary>
public sealed class KernelHandle
{
    private readonly int[]? _indexRemap;

    public Guid Id { get; }
    public KernelSource Source { get; }
    public KernelDescriptor Descriptor { get; }

    public uint GridDimX { get; set; } = 1;
    public uint GridDimY { get; set; } = 1;
    public uint GridDimZ { get; set; } = 1;

    /// <summary>
    /// Create a kernel handle from a filesystem PTX path (backward-compatible).
    /// </summary>
    public KernelHandle(string ptxPath, KernelDescriptor descriptor)
        : this(new KernelSource.FilesystemPtx(ptxPath), descriptor)
    {
    }

    /// <summary>
    /// Create a kernel handle from any kernel source.
    /// </summary>
    public KernelHandle(KernelSource source, KernelDescriptor descriptor, int[]? indexRemap = null)
    {
        Id = Guid.NewGuid();
        Source = source ?? throw new ArgumentNullException(nameof(source));
        Descriptor = descriptor ?? throw new ArgumentNullException(nameof(descriptor));
        _indexRemap = indexRemap;
    }

    /// <summary>
    /// Reference to an input parameter by index.
    /// For ILGPU kernels, the index is remapped to account for ArrayView length params.
    /// </summary>
    public KernelPin In(int index) => new(Id, _indexRemap != null ? _indexRemap[index] : index);

    /// <summary>
    /// Reference to an output parameter by index.
    /// For ILGPU kernels, the index is remapped to account for ArrayView length params.
    /// </summary>
    public KernelPin Out(int index) => new(Id, _indexRemap != null ? _indexRemap[index] : index);
}
