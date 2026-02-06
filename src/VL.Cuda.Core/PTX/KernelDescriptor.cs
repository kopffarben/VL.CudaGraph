using System.Collections.Generic;

namespace VL.Cuda.Core.PTX;

/// <summary>
/// Describes a CUDA kernel: entry point name, parameters, and grid configuration hints.
/// Parsed from the companion .json metadata file.
/// </summary>
public sealed class KernelDescriptor
{
    public required string EntryPoint { get; init; }
    public required IReadOnlyList<KernelParamDescriptor> Parameters { get; init; }

    /// <summary>Default block size hint (threads per block). 0 = not specified.</summary>
    public int BlockSize { get; init; }

    /// <summary>Shared memory size in bytes. 0 = not specified.</summary>
    public int SharedMemoryBytes { get; init; }
}
