using System;
using VL.Cuda.Core.PTX;

namespace VL.Cuda.Core.Context.Services;

/// <summary>
/// Identifies a kernel whose source code changed: which block, which kernel handle, and the new source.
/// </summary>
public readonly record struct DirtyCodeEntry(Guid BlockId, Guid KernelHandleId, KernelSource NewSource);
