using System;
using System.IO;
using System.Reflection;

namespace VL.Cuda.Core.PTX;

/// <summary>
/// Discriminated union describing how a kernel's PTX was produced.
/// Three variants: FilesystemPtx, IlgpuMethod, NvrtcSource.
/// </summary>
public abstract class KernelSource
{
    /// <summary>
    /// Returns a stable cache key for deduplication. Two KernelSources with the
    /// same cache key produce identical PTX.
    /// </summary>
    public abstract string GetCacheKey();

    /// <summary>
    /// Human-readable name for diagnostics.
    /// </summary>
    public abstract string GetDebugName();

    /// <summary>
    /// Kernel loaded from a filesystem PTX + JSON pair (Triton, nvcc, hand-written).
    /// </summary>
    public sealed class FilesystemPtx : KernelSource
    {
        public string PtxPath { get; }

        public FilesystemPtx(string ptxPath)
        {
            PtxPath = ptxPath ?? throw new ArgumentNullException(nameof(ptxPath));
        }

        public override string GetCacheKey() => $"file:{Path.GetFullPath(PtxPath)}";
        public override string GetDebugName() => Path.GetFileNameWithoutExtension(PtxPath);
    }

    /// <summary>
    /// Kernel compiled from a C# method via ILGPU IR -> PTX.
    /// </summary>
    public sealed class IlgpuMethod : KernelSource
    {
        public string MethodHash { get; }
        public MethodInfo KernelMethod { get; }

        public IlgpuMethod(string methodHash, MethodInfo kernelMethod)
        {
            MethodHash = methodHash ?? throw new ArgumentNullException(nameof(methodHash));
            KernelMethod = kernelMethod ?? throw new ArgumentNullException(nameof(kernelMethod));
        }

        public override string GetCacheKey() => $"ilgpu:{MethodHash}";
        public override string GetDebugName() => $"ILGPU:{KernelMethod.DeclaringType?.Name}.{KernelMethod.Name}";
    }

    /// <summary>
    /// Kernel compiled from CUDA C++ source via NVRTC.
    /// </summary>
    public sealed class NvrtcSource : KernelSource
    {
        public string SourceHash { get; }
        public string CudaSource { get; }
        public string EntryPoint { get; }

        public NvrtcSource(string sourceHash, string cudaSource, string entryPoint)
        {
            SourceHash = sourceHash ?? throw new ArgumentNullException(nameof(sourceHash));
            CudaSource = cudaSource ?? throw new ArgumentNullException(nameof(cudaSource));
            EntryPoint = entryPoint ?? throw new ArgumentNullException(nameof(entryPoint));
        }

        public override string GetCacheKey() => $"nvrtc:{SourceHash}";
        public override string GetDebugName() => $"NVRTC:{EntryPoint}";
    }
}
