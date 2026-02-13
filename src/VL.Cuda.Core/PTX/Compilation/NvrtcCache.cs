using System;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using ManagedCuda.NVRTC;
using VL.Cuda.Core.Device;

namespace VL.Cuda.Core.PTX.Compilation;

/// <summary>
/// Compiles CUDA C++ source to PTX via NVRTC and loads as CUDA modules.
/// Caches compiled kernels by source hash. ~100ms-2s per compilation.
/// </summary>
internal sealed class NvrtcCache : IDisposable
{
    private readonly DeviceContext _device;
    private readonly ConcurrentDictionary<string, LoadedKernel> _cache = new();
    private bool _disposed;

    public NvrtcCache(DeviceContext device)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));
    }

    /// <summary>
    /// Compile CUDA C++ source to PTX and load as a CUDA module. Cached by source hash.
    /// </summary>
    public LoadedKernel GetOrCompile(string cudaSource, string entryPoint, KernelDescriptor descriptor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var key = ComputeSourceKey(cudaSource);
        if (_cache.TryGetValue(key, out var cached))
            return cached;

        var ptxBytes = CompileToBytes(cudaSource, entryPoint);
        var loaded = PtxLoader.LoadFromBytes(_device, ptxBytes, descriptor);
        _cache[key] = loaded;
        return loaded;
    }

    /// <summary>
    /// Compile CUDA C++ source to PTX bytes without loading.
    /// </summary>
    internal byte[] CompileToBytes(string cudaSource, string entryPoint)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var sm = $"compute_{_device.ComputeCapabilityMajor}{_device.ComputeCapabilityMinor}";

        using var compiler = new CudaRuntimeCompiler(cudaSource, entryPoint);
        try
        {
            compiler.Compile(new[] { $"--gpu-architecture={sm}" });
        }
        catch (NVRTCException ex)
        {
            string log;
            try { log = compiler.GetLogAsString(); }
            catch { log = "(could not retrieve compile log)"; }

            throw new InvalidOperationException(
                $"NVRTC compilation failed for '{entryPoint}': {ex.NVRTCError}\n{log}", ex);
        }

        return compiler.GetPTX();
    }

    /// <summary>
    /// Remove a cached entry by source hash.
    /// </summary>
    public bool Invalidate(string sourceHash)
    {
        if (_cache.TryRemove(sourceHash, out var removed))
        {
            removed.Dispose();
            return true;
        }
        return false;
    }

    public int CacheCount => _cache.Count;

    internal static string ComputeSourceKey(string source)
    {
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(source));
        return Convert.ToHexString(hash);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var entry in _cache.Values)
            entry.Dispose();
        _cache.Clear();
    }
}
