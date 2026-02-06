using System;
using System.Collections.Generic;
using VL.Cuda.Core.Device;

namespace VL.Cuda.Core.PTX;

/// <summary>
/// Caches loaded PTX modules by file path. Avoids redundant JIT compilation.
/// </summary>
public sealed class ModuleCache : IDisposable
{
    private readonly DeviceContext _device;
    private readonly Dictionary<string, LoadedKernel> _cache = new(StringComparer.OrdinalIgnoreCase);
    private bool _disposed;

    public ModuleCache(DeviceContext device)
    {
        _device = device;
    }

    /// <summary>
    /// Get or load a kernel from PTX path. Cached on normalized full path.
    /// </summary>
    public LoadedKernel GetOrLoad(string ptxPath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var key = System.IO.Path.GetFullPath(ptxPath);
        if (_cache.TryGetValue(key, out var cached))
            return cached;

        var loaded = PtxLoader.Load(_device, ptxPath);
        _cache[key] = loaded;
        return loaded;
    }

    /// <summary>
    /// Evict a specific entry from the cache.
    /// </summary>
    public bool Evict(string ptxPath)
    {
        var key = System.IO.Path.GetFullPath(ptxPath);
        if (_cache.Remove(key, out var removed))
        {
            removed.Dispose();
            return true;
        }
        return false;
    }

    /// <summary>
    /// Number of cached modules.
    /// </summary>
    public int Count => _cache.Count;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var entry in _cache.Values)
            entry.Dispose();
        _cache.Clear();
    }
}
