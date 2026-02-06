using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace VL.Cuda.Core.Device;

/// <summary>
/// Wraps a CUDA device and its primary context.
/// Phase 0: thin wrapper around ManagedCuda.CudaContext.
/// Phase 2 introduces the full CudaContext facade with BlockRegistry, DirtyTracker, etc.
/// </summary>
public sealed class DeviceContext : IDisposable
{
    private readonly ManagedCuda.CudaContext _context;
    private bool _disposed;

    public int DeviceId { get; }
    public CudaDeviceProperties DeviceProperties { get; }
    public CUcontext CUContext => _context.Context;

    /// <summary>
    /// The underlying ManagedCuda context. Used internally by GpuBuffer, PtxLoader, etc.
    /// </summary>
    internal ManagedCuda.CudaContext Context => _context;

    public DeviceContext(int deviceId = 0)
    {
        DeviceId = deviceId;
        _context = new ManagedCuda.CudaContext(deviceId);
        DeviceProperties = _context.GetDeviceInfo();
    }

    public string DeviceName => DeviceProperties.DeviceName;
    public long TotalMemoryBytes => (long)(ulong)DeviceProperties.TotalGlobalMemory;
    public int ComputeCapabilityMajor => DeviceProperties.ComputeCapability.Major;
    public int ComputeCapabilityMinor => DeviceProperties.ComputeCapability.Minor;
    public int MultiProcessorCount => DeviceProperties.MultiProcessorCount;

    public void Synchronize() => _context.Synchronize();

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _context.Dispose();
    }
}
