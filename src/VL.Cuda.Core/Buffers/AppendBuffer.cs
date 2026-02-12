using System;
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Device;

namespace VL.Cuda.Core.Buffers;

/// <summary>
/// Type-erased interface for AppendBuffer tracking in CudaEngine.
/// Allows the engine to manage append buffers without knowing the element type.
/// </summary>
public interface IAppendBuffer : IDisposable
{
    CUdeviceptr DataPointer { get; }
    CUdeviceptr CounterPointer { get; }
    int MaxCapacity { get; }

    /// <summary>
    /// Read the counter value from GPU. Returns min(rawCount, MaxCapacity).
    /// </summary>
    int ReadCount();

    /// <summary>
    /// The last count returned by ReadCount(), clamped to MaxCapacity.
    /// </summary>
    int LastReadCount { get; }

    /// <summary>
    /// The last raw counter value from GPU (may exceed MaxCapacity if overflow).
    /// </summary>
    int LastRawCount { get; }

    /// <summary>
    /// True if the last raw count exceeded MaxCapacity.
    /// </summary>
    bool DidOverflow { get; }
}

/// <summary>
/// GPU buffer with an atomic counter for variable-length output.
/// Wraps GpuBuffer&lt;T&gt; Data + GpuBuffer&lt;uint&gt; Counter (1 element).
/// Use cases: particle emission, geometry generation, stream compaction, filtering.
/// Kernels use atomicAdd on the counter to claim write positions.
/// </summary>
public sealed class AppendBuffer<T> : IAppendBuffer where T : unmanaged
{
    private readonly Action<AppendBuffer<T>>? _onDispose;
    private bool _disposed;

    public GpuBuffer<T> Data { get; }
    public GpuBuffer<uint> Counter { get; }
    public int MaxCapacity { get; }
    public BufferLifetime Lifetime { get; }
    public DType ElementType => Data.ElementType;

    public CUdeviceptr DataPointer => Data.Pointer;
    public CUdeviceptr CounterPointer => Counter.Pointer;

    public int LastReadCount { get; private set; }
    public int LastRawCount { get; private set; }
    public bool DidOverflow => LastRawCount > MaxCapacity;

    internal AppendBuffer(GpuBuffer<T> data, GpuBuffer<uint> counter, int maxCapacity,
        BufferLifetime lifetime, Action<AppendBuffer<T>>? onDispose)
    {
        Data = data ?? throw new ArgumentNullException(nameof(data));
        Counter = counter ?? throw new ArgumentNullException(nameof(counter));
        MaxCapacity = maxCapacity;
        Lifetime = lifetime;
        _onDispose = onDispose;
    }

    /// <summary>
    /// Read counter from GPU, store raw and clamped values.
    /// Returns the clamped count (valid element count).
    /// </summary>
    public int ReadCount()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var counterData = Counter.Download();
        int raw = (int)counterData[0];
        LastRawCount = raw;
        LastReadCount = Math.Min(raw, MaxCapacity);
        return LastReadCount;
    }

    /// <summary>
    /// Set the read count directly (called by CudaEngine after auto-readback).
    /// </summary>
    internal void SetReadCount(int rawCount)
    {
        LastRawCount = rawCount;
        LastReadCount = Math.Min(rawCount, MaxCapacity);
    }

    /// <summary>
    /// Download only the valid elements (first LastReadCount items).
    /// </summary>
    public T[] DownloadValid()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (LastReadCount == 0)
            return Array.Empty<T>();

        // Download full buffer then slice
        var all = Data.Download();
        if (LastReadCount >= all.Length)
            return all;

        var result = new T[LastReadCount];
        Array.Copy(all, result, LastReadCount);
        return result;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_onDispose != null)
        {
            _onDispose(this);
        }
        else
        {
            Data.Dispose();
            Counter.Dispose();
        }
    }
}
