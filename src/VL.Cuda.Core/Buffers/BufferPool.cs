using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Device;

namespace VL.Cuda.Core.Buffers;

/// <summary>
/// Power-of-2 bucket pool for GPU memory. Avoids repeated alloc/free.
/// Thread-safe via ConcurrentBag per bucket.
/// </summary>
public sealed class BufferPool : IDisposable
{
    private readonly DeviceContext _device;
    private readonly ConcurrentDictionary<long, ConcurrentBag<CUdeviceptr>> _buckets = new();
    private readonly ConcurrentBag<CUdeviceptr> _allAllocations = new();
    private bool _disposed;

    /// <summary>Minimum allocation size (256 bytes).</summary>
    private const long MinBucketSize = 256;

    /// <summary>Maximum bucket size (1 GB). Above this, exact allocation.</summary>
    private const long MaxBucketSize = 1L << 30;

    public BufferPool(DeviceContext device)
    {
        _device = device;
    }

    /// <summary>
    /// Acquire a buffer from the pool. May reuse a previously released buffer.
    /// </summary>
    public GpuBuffer<T> Acquire<T>(int elementCount, BufferLifetime lifetime = BufferLifetime.Graph)
        where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        long requestedBytes = (long)elementCount * Marshal.SizeOf<T>();
        long bucketSize = RoundUpToPowerOf2(requestedBytes);

        var bucket = _buckets.GetOrAdd(bucketSize, _ => new ConcurrentBag<CUdeviceptr>());

        CUdeviceptr ptr;
        if (bucket.TryTake(out var reused))
        {
            ptr = reused;
        }
        else
        {
            ptr = _device.Context.AllocateMemory((SizeT)(ulong)bucketSize);
            _allAllocations.Add(ptr);
        }

        var shape = BufferShape.D1(elementCount);
        return new GpuBuffer<T>(_device, ptr, elementCount, shape, lifetime, onDispose: ReleaseBuffer);
    }

    private void ReleaseBuffer<T>(GpuBuffer<T> buffer) where T : unmanaged
    {
        if (_disposed) return;

        long bucketSize = RoundUpToPowerOf2(buffer.SizeInBytes);
        var bucket = _buckets.GetOrAdd(bucketSize, _ => new ConcurrentBag<CUdeviceptr>());
        bucket.Add(buffer.Pointer);
    }

    internal static long RoundUpToPowerOf2(long size)
    {
        if (size <= MinBucketSize) return MinBucketSize;
        if (size > MaxBucketSize) return size; // exact allocation for very large buffers

        // Round up to next power of 2
        long v = size - 1;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        return v + 1;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Free all allocations (including those currently in buckets)
        foreach (var ptr in _allAllocations)
        {
            try { _device.Context.FreeMemory(ptr); } catch { /* best effort */ }
        }
        _buckets.Clear();
    }
}
