using System;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Device;

namespace VL.Cuda.Core.Buffers;

/// <summary>
/// Typed GPU memory wrapper. Owns a device allocation and provides upload/download.
/// </summary>
public sealed class GpuBuffer<T> : IDisposable where T : unmanaged
{
    private readonly DeviceContext _device;
    private readonly Action<GpuBuffer<T>>? _onDispose;
    private bool _disposed;

    public CUdeviceptr Pointer { get; }
    public int ElementCount { get; }
    public long SizeInBytes { get; }
    public BufferShape Shape { get; }
    public BufferLifetime Lifetime { get; }
    public BufferState State { get; internal set; }
    public DType ElementType { get; }

    internal GpuBuffer(DeviceContext device, CUdeviceptr pointer, int elementCount,
        BufferShape shape, BufferLifetime lifetime, Action<GpuBuffer<T>>? onDispose)
    {
        _device = device;
        _onDispose = onDispose;
        Pointer = pointer;
        ElementCount = elementCount;
        SizeInBytes = (long)elementCount * Marshal.SizeOf<T>();
        Shape = shape;
        Lifetime = lifetime;
        State = BufferState.Uninitialized;
        ElementType = DTypeExtensions.FromClrType<T>();
    }

    /// <summary>
    /// Allocates a new GPU buffer directly (not from pool).
    /// </summary>
    public static GpuBuffer<T> Allocate(DeviceContext device, int elementCount,
        BufferLifetime lifetime = BufferLifetime.External)
    {
        long bytes = (long)elementCount * Marshal.SizeOf<T>();
        var ptr = device.Context.AllocateMemory((SizeT)(ulong)bytes);
        var shape = BufferShape.D1(elementCount);
        return new GpuBuffer<T>(device, ptr, elementCount, shape, lifetime, onDispose: null);
    }

    /// <summary>
    /// Upload data from host array to GPU.
    /// </summary>
    public void Upload(T[] source)
    {
        if (source.Length != ElementCount)
            throw new ArgumentException($"Source length {source.Length} != buffer element count {ElementCount}");
        ObjectDisposedException.ThrowIf(_disposed, this);

        _device.Context.CopyToDevice(Pointer, source);
        State = BufferState.Valid;
    }

    /// <summary>
    /// Download data from GPU to host array.
    /// </summary>
    public void Download(T[] destination)
    {
        if (destination.Length != ElementCount)
            throw new ArgumentException($"Destination length {destination.Length} != buffer element count {ElementCount}");
        ObjectDisposedException.ThrowIf(_disposed, this);

        _device.Context.CopyToHost(destination, Pointer);
    }

    /// <summary>
    /// Download data from GPU to a new host array.
    /// </summary>
    public T[] Download()
    {
        var result = new T[ElementCount];
        Download(result);
        return result;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        State = BufferState.Released;

        if (_onDispose != null)
        {
            _onDispose(this);
        }
        else
        {
            _device.Context.FreeMemory(Pointer);
        }
    }
}
