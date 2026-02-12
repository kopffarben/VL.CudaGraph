using System;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Device;
using Xunit;

namespace VL.Cuda.Tests.Buffers;

public class AppendBufferTests : IDisposable
{
    private readonly DeviceContext _device = new(0);

    [Fact]
    public void Create_HasCorrectProperties()
    {
        using var pool = new BufferPool(_device);
        using var ab = pool.AcquireAppend<float>(1024);

        Assert.Equal(1024, ab.MaxCapacity);
        Assert.Equal(BufferLifetime.Graph, ab.Lifetime);
        Assert.Equal(DType.F32, ab.ElementType);
        Assert.NotEqual(default, ab.DataPointer);
        Assert.NotEqual(default, ab.CounterPointer);
        Assert.NotEqual(ab.DataPointer, ab.CounterPointer);
    }

    [Fact]
    public void ReadCount_AfterManualUpload()
    {
        using var pool = new BufferPool(_device);
        using var ab = pool.AcquireAppend<float>(256);

        // Manually set counter to 42
        ab.Counter.Upload(new uint[] { 42 });

        var count = ab.ReadCount();
        Assert.Equal(42, count);
        Assert.Equal(42, ab.LastReadCount);
        Assert.Equal(42, ab.LastRawCount);
        Assert.False(ab.DidOverflow);
    }

    [Fact]
    public void DidOverflow_Detection()
    {
        using var pool = new BufferPool(_device);
        using var ab = pool.AcquireAppend<float>(100);

        // Set counter beyond capacity
        ab.Counter.Upload(new uint[] { 150 });

        var count = ab.ReadCount();
        Assert.Equal(100, count); // clamped
        Assert.Equal(150, ab.LastRawCount);
        Assert.True(ab.DidOverflow);
    }

    [Fact]
    public void DownloadValid_CorrectSlice()
    {
        using var pool = new BufferPool(_device);
        using var ab = pool.AcquireAppend<float>(256);

        // Upload some data
        var data = new float[256];
        for (int i = 0; i < 256; i++) data[i] = i * 1.5f;
        ab.Data.Upload(data);

        // Set counter to 10
        ab.Counter.Upload(new uint[] { 10 });
        ab.ReadCount();

        var result = ab.DownloadValid();
        Assert.Equal(10, result.Length);
        for (int i = 0; i < 10; i++)
            Assert.Equal(data[i], result[i]);
    }

    [Fact]
    public void DownloadValid_ZeroCount_ReturnsEmpty()
    {
        using var pool = new BufferPool(_device);
        using var ab = pool.AcquireAppend<float>(256);

        ab.Counter.Upload(new uint[] { 0 });
        ab.ReadCount();

        var result = ab.DownloadValid();
        Assert.Empty(result);
    }

    [Fact]
    public void Dispose_ReleasesToPool()
    {
        using var pool = new BufferPool(_device);

        var ab1 = pool.AcquireAppend<float>(256);
        var dataPtr = ab1.DataPointer;
        var counterPtr = ab1.CounterPointer;
        ab1.Dispose();

        // Reacquire — pool should reuse the allocations
        var ab2 = pool.AcquireAppend<float>(256);
        // Same bucket size → could reuse pointers (but order may vary)
        // Just verify we can acquire without error
        Assert.NotEqual(default, ab2.DataPointer);
        Assert.NotEqual(default, ab2.CounterPointer);
        ab2.Dispose();
    }

    [Fact]
    public void DoubleDispose_NoThrow()
    {
        using var pool = new BufferPool(_device);
        var ab = pool.AcquireAppend<float>(128);

        ab.Dispose();
        // Second dispose should not throw
        ab.Dispose();
    }

    [Fact]
    public void SetReadCount_Internal()
    {
        using var pool = new BufferPool(_device);
        using var ab = pool.AcquireAppend<float>(100);

        // Use internal SetReadCount (simulating what CudaEngine does)
        // We can't call SetReadCount directly from test (internal), but ReadCount works.
        ab.Counter.Upload(new uint[] { 75 });
        ab.ReadCount();

        Assert.Equal(75, ab.LastReadCount);
        Assert.Equal(75, ab.LastRawCount);
        Assert.False(ab.DidOverflow);
    }

    public void Dispose() => _device.Dispose();
}
