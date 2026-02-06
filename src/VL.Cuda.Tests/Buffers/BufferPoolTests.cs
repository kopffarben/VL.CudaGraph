using System;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Device;
using Xunit;

namespace VL.Cuda.Tests.Buffers;

public class BufferPoolTests : IDisposable
{
    private readonly DeviceContext _device = new(0);

    [Fact]
    public void Acquire_ReturnsBuffer()
    {
        using var pool = new BufferPool(_device);
        using var buf = pool.Acquire<float>(256);

        Assert.Equal(256, buf.ElementCount);
        Assert.Equal(BufferLifetime.Graph, buf.Lifetime);
    }

    [Fact]
    public void Release_And_Reacquire_ReusesSamePointer()
    {
        using var pool = new BufferPool(_device);

        var buf1 = pool.Acquire<float>(256);
        var ptr1 = buf1.Pointer;
        buf1.Dispose(); // returns to pool

        var buf2 = pool.Acquire<float>(256);
        var ptr2 = buf2.Pointer;
        buf2.Dispose();

        // Same bucket size → same allocation reused
        Assert.Equal(ptr1, ptr2);
    }

    [Fact]
    public void DifferentSizes_DifferentBuckets()
    {
        using var pool = new BufferPool(_device);

        using var small = pool.Acquire<float>(64);   // 256 bytes → bucket 256
        using var large = pool.Acquire<float>(1024);  // 4096 bytes → bucket 4096

        Assert.NotEqual(small.Pointer, large.Pointer);
    }

    [Fact]
    public void AcquiredBuffer_UploadDownload_Works()
    {
        using var pool = new BufferPool(_device);
        using var buf = pool.Acquire<float>(128);

        var data = new float[128];
        for (int i = 0; i < 128; i++) data[i] = i;

        buf.Upload(data);
        var result = buf.Download();

        for (int i = 0; i < 128; i++)
            Assert.Equal(data[i], result[i]);
    }

    [Theory]
    [InlineData(1, 256)]
    [InlineData(100, 256)]
    [InlineData(256, 256)]
    [InlineData(257, 512)]
    [InlineData(512, 512)]
    [InlineData(1023, 1024)]
    [InlineData(1024, 1024)]
    [InlineData(1025, 2048)]
    public void RoundUpToPowerOf2_Works(long input, long expected)
    {
        Assert.Equal(expected, BufferPool.RoundUpToPowerOf2(input));
    }

    public void Dispose() => _device.Dispose();
}
