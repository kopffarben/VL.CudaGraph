using System;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Device;
using Xunit;

namespace VL.Cuda.Tests.Buffers;

public class GpuBufferTests : IDisposable
{
    private readonly DeviceContext _device = new(0);

    [Fact]
    public void Allocate_SetsProperties()
    {
        using var buf = GpuBuffer<float>.Allocate(_device, 1024);

        Assert.Equal(1024, buf.ElementCount);
        Assert.Equal(1024 * 4, buf.SizeInBytes);
        Assert.Equal(DType.F32, buf.ElementType);
        Assert.Equal(BufferState.Uninitialized, buf.State);
        Assert.Equal(BufferLifetime.External, buf.Lifetime);
    }

    [Fact]
    public void UploadDownload_RoundTrips()
    {
        const int count = 512;
        using var buf = GpuBuffer<float>.Allocate(_device, count);

        var source = new float[count];
        for (int i = 0; i < count; i++)
            source[i] = i * 1.5f;

        buf.Upload(source);
        Assert.Equal(BufferState.Valid, buf.State);

        var result = buf.Download();

        for (int i = 0; i < count; i++)
            Assert.Equal(source[i], result[i]);
    }

    [Fact]
    public void Upload_WrongSize_Throws()
    {
        using var buf = GpuBuffer<float>.Allocate(_device, 100);
        Assert.Throws<ArgumentException>(() => buf.Upload(new float[50]));
    }

    [Fact]
    public void Download_WrongSize_Throws()
    {
        using var buf = GpuBuffer<float>.Allocate(_device, 100);
        Assert.Throws<ArgumentException>(() => buf.Download(new float[50]));
    }

    [Fact]
    public void Dispose_SetsReleasedState()
    {
        var buf = GpuBuffer<float>.Allocate(_device, 64);
        buf.Dispose();
        Assert.Equal(BufferState.Released, buf.State);
    }

    [Fact]
    public void IntBuffer_RoundTrips()
    {
        const int count = 256;
        using var buf = GpuBuffer<int>.Allocate(_device, count);

        var source = new int[count];
        for (int i = 0; i < count; i++)
            source[i] = i * 3 - 100;

        buf.Upload(source);
        var result = buf.Download();

        for (int i = 0; i < count; i++)
            Assert.Equal(source[i], result[i]);
    }

    public void Dispose() => _device.Dispose();
}
