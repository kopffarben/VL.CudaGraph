using System;
using System.IO;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.PTX;
using Xunit;

namespace VL.Cuda.Tests.PTX;

public class PtxLoaderTests : IDisposable
{
    private readonly DeviceContext _device = new(0);

    private static string TestKernelPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");
    private static string TestMetadataPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.json");

    [Fact]
    public void ParseMetadata_ReturnsDescriptor()
    {
        var descriptor = PtxMetadata.Parse(TestMetadataPath);

        Assert.Equal("vector_add", descriptor.EntryPoint);
        Assert.Equal(256, descriptor.BlockSize);
        Assert.Equal(4, descriptor.Parameters.Count);
        Assert.Equal("A", descriptor.Parameters[0].Name);
        Assert.True(descriptor.Parameters[0].IsPointer);
        Assert.Equal(ParamDirection.In, descriptor.Parameters[0].Direction);
        Assert.Equal("N", descriptor.Parameters[3].Name);
        Assert.False(descriptor.Parameters[3].IsPointer);
    }

    [Fact]
    public void LoadPtx_Succeeds()
    {
        using var loaded = PtxLoader.Load(_device, TestKernelPath);

        Assert.NotNull(loaded.Kernel);
        Assert.Equal("vector_add", loaded.Descriptor.EntryPoint);
    }

    [Fact]
    public void LoadPtx_MissingFile_Throws()
    {
        Assert.Throws<FileNotFoundException>(() => PtxLoader.Load(_device, "nonexistent.ptx"));
    }

    public void Dispose() => _device.Dispose();
}
