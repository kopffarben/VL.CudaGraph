using System;
using System.Reflection;
using VL.Cuda.Core.PTX;
using Xunit;

namespace VL.Cuda.Tests.PTX;

public class KernelSourceTests
{
    [Fact]
    public void FilesystemPtx_GetCacheKey_ContainsFullPath()
    {
        var source = new KernelSource.FilesystemPtx("TestKernels/vector_add.ptx");
        var key = source.GetCacheKey();

        Assert.StartsWith("file:", key);
        Assert.Contains("vector_add.ptx", key);
    }

    [Fact]
    public void FilesystemPtx_GetDebugName_ReturnsFileNameWithoutExtension()
    {
        var source = new KernelSource.FilesystemPtx("TestKernels/vector_add.ptx");
        Assert.Equal("vector_add", source.GetDebugName());
    }

    [Fact]
    public void FilesystemPtx_SamePath_SameCacheKey()
    {
        var a = new KernelSource.FilesystemPtx("test.ptx");
        var b = new KernelSource.FilesystemPtx("test.ptx");

        Assert.Equal(a.GetCacheKey(), b.GetCacheKey());
    }

    [Fact]
    public void FilesystemPtx_DifferentPath_DifferentCacheKey()
    {
        var a = new KernelSource.FilesystemPtx("a.ptx");
        var b = new KernelSource.FilesystemPtx("b.ptx");

        Assert.NotEqual(a.GetCacheKey(), b.GetCacheKey());
    }

    [Fact]
    public void IlgpuMethod_GetCacheKey_ContainsHash()
    {
        var method = typeof(KernelSourceTests).GetMethod(nameof(DummyKernel),
            BindingFlags.Static | BindingFlags.NonPublic)!;
        var source = new KernelSource.IlgpuMethod("ABC123", method);

        Assert.Equal("ilgpu:ABC123", source.GetCacheKey());
    }

    [Fact]
    public void IlgpuMethod_GetDebugName_ContainsTypeName()
    {
        var method = typeof(KernelSourceTests).GetMethod(nameof(DummyKernel),
            BindingFlags.Static | BindingFlags.NonPublic)!;
        var source = new KernelSource.IlgpuMethod("ABC123", method);

        Assert.Contains("KernelSourceTests", source.GetDebugName());
        Assert.Contains("DummyKernel", source.GetDebugName());
    }

    [Fact]
    public void NvrtcSource_GetCacheKey_ContainsHash()
    {
        var source = new KernelSource.NvrtcSource("DEADBEEF", "extern \"C\" ...", "my_kernel");
        Assert.Equal("nvrtc:DEADBEEF", source.GetCacheKey());
    }

    [Fact]
    public void NvrtcSource_GetDebugName_ContainsEntryPoint()
    {
        var source = new KernelSource.NvrtcSource("DEADBEEF", "extern \"C\" ...", "my_kernel");
        Assert.Contains("my_kernel", source.GetDebugName());
    }

    [Fact]
    public void DifferentSourceTypes_DifferentCacheKeyPrefixes()
    {
        var method = typeof(KernelSourceTests).GetMethod(nameof(DummyKernel),
            BindingFlags.Static | BindingFlags.NonPublic)!;

        var fs = new KernelSource.FilesystemPtx("test.ptx");
        var ilgpu = new KernelSource.IlgpuMethod("hash1", method);
        var nvrtc = new KernelSource.NvrtcSource("hash2", "source", "entry");

        Assert.StartsWith("file:", fs.GetCacheKey());
        Assert.StartsWith("ilgpu:", ilgpu.GetCacheKey());
        Assert.StartsWith("nvrtc:", nvrtc.GetCacheKey());
    }

    [Fact]
    public void NullArguments_Throw()
    {
        Assert.Throws<ArgumentNullException>(() => new KernelSource.FilesystemPtx(null!));
        Assert.Throws<ArgumentNullException>(() => new KernelSource.IlgpuMethod(null!, typeof(KernelSourceTests).GetMethods()[0]));
        Assert.Throws<ArgumentNullException>(() => new KernelSource.IlgpuMethod("hash", null!));
        Assert.Throws<ArgumentNullException>(() => new KernelSource.NvrtcSource(null!, "src", "entry"));
        Assert.Throws<ArgumentNullException>(() => new KernelSource.NvrtcSource("hash", null!, "entry"));
        Assert.Throws<ArgumentNullException>(() => new KernelSource.NvrtcSource("hash", "src", null!));
    }

    private static void DummyKernel() { }
}
