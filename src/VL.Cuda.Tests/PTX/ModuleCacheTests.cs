using System;
using System.IO;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.PTX;
using Xunit;

namespace VL.Cuda.Tests.PTX;

public class ModuleCacheTests : IDisposable
{
    private readonly DeviceContext _device = new(0);

    private static string TestKernelPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");

    [Fact]
    public void GetOrLoad_FirstCall_LoadsModule()
    {
        using var cache = new ModuleCache(_device);
        var loaded = cache.GetOrLoad(TestKernelPath);

        Assert.NotNull(loaded);
        Assert.Equal("vector_add", loaded.Descriptor.EntryPoint);
        Assert.Equal(1, cache.Count);
    }

    [Fact]
    public void GetOrLoad_SecondCall_ReturnsCached()
    {
        using var cache = new ModuleCache(_device);
        var first = cache.GetOrLoad(TestKernelPath);
        var second = cache.GetOrLoad(TestKernelPath);

        Assert.Same(first, second);
        Assert.Equal(1, cache.Count);
    }

    [Fact]
    public void Evict_RemovesEntry()
    {
        using var cache = new ModuleCache(_device);
        cache.GetOrLoad(TestKernelPath);
        Assert.Equal(1, cache.Count);

        var evicted = cache.Evict(TestKernelPath);
        Assert.True(evicted);
        Assert.Equal(0, cache.Count);
    }

    [Fact]
    public void Evict_NonExistent_ReturnsFalse()
    {
        using var cache = new ModuleCache(_device);
        Assert.False(cache.Evict("nonexistent.ptx"));
    }

    public void Dispose() => _device.Dispose();
}
