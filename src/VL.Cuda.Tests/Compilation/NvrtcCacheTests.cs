using System;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.PTX;
using VL.Cuda.Core.PTX.Compilation;
using VL.Cuda.Tests.Helpers;
using Xunit;

namespace VL.Cuda.Tests.Compilation;

public class NvrtcCacheTests : IDisposable
{
    private readonly DeviceContext _device = new(0);

    private const string SimpleCudaSource = @"
extern ""C"" __global__ void scale_kernel(float* data, float* output, float scale, unsigned int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        output[idx] = data[idx] * scale;
}
";

    private const string AddCudaSource = @"
extern ""C"" __global__ void add_kernel(float* a, float* b, float* c, unsigned int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}
";

    private const string InvalidCudaSource = @"
extern ""C"" __global__ void bad_kernel(
{
    // This is intentionally invalid CUDA C++
}
";

    private static KernelDescriptor MakeDescriptor(string entryPoint, params (string Name, string Type, bool IsPointer)[] parameters)
    {
        var paramDescs = new System.Collections.Generic.List<KernelParamDescriptor>();
        for (int idx = 0; idx < parameters.Length; idx++)
        {
            var (name, type, isPointer) = parameters[idx];
            paramDescs.Add(new KernelParamDescriptor
            {
                Name = name,
                Type = type,
                IsPointer = isPointer,
                Direction = isPointer ? ParamDirection.InOut : ParamDirection.In,
                Index = idx,
            });
        }
        return new KernelDescriptor
        {
            EntryPoint = entryPoint,
            Parameters = paramDescs,
            BlockSize = 256,
        };
    }

    [Fact(Skip = "NVRTC DLL (nvrtc64_130_0) requires CUDA Toolkit 13.0+ installed")]
    public void CompileToBytes_ProducesPTX()
    {
        using var cache = new NvrtcCache(_device);
        var ptxBytes = cache.CompileToBytes(SimpleCudaSource, "scale_kernel");

        Assert.NotNull(ptxBytes);
        Assert.True(ptxBytes.Length > 0);
    }

    [Fact(Skip = "NVRTC DLL (nvrtc64_130_0) requires CUDA Toolkit 13.0+ installed")]
    public void GetOrCompile_LoadsModule()
    {
        using var cache = new NvrtcCache(_device);

        var descriptor = MakeDescriptor("scale_kernel",
            ("data", "float*", true), ("output", "float*", true),
            ("scale", "float", false), ("N", "uint", false));

        var loaded = cache.GetOrCompile(SimpleCudaSource, "scale_kernel", descriptor);

        Assert.NotNull(loaded);
        Assert.Equal("scale_kernel", loaded.Descriptor.EntryPoint);
        Assert.Equal(4, loaded.Descriptor.Parameters.Count);
    }

    [Fact(Skip = "NVRTC DLL (nvrtc64_130_0) requires CUDA Toolkit 13.0+ installed")]
    public void GetOrCompile_CacheHit_ReturnsSameInstance()
    {
        using var cache = new NvrtcCache(_device);

        var descriptor = MakeDescriptor("scale_kernel",
            ("data", "float*", true), ("output", "float*", true),
            ("scale", "float", false), ("N", "uint", false));

        var first = cache.GetOrCompile(SimpleCudaSource, "scale_kernel", descriptor);
        var second = cache.GetOrCompile(SimpleCudaSource, "scale_kernel", descriptor);

        Assert.Same(first, second);
        Assert.Equal(1, cache.CacheCount);
    }

    [Fact(Skip = "NVRTC DLL (nvrtc64_130_0) requires CUDA Toolkit 13.0+ installed")]
    public void GetOrCompile_DifferentSources_DifferentCacheEntries()
    {
        using var cache = new NvrtcCache(_device);

        var descScale = MakeDescriptor("scale_kernel",
            ("data", "float*", true), ("output", "float*", true),
            ("scale", "float", false), ("N", "uint", false));
        var descAdd = MakeDescriptor("add_kernel",
            ("a", "float*", true), ("b", "float*", true),
            ("c", "float*", true), ("N", "uint", false));

        var scaleLoaded = cache.GetOrCompile(SimpleCudaSource, "scale_kernel", descScale);
        var addLoaded = cache.GetOrCompile(AddCudaSource, "add_kernel", descAdd);

        Assert.NotSame(scaleLoaded, addLoaded);
        Assert.Equal(2, cache.CacheCount);
    }

    [Fact(Skip = "NVRTC DLL (nvrtc64_130_0) requires CUDA Toolkit 13.0+ installed")]
    public void Invalidate_RemovesFromCache()
    {
        using var cache = new NvrtcCache(_device);

        var descriptor = MakeDescriptor("scale_kernel",
            ("data", "float*", true), ("output", "float*", true),
            ("scale", "float", false), ("N", "uint", false));

        cache.GetOrCompile(SimpleCudaSource, "scale_kernel", descriptor);
        Assert.Equal(1, cache.CacheCount);

        var hash = NvrtcCache.ComputeSourceKey(SimpleCudaSource);
        var invalidated = cache.Invalidate(hash);

        Assert.True(invalidated);
        Assert.Equal(0, cache.CacheCount);
    }

    [Fact(Skip = "NVRTC DLL (nvrtc64_130_0) requires CUDA Toolkit 13.0+ installed")]
    public void CompileError_ThrowsWithMessage()
    {
        using var cache = new NvrtcCache(_device);

        var descriptor = MakeDescriptor("bad_kernel");

        var ex = Assert.Throws<InvalidOperationException>(() =>
            cache.GetOrCompile(InvalidCudaSource, "bad_kernel", descriptor));

        Assert.Contains("NVRTC compilation failed", ex.Message);
        Assert.Contains("bad_kernel", ex.Message);
    }

    // These tests don't need the NVRTC DLL — they're pure hash computation
    [Fact]
    public void ComputeSourceKey_Deterministic()
    {
        var key1 = NvrtcCache.ComputeSourceKey(SimpleCudaSource);
        var key2 = NvrtcCache.ComputeSourceKey(SimpleCudaSource);

        Assert.Equal(key1, key2);
        Assert.Equal(64, key1.Length); // SHA256 = 32 bytes = 64 hex chars
    }

    [Fact]
    public void ComputeSourceKey_DifferentSources_DifferentKeys()
    {
        var key1 = NvrtcCache.ComputeSourceKey(SimpleCudaSource);
        var key2 = NvrtcCache.ComputeSourceKey(AddCudaSource);

        Assert.NotEqual(key1, key2);
    }

    public void Dispose()
    {
        _device.Dispose();
    }
}
