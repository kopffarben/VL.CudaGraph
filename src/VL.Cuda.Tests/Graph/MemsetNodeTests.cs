using System;
using System.IO;
using System.Linq;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.Graph;
using VL.Cuda.Core.PTX;
using Xunit;

namespace VL.Cuda.Tests.Graph;

public class MemsetNodeTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly ModuleCache _cache;

    private static string VectorAddPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");

    public MemsetNodeTests()
    {
        _cache = new ModuleCache(_device);
    }

    [Fact]
    public void AddMemset_TracksDescriptor()
    {
        var builder = new GraphBuilder(_device, _cache);

        using var buf = GpuBuffer<uint>.Allocate(_device, 1);
        var memset = builder.AddMemset(buf.Pointer, 0, 4, 1, "TestReset");

        Assert.Single(builder.MemsetDescriptors);
        Assert.Equal(buf.Pointer, memset.Destination);
        Assert.Equal(0u, memset.Value);
        Assert.Equal(4u, memset.ElementSize);
        Assert.Equal(1ul, memset.Width);
        Assert.Equal("TestReset", memset.DebugName);
    }

    [Fact]
    public void AddMemsetDependency_WiresCorrectly()
    {
        var builder = new GraphBuilder(_device, _cache);

        var kernel = builder.AddKernel(VectorAddPath, "add");
        using var buf = GpuBuffer<uint>.Allocate(_device, 1);
        var memset = builder.AddMemset(buf.Pointer, 0, 4, 1, "Reset");

        builder.AddMemsetDependency(memset, kernel);

        Assert.Contains(kernel.Id, memset.DependentKernelNodeIds);
    }

    [Fact]
    public void MemsetDescriptors_PassedToGraphDescription()
    {
        var builder = new GraphBuilder(_device, _cache);

        using var buf = GpuBuffer<uint>.Allocate(_device, 1);
        builder.AddMemset(buf.Pointer, 0, 4, 1, "Reset");

        var desc = builder.Build();

        Assert.Single(desc.MemsetDescriptors);
    }

    [Fact]
    public void MemsetNode_InsertedInGraph()
    {
        // Build a graph with a kernel + memset and compile it
        using var pool = new BufferPool(_device);
        var builder = new GraphBuilder(_device, _cache);

        const int N = 256;
        var kernel = builder.AddKernel(VectorAddPath, "add");
        kernel.GridDimX = 1;
        kernel.BlockDimX = 256;

        // External buffers
        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);
        using var counter = GpuBuffer<uint>.Allocate(_device, 1);

        builder.SetExternalBuffer(kernel, 0, bufA.Pointer);
        builder.SetExternalBuffer(kernel, 1, bufB.Pointer);
        builder.SetExternalBuffer(kernel, 2, bufC.Pointer);
        kernel.SetScalar(3, (uint)N);

        // Add memset node with dependency
        var memset = builder.AddMemset(counter.Pointer, 0, 4, 1, "Reset counter");
        builder.AddMemsetDependency(memset, kernel);

        // Compile — should not throw
        var compiler = new GraphCompiler(_device, pool);
        using var compiled = compiler.Compile(builder);

        Assert.NotNull(compiled);
        Assert.Single(compiled.MemsetNodeHandles);
    }

    [Fact]
    public void MemsetNode_ResetsCounter_OnLaunch()
    {
        // Verify that memset actually resets the counter to 0
        using var pool = new BufferPool(_device);
        var builder = new GraphBuilder(_device, _cache);

        const int N = 256;
        var kernel = builder.AddKernel(VectorAddPath, "add");
        kernel.GridDimX = 1;
        kernel.BlockDimX = 256;

        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);
        using var counter = GpuBuffer<uint>.Allocate(_device, 1);

        // Set counter to non-zero
        counter.Upload(new uint[] { 999 });

        builder.SetExternalBuffer(kernel, 0, bufA.Pointer);
        builder.SetExternalBuffer(kernel, 1, bufB.Pointer);
        builder.SetExternalBuffer(kernel, 2, bufC.Pointer);
        kernel.SetScalar(3, (uint)N);

        var memset = builder.AddMemset(counter.Pointer, 0, 4, 1, "Reset counter");
        builder.AddMemsetDependency(memset, kernel);

        var compiler = new GraphCompiler(_device, pool);
        using var compiled = compiler.Compile(builder);

        // Launch
        using var stream = new ManagedCuda.CudaStream();
        compiled.LaunchAndSync(stream);

        // Counter should be 0 (reset by memset node)
        var result = counter.Download();
        Assert.Equal(0u, result[0]);
    }

    public void Dispose()
    {
        _cache.Dispose();
        _device.Dispose();
    }
}
