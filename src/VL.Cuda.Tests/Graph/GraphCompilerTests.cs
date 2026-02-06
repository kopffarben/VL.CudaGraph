using System;
using System.IO;
using ManagedCuda;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.PTX;
using VL.Cuda.Core.Graph;
using Xunit;

namespace VL.Cuda.Tests.Graph;

public class GraphCompilerTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly ModuleCache _cache;
    private readonly BufferPool _pool;

    private static string VectorAddPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");
    private static string ScalarMulPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "scalar_mul.ptx");

    public GraphCompilerTests()
    {
        _cache = new ModuleCache(_device);
        _pool = new BufferPool(_device);
    }

    [Fact]
    public void Compile_SingleNode_Succeeds()
    {
        const int N = 256;
        var builder = new GraphBuilder(_device, _cache);
        var add = builder.AddKernel(VectorAddPath, "add");

        int blockSize = 256;
        add.GridDimX = (uint)((N + blockSize - 1) / blockSize);

        // Set external buffers
        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);

        builder.SetExternalBuffer(add, 0, bufA.Pointer);
        builder.SetExternalBuffer(add, 1, bufB.Pointer);
        builder.SetExternalBuffer(add, 2, bufC.Pointer);
        add.SetScalar(3, (uint)N);

        var compiler = new GraphCompiler(_device, _pool);
        using var compiled = compiler.Compile(builder);

        Assert.NotNull(compiled);
        Assert.NotNull(compiled.Exec);
    }

    [Fact]
    public void Compile_InvalidGraph_Throws()
    {
        var builder = new GraphBuilder(_device, _cache); // empty graph
        var compiler = new GraphCompiler(_device, _pool);

        Assert.Throws<InvalidOperationException>(() => compiler.Compile(builder));
    }

    [Fact]
    public void Compile_TwoNodeChain_Succeeds()
    {
        const int N = 256;
        var builder = new GraphBuilder(_device, _cache);

        var add = builder.AddKernel(VectorAddPath, "add");
        var mul = builder.AddKernel(ScalarMulPath, "mul");

        int blockSize = 256;
        uint gridSize = (uint)((N + blockSize - 1) / blockSize);
        add.GridDimX = gridSize;
        mul.GridDimX = gridSize;

        // External buffers
        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufD = GpuBuffer<float>.Allocate(_device, N); // final output

        builder.SetExternalBuffer(add, 0, bufA.Pointer); // A input
        builder.SetExternalBuffer(add, 1, bufB.Pointer); // B input
        builder.SetExternalBuffer(mul, 1, bufD.Pointer);  // C output (final)
        add.SetScalar(3, (uint)N);
        mul.SetScalar(2, 2.0f); // scale
        mul.SetScalar(3, (uint)N);

        // Edge: add.C (index 2) → mul.A (index 0)
        builder.AddEdge(add, 2, mul, 0);

        var compiler = new GraphCompiler(_device, _pool);
        using var compiled = compiler.Compile(builder);

        Assert.NotNull(compiled);
    }

    [Fact]
    public void Compile_CyclicGraph_Throws()
    {
        var builder = new GraphBuilder(_device, _cache);
        var a = builder.AddKernel(VectorAddPath, "A");
        var b = builder.AddKernel(ScalarMulPath, "B");

        builder.AddEdge(a, 2, b, 0);
        builder.AddEdge(b, 1, a, 0);

        var compiler = new GraphCompiler(_device, _pool);
        Assert.Throws<InvalidOperationException>(() => compiler.Compile(builder));
    }

    public void Dispose()
    {
        _pool.Dispose();
        _cache.Dispose();
        _device.Dispose();
    }
}
