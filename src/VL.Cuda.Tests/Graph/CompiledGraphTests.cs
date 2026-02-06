using System;
using System.IO;
using ManagedCuda;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.PTX;
using VL.Cuda.Core.Graph;
using Xunit;

namespace VL.Cuda.Tests.Graph;

public class CompiledGraphTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly ModuleCache _cache;
    private readonly BufferPool _pool;
    private readonly CudaStream _stream;

    private static string VectorAddPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");
    private static string ScalarMulPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "scalar_mul.ptx");

    public CompiledGraphTests()
    {
        _cache = new ModuleCache(_device);
        _pool = new BufferPool(_device);
        _stream = new CudaStream();
    }

    [Fact]
    public void Launch_SingleKernel_ProducesCorrectResult()
    {
        const int N = 1024;
        var hostA = new float[N];
        var hostB = new float[N];
        for (int i = 0; i < N; i++)
        {
            hostA[i] = i;
            hostB[i] = i * 2.0f;
        }

        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);
        bufA.Upload(hostA);
        bufB.Upload(hostB);

        var builder = new GraphBuilder(_device, _cache);
        var add = builder.AddKernel(VectorAddPath, "add");
        add.GridDimX = (uint)((N + 255) / 256);

        builder.SetExternalBuffer(add, 0, bufA.Pointer);
        builder.SetExternalBuffer(add, 1, bufB.Pointer);
        builder.SetExternalBuffer(add, 2, bufC.Pointer);
        add.SetScalar(3, (uint)N);

        var compiler = new GraphCompiler(_device, _pool);
        using var compiled = compiler.Compile(builder);

        compiled.LaunchAndSync(_stream);

        var result = bufC.Download();
        for (int i = 0; i < N; i++)
            Assert.Equal(hostA[i] + hostB[i], result[i]);
    }

    [Fact]
    public void HotUpdate_Scalar_ProducesNewResult()
    {
        const int N = 512;
        var hostA = new float[N];
        for (int i = 0; i < N; i++)
            hostA[i] = i + 1.0f;

        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);
        bufA.Upload(hostA);

        var builder = new GraphBuilder(_device, _cache);
        var mul = builder.AddKernel(ScalarMulPath, "mul");
        mul.GridDimX = (uint)((N + 255) / 256);

        builder.SetExternalBuffer(mul, 0, bufA.Pointer); // A input
        builder.SetExternalBuffer(mul, 1, bufC.Pointer);  // C output
        mul.SetScalar(2, 2.0f); // scale = 2
        mul.SetScalar(3, (uint)N);

        var compiler = new GraphCompiler(_device, _pool);
        using var compiled = compiler.Compile(builder);

        // First launch: scale = 2
        compiled.LaunchAndSync(_stream);
        var result1 = bufC.Download();
        for (int i = 0; i < N; i++)
            Assert.Equal(hostA[i] * 2.0f, result1[i]);

        // Hot update: change scale to 5
        compiled.UpdateScalar(mul.Id, 2, 5.0f);

        compiled.LaunchAndSync(_stream);
        var result2 = bufC.Download();
        for (int i = 0; i < N; i++)
            Assert.Equal(hostA[i] * 5.0f, result2[i]);
    }

    [Fact]
    public void WarmUpdate_Pointer_SwapsBuffer()
    {
        const int N = 256;
        var hostA = new float[N];
        var hostB = new float[N];
        for (int i = 0; i < N; i++)
        {
            hostA[i] = 1.0f;
            hostB[i] = 100.0f;
        }

        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);
        bufA.Upload(hostA);
        bufB.Upload(hostB);

        var builder = new GraphBuilder(_device, _cache);
        var mul = builder.AddKernel(ScalarMulPath, "mul");
        mul.GridDimX = (uint)((N + 255) / 256);

        builder.SetExternalBuffer(mul, 0, bufA.Pointer);
        builder.SetExternalBuffer(mul, 1, bufC.Pointer);
        mul.SetScalar(2, 3.0f);
        mul.SetScalar(3, (uint)N);

        var compiler = new GraphCompiler(_device, _pool);
        using var compiled = compiler.Compile(builder);

        // First launch with bufA: result = 1.0 * 3.0 = 3.0
        compiled.LaunchAndSync(_stream);
        var result1 = bufC.Download();
        for (int i = 0; i < N; i++)
            Assert.Equal(3.0f, result1[i]);

        // Warm update: swap input from bufA to bufB
        compiled.UpdatePointer(mul.Id, 0, bufB.Pointer);

        compiled.LaunchAndSync(_stream);
        var result2 = bufC.Download();
        for (int i = 0; i < N; i++)
            Assert.Equal(300.0f, result2[i]); // 100.0 * 3.0
    }

    [Fact]
    public void WarmUpdate_Grid_ChangesLaunchSize()
    {
        const int N = 256;
        var hostA = new float[N];
        for (int i = 0; i < N; i++)
            hostA[i] = (float)i;

        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);
        bufA.Upload(hostA);

        var builder = new GraphBuilder(_device, _cache);
        var mul = builder.AddKernel(ScalarMulPath, "mul");

        // Start with grid that covers only first 128 elements
        mul.GridDimX = 1; // 1 block * 256 threads = 256 threads
        // But N=128 in first launch
        builder.SetExternalBuffer(mul, 0, bufA.Pointer);
        builder.SetExternalBuffer(mul, 1, bufC.Pointer);
        mul.SetScalar(2, 2.0f);
        mul.SetScalar(3, 128u); // only first 128

        var compiler = new GraphCompiler(_device, _pool);
        using var compiled = compiler.Compile(builder);

        compiled.LaunchAndSync(_stream);
        var result = bufC.Download();
        for (int i = 0; i < 128; i++)
            Assert.Equal(hostA[i] * 2.0f, result[i]);

        // Update N and grid to cover all 256 elements
        compiled.UpdateScalar(mul.Id, 3, (uint)N);
        compiled.UpdateGrid(mul.Id, (uint)((N + 255) / 256));

        compiled.LaunchAndSync(_stream);
        var result2 = bufC.Download();
        for (int i = 0; i < N; i++)
            Assert.Equal(hostA[i] * 2.0f, result2[i]);
    }

    [Fact]
    public void Dispose_CleansUp()
    {
        const int N = 256;

        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);

        var builder = new GraphBuilder(_device, _cache);
        var add = builder.AddKernel(VectorAddPath, "add");
        add.GridDimX = 1;
        builder.SetExternalBuffer(add, 0, bufA.Pointer);
        builder.SetExternalBuffer(add, 1, bufB.Pointer);
        builder.SetExternalBuffer(add, 2, bufC.Pointer);
        add.SetScalar(3, (uint)N);

        var compiler = new GraphCompiler(_device, _pool);
        var compiled = compiler.Compile(builder);
        compiled.Dispose();

        Assert.Throws<ObjectDisposedException>(() => compiled.Launch(_stream));
    }

    public void Dispose()
    {
        _stream.Dispose();
        _pool.Dispose();
        _cache.Dispose();
        _device.Dispose();
    }
}
