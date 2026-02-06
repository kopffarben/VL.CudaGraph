using System;
using System.IO;
using ManagedCuda;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.PTX;
using VL.Cuda.Core.Graph;
using Xunit;

namespace VL.Cuda.Tests.Integration;

/// <summary>
/// End-to-end multi-kernel pipeline tests through the CUDA Graph API.
/// </summary>
public class GraphPipelineTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly ModuleCache _cache;
    private readonly BufferPool _pool;
    private readonly CudaStream _stream;

    private static string VectorAddPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");
    private static string ScalarMulPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "scalar_mul.ptx");

    public GraphPipelineTests()
    {
        _cache = new ModuleCache(_device);
        _pool = new BufferPool(_device);
        _stream = new CudaStream();
    }

    [Fact]
    public void TwoNodeChain_VectorAdd_ThenScalarMul()
    {
        // Pipeline: C = (A + B) * scale
        // Node 1: vector_add(A, B, C_intermediate, N)
        // Node 2: scalar_mul(C_intermediate, D, scale, N)
        const int N = 1024;
        const float scale = 3.0f;

        var hostA = new float[N];
        var hostB = new float[N];
        for (int i = 0; i < N; i++)
        {
            hostA[i] = i;
            hostB[i] = i * 0.5f;
        }

        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufD = GpuBuffer<float>.Allocate(_device, N);
        bufA.Upload(hostA);
        bufB.Upload(hostB);

        var builder = new GraphBuilder(_device, _cache);

        var add = builder.AddKernel(VectorAddPath, "add");
        var mul = builder.AddKernel(ScalarMulPath, "mul");

        uint gridSize = (uint)((N + 255) / 256);
        add.GridDimX = gridSize;
        mul.GridDimX = gridSize;

        // External inputs
        builder.SetExternalBuffer(add, 0, bufA.Pointer);
        builder.SetExternalBuffer(add, 1, bufB.Pointer);
        add.SetScalar(3, (uint)N);

        // External output
        builder.SetExternalBuffer(mul, 1, bufD.Pointer);
        mul.SetScalar(2, scale);
        mul.SetScalar(3, (uint)N);

        // Edge: add.C (index 2) → mul.A (index 0)
        builder.AddEdge(add, 2, mul, 0);

        var compiler = new GraphCompiler(_device, _pool);
        using var compiled = compiler.Compile(builder);

        compiled.LaunchAndSync(_stream);

        var result = bufD.Download();
        for (int i = 0; i < N; i++)
        {
            var expected = (hostA[i] + hostB[i]) * scale;
            Assert.Equal(expected, result[i], 4); // float precision
        }
    }

    [Fact]
    public void TwoNodeChain_HotUpdate_ChangesResult()
    {
        // Same pipeline as above, but we change `scale` mid-flight
        const int N = 512;

        var hostA = new float[N];
        var hostB = new float[N];
        for (int i = 0; i < N; i++)
        {
            hostA[i] = 1.0f;
            hostB[i] = 2.0f;
        }

        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufD = GpuBuffer<float>.Allocate(_device, N);
        bufA.Upload(hostA);
        bufB.Upload(hostB);

        var builder = new GraphBuilder(_device, _cache);
        var add = builder.AddKernel(VectorAddPath, "add");
        var mul = builder.AddKernel(ScalarMulPath, "mul");

        uint gridSize = (uint)((N + 255) / 256);
        add.GridDimX = gridSize;
        mul.GridDimX = gridSize;

        builder.SetExternalBuffer(add, 0, bufA.Pointer);
        builder.SetExternalBuffer(add, 1, bufB.Pointer);
        add.SetScalar(3, (uint)N);

        builder.SetExternalBuffer(mul, 1, bufD.Pointer);
        mul.SetScalar(2, 10.0f); // scale = 10
        mul.SetScalar(3, (uint)N);

        builder.AddEdge(add, 2, mul, 0);

        var compiler = new GraphCompiler(_device, _pool);
        using var compiled = compiler.Compile(builder);

        // Launch 1: (1+2) * 10 = 30
        compiled.LaunchAndSync(_stream);
        var result1 = bufD.Download();
        for (int i = 0; i < N; i++)
            Assert.Equal(30.0f, result1[i]);

        // Hot update: scale = 0.5
        compiled.UpdateScalar(mul.Id, 2, 0.5f);

        // Launch 2: (1+2) * 0.5 = 1.5
        compiled.LaunchAndSync(_stream);
        var result2 = bufD.Download();
        for (int i = 0; i < N; i++)
            Assert.Equal(1.5f, result2[i]);
    }

    [Fact]
    public void DiamondGraph_ProducesCorrectResult()
    {
        // Diamond: A → B, A → C, B+C → D
        // Node A: vector_add(X, Y, A_out, N)       — creates A_out
        // Node B: scalar_mul(A_out, B_out, 2.0, N)  — B_out = A_out * 2
        // Node C: scalar_mul(A_out, C_out, 3.0, N)  — C_out = A_out * 3
        // Node D: vector_add(B_out, C_out, D_out, N) — D_out = B_out + C_out
        //
        // Expected: D_out = (X+Y)*2 + (X+Y)*3 = (X+Y)*5
        const int N = 512;

        var hostX = new float[N];
        var hostY = new float[N];
        for (int i = 0; i < N; i++)
        {
            hostX[i] = i;
            hostY[i] = 1.0f;
        }

        using var bufX = GpuBuffer<float>.Allocate(_device, N);
        using var bufY = GpuBuffer<float>.Allocate(_device, N);
        using var bufDout = GpuBuffer<float>.Allocate(_device, N);
        bufX.Upload(hostX);
        bufY.Upload(hostY);

        var builder = new GraphBuilder(_device, _cache);
        var nodeA = builder.AddKernel(VectorAddPath, "A");
        var nodeB = builder.AddKernel(ScalarMulPath, "B");
        var nodeC = builder.AddKernel(ScalarMulPath, "C");
        var nodeD = builder.AddKernel(VectorAddPath, "D");

        uint gridSize = (uint)((N + 255) / 256);
        nodeA.GridDimX = gridSize;
        nodeB.GridDimX = gridSize;
        nodeC.GridDimX = gridSize;
        nodeD.GridDimX = gridSize;

        // A inputs
        builder.SetExternalBuffer(nodeA, 0, bufX.Pointer);
        builder.SetExternalBuffer(nodeA, 1, bufY.Pointer);
        nodeA.SetScalar(3, (uint)N);

        // B: scale = 2
        nodeB.SetScalar(2, 2.0f);
        nodeB.SetScalar(3, (uint)N);

        // C: scale = 3
        nodeC.SetScalar(2, 3.0f);
        nodeC.SetScalar(3, (uint)N);

        // D: output
        builder.SetExternalBuffer(nodeD, 2, bufDout.Pointer);
        nodeD.SetScalar(3, (uint)N);

        // Edges:
        // A.C (output, index 2) → B.A (input, index 0)
        builder.AddEdge(nodeA, 2, nodeB, 0);
        // A.C (output, index 2) → C.A (input, index 0)
        builder.AddEdge(nodeA, 2, nodeC, 0);
        // B.C (output, index 1) → D.A (input, index 0)
        builder.AddEdge(nodeB, 1, nodeD, 0);
        // C.C (output, index 1) → D.B (input, index 1)
        builder.AddEdge(nodeC, 1, nodeD, 1);

        var compiler = new GraphCompiler(_device, _pool);
        using var compiled = compiler.Compile(builder);

        compiled.LaunchAndSync(_stream);

        var result = bufDout.Download();
        for (int i = 0; i < N; i++)
        {
            var expected = (hostX[i] + hostY[i]) * 5.0f;
            Assert.Equal(expected, result[i], 4);
        }
    }

    [Fact]
    public void ColdRebuild_NewGraph_Works()
    {
        const int N = 256;
        var hostA = new float[N];
        for (int i = 0; i < N; i++)
            hostA[i] = 10.0f;

        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);
        bufA.Upload(hostA);

        // Build first graph with scale = 2
        var builder1 = new GraphBuilder(_device, _cache);
        var mul1 = builder1.AddKernel(ScalarMulPath, "mul");
        mul1.GridDimX = 1;
        builder1.SetExternalBuffer(mul1, 0, bufA.Pointer);
        builder1.SetExternalBuffer(mul1, 1, bufC.Pointer);
        mul1.SetScalar(2, 2.0f);
        mul1.SetScalar(3, (uint)N);

        var compiler = new GraphCompiler(_device, _pool);
        using var compiled1 = compiler.Compile(builder1);
        compiled1.LaunchAndSync(_stream);

        var result1 = bufC.Download();
        for (int i = 0; i < N; i++)
            Assert.Equal(20.0f, result1[i]);

        // Build a completely new graph (cold rebuild) with scale = 5
        var builder2 = new GraphBuilder(_device, _cache);
        var mul2 = builder2.AddKernel(ScalarMulPath, "mul2");
        mul2.GridDimX = 1;
        builder2.SetExternalBuffer(mul2, 0, bufA.Pointer);
        builder2.SetExternalBuffer(mul2, 1, bufC.Pointer);
        mul2.SetScalar(2, 5.0f);
        mul2.SetScalar(3, (uint)N);

        using var compiled2 = compiler.Compile(builder2);
        compiled2.LaunchAndSync(_stream);

        var result2 = bufC.Download();
        for (int i = 0; i < N; i++)
            Assert.Equal(50.0f, result2[i]);
    }

    public void Dispose()
    {
        _stream.Dispose();
        _pool.Dispose();
        _cache.Dispose();
        _device.Dispose();
    }
}
