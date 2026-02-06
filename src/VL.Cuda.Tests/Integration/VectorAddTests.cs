using System;
using System.IO;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.PTX;
using Xunit;

namespace VL.Cuda.Tests.Integration;

public class VectorAddTests : IDisposable
{
    private readonly DeviceContext _device = new(0);

    private static string TestKernelPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");

    [Fact]
    public void VectorAdd_EndToEnd()
    {
        const int N = 1024;

        // Prepare host data
        var hostA = new float[N];
        var hostB = new float[N];
        for (int i = 0; i < N; i++)
        {
            hostA[i] = i;
            hostB[i] = i * 2.0f;
        }

        // Allocate GPU buffers and upload
        using var bufA = GpuBuffer<float>.Allocate(_device, N);
        using var bufB = GpuBuffer<float>.Allocate(_device, N);
        using var bufC = GpuBuffer<float>.Allocate(_device, N);

        bufA.Upload(hostA);
        bufB.Upload(hostB);

        // Load kernel
        using var loaded = PtxLoader.Load(_device, TestKernelPath);
        var kernel = loaded.Kernel;

        // Configure grid: ceil(N / blockSize) blocks
        int blockSize = loaded.Descriptor.BlockSize > 0 ? loaded.Descriptor.BlockSize : 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        kernel.BlockDimensions = new dim3(blockSize, 1, 1);
        kernel.GridDimensions = new dim3(gridSize, 1, 1);

        // Launch kernel: vector_add(A, B, C, N)
        kernel.Run(bufA.Pointer, bufB.Pointer, bufC.Pointer, (uint)N);

        // Download result
        var result = bufC.Download();

        // Verify
        for (int i = 0; i < N; i++)
        {
            var expected = hostA[i] + hostB[i];
            Assert.Equal(expected, result[i]);
        }
    }

    [Fact]
    public void VectorAdd_WithPool()
    {
        const int N = 2048;

        var hostA = new float[N];
        var hostB = new float[N];
        for (int i = 0; i < N; i++)
        {
            hostA[i] = i * 0.5f;
            hostB[i] = 100.0f - i * 0.25f;
        }

        using var pool = new BufferPool(_device);
        using var bufA = pool.Acquire<float>(N);
        using var bufB = pool.Acquire<float>(N);
        using var bufC = pool.Acquire<float>(N);

        bufA.Upload(hostA);
        bufB.Upload(hostB);

        using var loaded = PtxLoader.Load(_device, TestKernelPath);
        var kernel = loaded.Kernel;

        int blockSize = 256;
        kernel.BlockDimensions = new dim3(blockSize, 1, 1);
        kernel.GridDimensions = new dim3((N + blockSize - 1) / blockSize, 1, 1);

        kernel.Run(bufA.Pointer, bufB.Pointer, bufC.Pointer, (uint)N);

        var result = bufC.Download();

        for (int i = 0; i < N; i++)
        {
            var expected = hostA[i] + hostB[i];
            Assert.Equal(expected, result[i], 4); // float precision
        }
    }

    [Fact]
    public void VectorAdd_WithModuleCache()
    {
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
        using var bufC = GpuBuffer<float>.Allocate(_device, N);

        bufA.Upload(hostA);
        bufB.Upload(hostB);

        using var cache = new ModuleCache(_device);
        var loaded = cache.GetOrLoad(TestKernelPath);

        int blockSize = 256;
        loaded.Kernel.BlockDimensions = new dim3(blockSize, 1, 1);
        loaded.Kernel.GridDimensions = new dim3((N + blockSize - 1) / blockSize, 1, 1);

        loaded.Kernel.Run(bufA.Pointer, bufB.Pointer, bufC.Pointer, (uint)N);

        var result = bufC.Download();

        for (int i = 0; i < N; i++)
            Assert.Equal(3.0f, result[i]);
    }

    public void Dispose() => _device.Dispose();
}
