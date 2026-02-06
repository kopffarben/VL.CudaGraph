using System;
using System.IO;
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.PTX;
using VL.Cuda.Core.Graph;
using Xunit;

namespace VL.Cuda.Tests.Graph;

public class KernelNodeTests : IDisposable
{
    private readonly DeviceContext _device = new(0);

    private static string VectorAddPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");
    private static string ScalarMulPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "scalar_mul.ptx");

    [Fact]
    public void Create_SetsProperties()
    {
        using var loaded = PtxLoader.Load(_device, VectorAddPath);
        using var node = new KernelNode(loaded, "test_node");

        Assert.Equal("test_node", node.DebugName);
        Assert.NotEqual(Guid.Empty, node.Id);
        Assert.Equal(4, node.ParameterCount); // A, B, C, N
        Assert.Equal(256u, node.BlockDimX);
        Assert.Equal(1u, node.BlockDimY);
        Assert.Equal(1u, node.BlockDimZ);
    }

    [Fact]
    public void Create_DefaultName_UsesEntryPoint()
    {
        using var loaded = PtxLoader.Load(_device, VectorAddPath);
        using var node = new KernelNode(loaded);

        Assert.Equal("vector_add", node.DebugName);
    }

    [Fact]
    public void SetScalar_WritesValue()
    {
        using var loaded = PtxLoader.Load(_device, ScalarMulPath);
        using var node = new KernelNode(loaded);

        // param index 2 = "scale" (float)
        node.SetScalar(2, 3.14f);

        // Verify via GetParamsPtr — read the value back through the native pointer
        var paramsPtr = node.GetParamsPtr();
        var slot = Marshal.ReadIntPtr(paramsPtr, 2 * IntPtr.Size);
        unsafe
        {
            float value = *(float*)slot;
            Assert.Equal(3.14f, value);
        }
    }

    [Fact]
    public void SetPointer_WritesDeviceAddress()
    {
        using var loaded = PtxLoader.Load(_device, VectorAddPath);
        using var node = new KernelNode(loaded);

        var ptr = new CUdeviceptr((SizeT)0x12345678UL);
        node.SetPointer(0, ptr);

        var paramsPtr = node.GetParamsPtr();
        var slot = Marshal.ReadIntPtr(paramsPtr, 0 * IntPtr.Size);
        var readBack = Marshal.ReadIntPtr(slot);
        Assert.Equal(new IntPtr(0x12345678), readBack);
    }

    [Fact]
    public void SetScalar_OutOfRange_Throws()
    {
        using var loaded = PtxLoader.Load(_device, VectorAddPath);
        using var node = new KernelNode(loaded);

        Assert.Throws<ArgumentOutOfRangeException>(() => node.SetScalar(10, 1.0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => node.SetScalar(-1, 1.0f));
    }

    [Fact]
    public void BuildNodeParams_ReturnsValidStruct()
    {
        using var loaded = PtxLoader.Load(_device, VectorAddPath);
        using var node = new KernelNode(loaded);

        node.GridDimX = 4;
        node.GridDimY = 2;
        node.GridDimZ = 1;

        var p = node.BuildNodeParams();

        Assert.Equal(4u, p.gridDimX);
        Assert.Equal(2u, p.gridDimY);
        Assert.Equal(1u, p.gridDimZ);
        Assert.Equal(256u, p.blockDimX);
        Assert.Equal(1u, p.blockDimY);
        Assert.Equal(1u, p.blockDimZ);
        Assert.NotEqual(IntPtr.Zero, p.kernelParams);
        Assert.Equal(IntPtr.Zero, p.extra);
    }

    [Fact]
    public void Dispose_PreventsFurtherUse()
    {
        using var loaded = PtxLoader.Load(_device, VectorAddPath);
        var node = new KernelNode(loaded);
        node.Dispose();

        Assert.Throws<ObjectDisposedException>(() => node.SetScalar(0, 1.0f));
    }

    public void Dispose() => _device.Dispose();
}
