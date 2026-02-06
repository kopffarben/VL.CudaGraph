using System;
using System.IO;
using System.Linq;
using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Context;
using VL.Cuda.Core.Device;
using VL.Cuda.Tests.Helpers;
using Xunit;

namespace VL.Cuda.Tests.Blocks;

public class BlockBuilderTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly CudaContext _ctx;

    private static string VectorAddPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");
    private static string ScalarMulPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "scalar_mul.ptx");

    public BlockBuilderTests()
    {
        _ctx = new CudaContext(_device);
    }

    [Fact]
    public void AddKernel_ReturnsHandle()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);

        var kernel = builder.AddKernel(VectorAddPath);

        Assert.NotNull(kernel);
        Assert.NotEqual(Guid.Empty, kernel.Id);
        Assert.Equal("vector_add", kernel.Descriptor.EntryPoint);
    }

    [Fact]
    public void Input_CreatesPort()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var kernel = builder.AddKernel(VectorAddPath);

        var port = builder.Input<float>("A", kernel.In(0));

        Assert.Equal("A", port.Name);
        Assert.Equal(PortDirection.Input, port.Direction);
        Assert.Equal(PinType.Buffer<float>(), port.Type);
    }

    [Fact]
    public void Output_CreatesPort()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var kernel = builder.AddKernel(VectorAddPath);

        var port = builder.Output<float>("C", kernel.Out(2));

        Assert.Equal("C", port.Name);
        Assert.Equal(PortDirection.Output, port.Direction);
    }

    [Fact]
    public void InputScalar_CreatesParameter()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var kernel = builder.AddKernel(VectorAddPath);

        var param = builder.InputScalar<uint>("N", kernel.In(3), 1024u);

        Assert.Equal("N", param.Name);
        Assert.Equal(1024u, param.TypedValue);
    }

    [Fact]
    public void Connect_InternalConnection()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var k1 = builder.AddKernel(VectorAddPath);
        var k2 = builder.AddKernel(ScalarMulPath);

        builder.Connect(k1.Out(2), k2.In(0));

        Assert.Single(builder.InternalConnections);
    }

    [Fact]
    public void Commit_StoresDescription()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var kernel = builder.AddKernel(VectorAddPath);
        builder.Input<float>("A", kernel.In(0));
        builder.Output<float>("C", kernel.Out(2));
        builder.InputScalar<uint>("N", kernel.In(3));

        builder.Commit();

        var desc = _ctx.GetBlockDescription(block.Id);
        Assert.NotNull(desc);
        Assert.Single(desc.KernelEntries);
        Assert.Equal(2, desc.Ports.Count); // Input + Output (scalar not counted as port)
    }

    [Fact]
    public void Commit_ThrowsOnDoubleCommit()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        builder.AddKernel(VectorAddPath);
        builder.Commit();

        Assert.Throws<InvalidOperationException>(() => builder.Commit());
    }

    [Fact]
    public void AfterCommit_CannotAddKernel()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        builder.AddKernel(VectorAddPath);
        builder.Commit();

        Assert.Throws<InvalidOperationException>(() => builder.AddKernel(VectorAddPath));
    }

    [Fact]
    public void ParameterChange_MarksContextDirty()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var kernel = builder.AddKernel(ScalarMulPath);
        var scale = builder.InputScalar<float>("scale", kernel.In(2), 1.0f);
        builder.Commit();
        _ctx.Dirty.ClearStructureDirty();
        _ctx.Dirty.ClearParametersDirty();

        // Change the parameter value
        scale.TypedValue = 5.0f;

        Assert.True(_ctx.Dirty.AreParametersDirty);
    }

    [Fact]
    public void MultipleKernels_AllTracked()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var k1 = builder.AddKernel(VectorAddPath);
        var k2 = builder.AddKernel(ScalarMulPath);
        builder.Commit();

        Assert.Equal(2, builder.Kernels.Count);
        var desc = _ctx.GetBlockDescription(block.Id);
        Assert.Equal(2, desc!.KernelEntries.Count);
    }

    [Fact]
    public void KernelHandle_InOut_ReferenceCorrectParams()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var kernel = builder.AddKernel(VectorAddPath);

        var pin0 = kernel.In(0);
        var pin2 = kernel.Out(2);

        Assert.Equal(kernel.Id, pin0.KernelHandleId);
        Assert.Equal(0, pin0.ParamIndex);
        Assert.Equal(kernel.Id, pin2.KernelHandleId);
        Assert.Equal(2, pin2.ParamIndex);
    }

    public void Dispose()
    {
        _ctx.Dispose();
    }
}
