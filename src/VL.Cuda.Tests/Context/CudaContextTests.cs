using System;
using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Context;
using VL.Cuda.Core.Context.Services;
using VL.Cuda.Core.Device;
using VL.Cuda.Tests.Helpers;
using Xunit;

namespace VL.Cuda.Tests.Context;

public class CudaContextTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly CudaContext _ctx;

    public CudaContextTests()
    {
        _ctx = new CudaContext(_device);
    }

    [Fact]
    public void RegisterBlock_AddsToRegistry()
    {
        var block = new TestBlock("A");
        _ctx.RegisterBlock(block);

        Assert.Equal(1, _ctx.Registry.Count);
        Assert.Same(block, _ctx.Registry.Get(block.Id));
    }

    [Fact]
    public void UnregisterBlock_RemovesFromRegistry()
    {
        var block = new TestBlock("A");
        _ctx.RegisterBlock(block);

        _ctx.UnregisterBlock(block.Id);

        Assert.Equal(0, _ctx.Registry.Count);
    }

    [Fact]
    public void UnregisterBlock_RemovesConnections()
    {
        var a = new TestBlock("A");
        var b = new TestBlock("B");
        _ctx.RegisterBlock(a);
        _ctx.RegisterBlock(b);
        _ctx.Connect(a.Id, "Out", b.Id, "In");

        _ctx.UnregisterBlock(a.Id);

        Assert.Equal(0, _ctx.Connections.Count);
    }

    [Fact]
    public void Connect_AddsConnection()
    {
        var a = new TestBlock("A");
        var b = new TestBlock("B");
        _ctx.RegisterBlock(a);
        _ctx.RegisterBlock(b);

        _ctx.Connect(a.Id, "Out", b.Id, "In");

        Assert.Equal(1, _ctx.Connections.Count);
    }

    [Fact]
    public void Disconnect_RemovesConnection()
    {
        var a = new TestBlock("A");
        var b = new TestBlock("B");
        _ctx.RegisterBlock(a);
        _ctx.RegisterBlock(b);
        _ctx.Connect(a.Id, "Out", b.Id, "In");

        _ctx.Disconnect(a.Id, "Out", b.Id, "In");

        Assert.Equal(0, _ctx.Connections.Count);
    }

    [Fact]
    public void RegisterBlock_MarksStructureDirty()
    {
        _ctx.Dirty.ClearStructureDirty();
        Assert.False(_ctx.Dirty.IsStructureDirty);

        _ctx.RegisterBlock(new TestBlock("A"));

        Assert.True(_ctx.Dirty.IsStructureDirty);
    }

    [Fact]
    public void Connect_MarksStructureDirty()
    {
        _ctx.Dirty.ClearStructureDirty();

        _ctx.Connect(Guid.NewGuid(), "Out", Guid.NewGuid(), "In");

        Assert.True(_ctx.Dirty.IsStructureDirty);
    }

    [Fact]
    public void OnParameterChanged_MarksParameterDirty()
    {
        var blockId = Guid.NewGuid();
        _ctx.OnParameterChanged(blockId, "Scale");

        Assert.True(_ctx.Dirty.AreParametersDirty);
        Assert.Contains(new DirtyParameter(blockId, "Scale"), _ctx.Dirty.GetDirtyParameters());
    }

    [Fact]
    public void SetBlockDescription_StoresAndRetrieves()
    {
        var blockId = Guid.NewGuid();
        var desc = new VL.Cuda.Core.Blocks.Builder.BlockDescription(
            new[] { new VL.Cuda.Core.Blocks.Builder.KernelEntry(Guid.NewGuid(), "test.ptx", "kernel1", 1, 1, 1) },
            new[] { ("In", PortDirection.Input, PinType.Buffer<float>()) },
            Array.Empty<(int, int, int, int)>());

        _ctx.SetBlockDescription(blockId, desc);

        var retrieved = _ctx.GetBlockDescription(blockId);
        Assert.NotNull(retrieved);
        Assert.True(desc.StructuralEquals(retrieved));
    }

    [Fact]
    public void GetBlockDescription_Unknown_ReturnsNull()
    {
        Assert.Null(_ctx.GetBlockDescription(Guid.NewGuid()));
    }

    [Fact]
    public void SetExternalBuffer_StoresBinding()
    {
        var blockId = Guid.NewGuid();
        var ptr = new ManagedCuda.BasicTypes.CUdeviceptr(12345);

        _ctx.SetExternalBuffer(blockId, "Input", ptr);

        Assert.True(_ctx.ExternalBuffers.ContainsKey((blockId, "Input")));
    }

    public void Dispose()
    {
        _ctx.Dispose();
    }
}
