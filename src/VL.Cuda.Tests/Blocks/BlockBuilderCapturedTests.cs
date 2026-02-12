using System;
using System.IO;
using System.Linq;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Context;
using VL.Cuda.Core.Context.Services;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.Graph;
using VL.Cuda.Tests.Helpers;
using Xunit;

namespace VL.Cuda.Tests.Blocks;

public class BlockBuilderCapturedTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly CudaContext _ctx;

    public BlockBuilderCapturedTests()
    {
        _ctx = new CudaContext(_device);
    }

    [Fact]
    public void AddCaptured_ReturnsHandle()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);

        var desc = new CapturedNodeDescriptor("test.op",
            inputs: new[] { CapturedParam.Pointer("A", "float*") },
            outputs: new[] { CapturedParam.Pointer("C", "float*") });

        var handle = builder.AddCaptured((s, b) => { }, desc);

        Assert.NotNull(handle);
        Assert.NotEqual(Guid.Empty, handle.Id);
        Assert.Same(desc, handle.Descriptor);
    }

    [Fact]
    public void AddCaptured_TrackedInHandles()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var desc = new CapturedNodeDescriptor("test");

        builder.AddCaptured((s, b) => { }, desc);

        Assert.Single(builder.CapturedHandles);
    }

    [Fact]
    public void Input_WithCapturedPin_CreatesPort()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var desc = new CapturedNodeDescriptor("test",
            inputs: new[] { CapturedParam.Pointer("A", "float*") });
        var handle = builder.AddCaptured((s, b) => { }, desc);

        var port = builder.Input<float>("MatrixA", handle.In(0));

        Assert.Equal("MatrixA", port.Name);
        Assert.Equal(PortDirection.Input, port.Direction);
        Assert.Equal(handle.Id, port.KernelNodeId);
        Assert.Equal(0, port.KernelParamIndex);
    }

    [Fact]
    public void Output_WithCapturedPin_CreatesPort()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var desc = new CapturedNodeDescriptor("test",
            outputs: new[] { CapturedParam.Pointer("C", "float*") });
        var handle = builder.AddCaptured((s, b) => { }, desc);

        var port = builder.Output<float>("Result", handle.Out(0));

        Assert.Equal("Result", port.Name);
        Assert.Equal(PortDirection.Output, port.Direction);
        Assert.Equal(handle.Id, port.KernelNodeId);
        Assert.Equal(0, port.KernelParamIndex);
    }

    [Fact]
    public void Commit_WithCaptured_StoresDescription()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var desc = new CapturedNodeDescriptor("test.op",
            inputs: new[] { CapturedParam.Pointer("A", "float*") },
            outputs: new[] { CapturedParam.Pointer("C", "float*") });
        var handle = builder.AddCaptured((s, b) => { }, desc);
        builder.Input<float>("A", handle.In(0));
        builder.Output<float>("C", handle.Out(0));

        builder.Commit();

        var blockDesc = _ctx.GetBlockDescription(block.Id);
        Assert.NotNull(blockDesc);
        Assert.Single(blockDesc.CapturedEntries);
        Assert.Equal(2, blockDesc.Ports.Count);
    }

    [Fact]
    public void CapturedEntry_HandlesStructuralEquality()
    {
        var desc = new CapturedNodeDescriptor("test");
        Action<CUstream, CUdeviceptr[]> action = (s, b) => { };

        var entry1 = new CapturedEntry(Guid.NewGuid(), desc, action);
        var entry2 = new CapturedEntry(entry1.HandleId, desc, action);

        Assert.True(entry1.StructuralEquals(entry2));
    }

    [Fact]
    public void CapturedEntry_StructuralEquals_IgnoresHandleId()
    {
        // StructuralEquals intentionally ignores HandleId (same as KernelEntry),
        // because HandleIds change on every construction. Only descriptor structure matters.
        var desc = new CapturedNodeDescriptor("test");
        Action<CUstream, CUdeviceptr[]> action = (s, b) => { };

        var entry1 = new CapturedEntry(Guid.NewGuid(), desc, action);
        var entry2 = new CapturedEntry(Guid.NewGuid(), desc, action);

        Assert.True(entry1.StructuralEquals(entry2));
    }

    [Fact]
    public void CapturedEntry_DifferentDescriptor_NotEqual()
    {
        Action<CUstream, CUdeviceptr[]> action = (s, b) => { };
        var desc1 = new CapturedNodeDescriptor("test",
            inputs: new[] { CapturedParam.Pointer("A", "float*") });
        var desc2 = new CapturedNodeDescriptor("test");

        var entry1 = new CapturedEntry(Guid.NewGuid(), desc1, action);
        var entry2 = new CapturedEntry(Guid.NewGuid(), desc2, action);

        Assert.False(entry1.StructuralEquals(entry2));
    }

    [Fact]
    public void AfterCommit_CannotAddCaptured()
    {
        var block = new TestBlock("Test");
        var builder = new BlockBuilder(_ctx, block);
        var desc = new CapturedNodeDescriptor("test");
        builder.AddCaptured((s, b) => { }, desc);
        builder.Commit();

        Assert.Throws<InvalidOperationException>(() =>
            builder.AddCaptured((s, b) => { }, desc));
    }

    [Fact]
    public void DirtyTracker_CapturedNodeDirty_TracksSeparately()
    {
        var tracker = new DirtyTracker();
        tracker.ClearStructureDirty();

        Assert.False(tracker.AreCapturedNodesDirty);

        var dirty = new DirtyCapturedNode(Guid.NewGuid(), Guid.NewGuid());
        tracker.MarkCapturedNodeDirty(dirty);

        Assert.True(tracker.AreCapturedNodesDirty);
        Assert.False(tracker.AreParametersDirty);
        Assert.False(tracker.IsStructureDirty);
    }

    [Fact]
    public void DirtyTracker_GetDirtyCapturedNodes_ReturnsAll()
    {
        var tracker = new DirtyTracker();
        var d1 = new DirtyCapturedNode(Guid.NewGuid(), Guid.NewGuid());
        var d2 = new DirtyCapturedNode(Guid.NewGuid(), Guid.NewGuid());

        tracker.MarkCapturedNodeDirty(d1);
        tracker.MarkCapturedNodeDirty(d2);

        var dirty = tracker.GetDirtyCapturedNodes();
        Assert.Equal(2, dirty.Count);
    }

    [Fact]
    public void DirtyTracker_ClearCapturedNodesDirty_ClearsFlag()
    {
        var tracker = new DirtyTracker();
        tracker.MarkCapturedNodeDirty(new DirtyCapturedNode(Guid.NewGuid(), Guid.NewGuid()));

        tracker.ClearCapturedNodesDirty();

        Assert.False(tracker.AreCapturedNodesDirty);
    }

    [Fact]
    public void DirtyTracker_ClearStructureDirty_AlsoClearsCaptured()
    {
        var tracker = new DirtyTracker();
        tracker.MarkCapturedNodeDirty(new DirtyCapturedNode(Guid.NewGuid(), Guid.NewGuid()));

        tracker.ClearStructureDirty();

        Assert.False(tracker.AreCapturedNodesDirty);
    }

    [Fact]
    public void CudaContext_OnCapturedNodeChanged_MarksDirty()
    {
        var blockId = Guid.NewGuid();
        var handleId = Guid.NewGuid();

        _ctx.OnCapturedNodeChanged(blockId, handleId);

        Assert.True(_ctx.Dirty.AreCapturedNodesDirty);
    }

    public void Dispose()
    {
        _ctx.Dispose();
    }
}
