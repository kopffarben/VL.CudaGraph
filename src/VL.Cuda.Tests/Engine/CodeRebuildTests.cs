using System;
using VL.Cuda.Core.Context.Services;
using VL.Cuda.Core.PTX;
using VL.Cuda.Tests.Helpers;
using Xunit;

namespace VL.Cuda.Tests.Engine;

public class CodeRebuildTests
{
    [Fact]
    public void DirtyTracker_CodeDirty_StartsClean()
    {
        var tracker = new DirtyTracker();
        Assert.False(tracker.IsCodeDirty);
    }

    [Fact]
    public void DirtyTracker_MarkCodeDirty_SetsFlag()
    {
        var tracker = new DirtyTracker();
        var source = new KernelSource.FilesystemPtx("test.ptx");
        tracker.MarkCodeDirty(new DirtyCodeEntry(Guid.NewGuid(), Guid.NewGuid(), source));

        Assert.True(tracker.IsCodeDirty);
    }

    [Fact]
    public void DirtyTracker_ClearCodeDirty_ClearsFlag()
    {
        var tracker = new DirtyTracker();
        var source = new KernelSource.FilesystemPtx("test.ptx");
        tracker.MarkCodeDirty(new DirtyCodeEntry(Guid.NewGuid(), Guid.NewGuid(), source));

        tracker.ClearCodeDirty();

        Assert.False(tracker.IsCodeDirty);
    }

    [Fact]
    public void DirtyTracker_ClearStructure_AlsoClearsCode()
    {
        var tracker = new DirtyTracker();
        var source = new KernelSource.FilesystemPtx("test.ptx");
        tracker.MarkCodeDirty(new DirtyCodeEntry(Guid.NewGuid(), Guid.NewGuid(), source));

        tracker.ClearStructureDirty();

        Assert.False(tracker.IsCodeDirty);
    }

    [Fact]
    public void DirtyTracker_GetDirtyCodeEntries_ReturnsAll()
    {
        var tracker = new DirtyTracker();
        var blockId = Guid.NewGuid();
        var source1 = new KernelSource.FilesystemPtx("a.ptx");
        var source2 = new KernelSource.FilesystemPtx("b.ptx");
        var id1 = Guid.NewGuid();
        var id2 = Guid.NewGuid();

        tracker.MarkCodeDirty(new DirtyCodeEntry(blockId, id1, source1));
        tracker.MarkCodeDirty(new DirtyCodeEntry(blockId, id2, source2));

        var dirty = tracker.GetDirtyCodeEntries();
        Assert.Equal(2, dirty.Count);
    }

    [Fact]
    public void DirtyTracker_Priority_StructureBeforeCode()
    {
        var tracker = new DirtyTracker();
        var source = new KernelSource.FilesystemPtx("test.ptx");
        tracker.MarkCodeDirty(new DirtyCodeEntry(Guid.NewGuid(), Guid.NewGuid(), source));

        // Structure starts dirty (initial state)
        Assert.True(tracker.IsStructureDirty);
        Assert.True(tracker.IsCodeDirty);

        // Structure takes priority
        tracker.ClearStructureDirty();
        Assert.False(tracker.IsStructureDirty);
        Assert.False(tracker.IsCodeDirty); // also cleared
    }

    [Fact]
    public void DirtyTracker_Priority_CodeBeforeRecapture()
    {
        var tracker = new DirtyTracker();
        tracker.ClearStructureDirty(); // clear initial dirty

        var source = new KernelSource.FilesystemPtx("test.ptx");
        tracker.MarkCodeDirty(new DirtyCodeEntry(Guid.NewGuid(), Guid.NewGuid(), source));
        tracker.MarkCapturedNodeDirty(new DirtyCapturedNode(Guid.NewGuid(), Guid.NewGuid()));

        // Code and Recapture both dirty
        Assert.True(tracker.IsCodeDirty);
        Assert.True(tracker.AreCapturedNodesDirty);

        // Code has higher priority — code clear doesn't clear recapture
        tracker.ClearCodeDirty();
        Assert.False(tracker.IsCodeDirty);
        Assert.True(tracker.AreCapturedNodesDirty); // still dirty
    }

    [Fact]
    public void DirtyTracker_Priority_CodeBeforeParameters()
    {
        var tracker = new DirtyTracker();
        tracker.ClearStructureDirty();

        var source = new KernelSource.FilesystemPtx("test.ptx");
        tracker.MarkCodeDirty(new DirtyCodeEntry(Guid.NewGuid(), Guid.NewGuid(), source));
        tracker.MarkParameterDirty(new DirtyParameter(Guid.NewGuid(), "Scale"));

        Assert.True(tracker.IsCodeDirty);
        Assert.True(tracker.AreParametersDirty);

        // Code clear doesn't clear parameters
        tracker.ClearCodeDirty();
        Assert.False(tracker.IsCodeDirty);
        Assert.True(tracker.AreParametersDirty); // still dirty
    }

    [Fact]
    public void CudaContext_OnCodeChanged_MarksDirty()
    {
        using var device = new VL.Cuda.Core.Device.DeviceContext(0);
        using var ctx = new VL.Cuda.Core.Context.CudaContext(device);

        ctx.Dirty.ClearStructureDirty();

        var blockId = Guid.NewGuid();
        var handleId = Guid.NewGuid();
        var source = new KernelSource.NvrtcSource("hash", "source", "entry");

        ctx.OnCodeChanged(blockId, handleId, source);

        Assert.True(ctx.Dirty.IsCodeDirty);
        var entries = ctx.Dirty.GetDirtyCodeEntries();
        Assert.Single(entries);
    }
}
