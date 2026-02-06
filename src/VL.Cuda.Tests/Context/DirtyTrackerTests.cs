using System;
using VL.Cuda.Core.Context.Services;
using VL.Cuda.Tests.Helpers;
using Xunit;

namespace VL.Cuda.Tests.Context;

public class DirtyTrackerTests
{
    [Fact]
    public void StartsStructureDirty()
    {
        var tracker = new DirtyTracker();
        Assert.True(tracker.IsStructureDirty);
    }

    [Fact]
    public void StartsParametersClean()
    {
        var tracker = new DirtyTracker();
        Assert.False(tracker.AreParametersDirty);
    }

    [Fact]
    public void ClearStructureDirty_ClearsFlag()
    {
        var tracker = new DirtyTracker();
        Assert.True(tracker.IsStructureDirty);

        tracker.ClearStructureDirty();

        Assert.False(tracker.IsStructureDirty);
    }

    [Fact]
    public void MarkParameterDirty_SetsFlag()
    {
        var tracker = new DirtyTracker();
        tracker.ClearStructureDirty();

        tracker.MarkParameterDirty(new DirtyParameter(Guid.NewGuid(), "Scale"));

        Assert.True(tracker.AreParametersDirty);
    }

    [Fact]
    public void ClearParametersDirty_ClearsFlag()
    {
        var tracker = new DirtyTracker();
        tracker.MarkParameterDirty(new DirtyParameter(Guid.NewGuid(), "Scale"));

        tracker.ClearParametersDirty();

        Assert.False(tracker.AreParametersDirty);
    }

    [Fact]
    public void ClearStructureDirty_AlsoClearsParameters()
    {
        var tracker = new DirtyTracker();
        tracker.MarkParameterDirty(new DirtyParameter(Guid.NewGuid(), "Scale"));

        tracker.ClearStructureDirty();

        Assert.False(tracker.AreParametersDirty);
    }

    [Fact]
    public void Subscribe_RegistryChange_MarksStructureDirty()
    {
        var registry = new BlockRegistry();
        var connections = new ConnectionGraph();
        var tracker = new DirtyTracker();
        tracker.Subscribe(registry, connections);
        tracker.ClearStructureDirty();

        registry.Register(new TestBlock("A"));

        Assert.True(tracker.IsStructureDirty);
    }

    [Fact]
    public void Subscribe_ConnectionChange_MarksStructureDirty()
    {
        var registry = new BlockRegistry();
        var connections = new ConnectionGraph();
        var tracker = new DirtyTracker();
        tracker.Subscribe(registry, connections);
        tracker.ClearStructureDirty();

        connections.Connect(Guid.NewGuid(), "Out", Guid.NewGuid(), "In");

        Assert.True(tracker.IsStructureDirty);
    }

    [Fact]
    public void GetDirtyParameters_ReturnsAllDirty()
    {
        var tracker = new DirtyTracker();
        var blockId = Guid.NewGuid();
        tracker.MarkParameterDirty(new DirtyParameter(blockId, "Scale"));
        tracker.MarkParameterDirty(new DirtyParameter(blockId, "Offset"));

        var dirty = tracker.GetDirtyParameters();

        Assert.Equal(2, dirty.Count);
        Assert.Contains(new DirtyParameter(blockId, "Scale"), dirty);
        Assert.Contains(new DirtyParameter(blockId, "Offset"), dirty);
    }

    [Fact]
    public void MarkParameterDirty_Duplicate_NoDuplicate()
    {
        var tracker = new DirtyTracker();
        var blockId = Guid.NewGuid();
        tracker.MarkParameterDirty(new DirtyParameter(blockId, "Scale"));
        tracker.MarkParameterDirty(new DirtyParameter(blockId, "Scale"));

        Assert.Single(tracker.GetDirtyParameters());
    }
}
