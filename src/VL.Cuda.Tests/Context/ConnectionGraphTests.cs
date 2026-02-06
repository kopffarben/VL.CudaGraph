using System;
using System.Linq;
using VL.Cuda.Core.Context.Services;
using Xunit;

namespace VL.Cuda.Tests.Context;

public class ConnectionGraphTests
{
    private static readonly Guid BlockA = Guid.NewGuid();
    private static readonly Guid BlockB = Guid.NewGuid();
    private static readonly Guid BlockC = Guid.NewGuid();

    [Fact]
    public void Connect_IncreasesCount()
    {
        var graph = new ConnectionGraph();
        graph.Connect(BlockA, "Out", BlockB, "In");

        Assert.Equal(1, graph.Count);
    }

    [Fact]
    public void Connect_Duplicate_DoesNotAdd()
    {
        var graph = new ConnectionGraph();
        graph.Connect(BlockA, "Out", BlockB, "In");
        graph.Connect(BlockA, "Out", BlockB, "In");

        Assert.Equal(1, graph.Count);
    }

    [Fact]
    public void Disconnect_RemovesConnection()
    {
        var graph = new ConnectionGraph();
        graph.Connect(BlockA, "Out", BlockB, "In");

        var removed = graph.Disconnect(BlockA, "Out", BlockB, "In");

        Assert.True(removed);
        Assert.Equal(0, graph.Count);
    }

    [Fact]
    public void Disconnect_NonExistent_ReturnsFalse()
    {
        var graph = new ConnectionGraph();
        Assert.False(graph.Disconnect(BlockA, "Out", BlockB, "In"));
    }

    [Fact]
    public void Connect_FiresStructureChanged()
    {
        var graph = new ConnectionGraph();
        int fireCount = 0;
        graph.StructureChanged += () => fireCount++;

        graph.Connect(BlockA, "Out", BlockB, "In");

        Assert.Equal(1, fireCount);
    }

    [Fact]
    public void Disconnect_FiresStructureChanged()
    {
        var graph = new ConnectionGraph();
        graph.Connect(BlockA, "Out", BlockB, "In");

        int fireCount = 0;
        graph.StructureChanged += () => fireCount++;
        graph.Disconnect(BlockA, "Out", BlockB, "In");

        Assert.Equal(1, fireCount);
    }

    [Fact]
    public void Connect_Duplicate_DoesNotFire()
    {
        var graph = new ConnectionGraph();
        graph.Connect(BlockA, "Out", BlockB, "In");

        int fireCount = 0;
        graph.StructureChanged += () => fireCount++;
        graph.Connect(BlockA, "Out", BlockB, "In");

        Assert.Equal(0, fireCount);
    }

    [Fact]
    public void RemoveBlock_RemovesAllConnections()
    {
        var graph = new ConnectionGraph();
        graph.Connect(BlockA, "Out", BlockB, "In");
        graph.Connect(BlockB, "Out", BlockC, "In");
        graph.Connect(BlockC, "Out", BlockA, "In");

        int removed = graph.RemoveBlock(BlockB);

        Assert.Equal(2, removed);
        Assert.Equal(1, graph.Count);
    }

    [Fact]
    public void RemoveBlock_NoConnections_ReturnsZero()
    {
        var graph = new ConnectionGraph();
        Assert.Equal(0, graph.RemoveBlock(BlockA));
    }

    [Fact]
    public void GetOutgoing_ReturnsSourceConnections()
    {
        var graph = new ConnectionGraph();
        graph.Connect(BlockA, "Out1", BlockB, "In");
        graph.Connect(BlockA, "Out2", BlockC, "In");
        graph.Connect(BlockB, "Out", BlockC, "In");

        var outgoing = graph.GetOutgoing(BlockA).ToList();

        Assert.Equal(2, outgoing.Count);
    }

    [Fact]
    public void GetIncoming_ReturnsTargetConnections()
    {
        var graph = new ConnectionGraph();
        graph.Connect(BlockA, "Out", BlockC, "In1");
        graph.Connect(BlockB, "Out", BlockC, "In2");

        var incoming = graph.GetIncoming(BlockC).ToList();

        Assert.Equal(2, incoming.Count);
    }
}
