using System;
using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Context.Services;
using VL.Cuda.Tests.Helpers;
using Xunit;

namespace VL.Cuda.Tests.Context;

public class BlockRegistryTests
{
    [Fact]
    public void Register_IncreasesCount()
    {
        var registry = new BlockRegistry();
        var block = new TestBlock("A");

        registry.Register(block);

        Assert.Equal(1, registry.Count);
    }

    [Fact]
    public void Register_Multiple_TracksAll()
    {
        var registry = new BlockRegistry();
        registry.Register(new TestBlock("A"));
        registry.Register(new TestBlock("B"));
        registry.Register(new TestBlock("C"));

        Assert.Equal(3, registry.Count);
    }

    [Fact]
    public void Unregister_DecreasesCount()
    {
        var registry = new BlockRegistry();
        var block = new TestBlock("A");
        registry.Register(block);

        var removed = registry.Unregister(block.Id);

        Assert.True(removed);
        Assert.Equal(0, registry.Count);
    }

    [Fact]
    public void Unregister_UnknownId_ReturnsFalse()
    {
        var registry = new BlockRegistry();
        Assert.False(registry.Unregister(Guid.NewGuid()));
    }

    [Fact]
    public void Get_ReturnsRegisteredBlock()
    {
        var registry = new BlockRegistry();
        var block = new TestBlock("A");
        registry.Register(block);

        var result = registry.Get(block.Id);
        Assert.Same(block, result);
    }

    [Fact]
    public void Get_UnknownId_ReturnsNull()
    {
        var registry = new BlockRegistry();
        Assert.Null(registry.Get(Guid.NewGuid()));
    }

    [Fact]
    public void Register_FiresStructureChanged()
    {
        var registry = new BlockRegistry();
        int fireCount = 0;
        registry.StructureChanged += () => fireCount++;

        registry.Register(new TestBlock("A"));

        Assert.Equal(1, fireCount);
    }

    [Fact]
    public void Unregister_FiresStructureChanged()
    {
        var registry = new BlockRegistry();
        var block = new TestBlock("A");
        registry.Register(block);

        int fireCount = 0;
        registry.StructureChanged += () => fireCount++;

        registry.Unregister(block.Id);

        Assert.Equal(1, fireCount);
    }

    [Fact]
    public void Unregister_UnknownId_DoesNotFire()
    {
        var registry = new BlockRegistry();
        int fireCount = 0;
        registry.StructureChanged += () => fireCount++;

        registry.Unregister(Guid.NewGuid());

        Assert.Equal(0, fireCount);
    }

    [Fact]
    public void All_ReturnsAllBlocks()
    {
        var registry = new BlockRegistry();
        var a = new TestBlock("A");
        var b = new TestBlock("B");
        registry.Register(a);
        registry.Register(b);

        Assert.Contains(a, registry.All);
        Assert.Contains(b, registry.All);
    }
}
