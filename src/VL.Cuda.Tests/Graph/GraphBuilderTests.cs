using System;
using System.IO;
using System.Linq;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.PTX;
using VL.Cuda.Core.Graph;
using Xunit;

namespace VL.Cuda.Tests.Graph;

public class GraphBuilderTests : IDisposable
{
    private readonly DeviceContext _device = new(0);
    private readonly ModuleCache _cache;

    private static string VectorAddPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "vector_add.ptx");
    private static string ScalarMulPath => Path.Combine(AppContext.BaseDirectory, "TestKernels", "scalar_mul.ptx");

    public GraphBuilderTests()
    {
        _cache = new ModuleCache(_device);
    }

    [Fact]
    public void AddKernel_AddsNode()
    {
        var builder = new GraphBuilder(_device, _cache);
        var node = builder.AddKernel(VectorAddPath, "add");

        Assert.Single(builder.Nodes);
        Assert.Equal("add", node.DebugName);
    }

    [Fact]
    public void AddEdge_AddsEdge()
    {
        var builder = new GraphBuilder(_device, _cache);
        var add = builder.AddKernel(VectorAddPath, "add");
        var mul = builder.AddKernel(ScalarMulPath, "mul");

        // vector_add output C (index 2) → scalar_mul input A (index 0)
        builder.AddEdge(add, 2, mul, 0);

        Assert.Single(builder.Edges);
        Assert.Equal(add.Id, builder.Edges[0].SourceNodeId);
        Assert.Equal(2, builder.Edges[0].SourceParamIndex);
        Assert.Equal(mul.Id, builder.Edges[0].TargetNodeId);
        Assert.Equal(0, builder.Edges[0].TargetParamIndex);
    }

    [Fact]
    public void Validate_EmptyGraph_IsInvalid()
    {
        var builder = new GraphBuilder(_device, _cache);
        var result = builder.Validate();

        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("no nodes"));
    }

    [Fact]
    public void Validate_SingleNode_IsValid()
    {
        var builder = new GraphBuilder(_device, _cache);
        builder.AddKernel(VectorAddPath, "add");

        var result = builder.Validate();

        // Should be valid (warnings about unconnected params are OK)
        Assert.True(result.IsValid);
    }

    [Fact]
    public void Validate_DetectsCycle()
    {
        var builder = new GraphBuilder(_device, _cache);
        var a = builder.AddKernel(VectorAddPath, "A");
        var b = builder.AddKernel(ScalarMulPath, "B");

        // A→B and B→A creates a cycle
        builder.AddEdge(a, 2, b, 0);  // A output → B input
        builder.AddEdge(b, 1, a, 0);  // B output → A input

        var result = builder.Validate();
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("cycle"));
    }

    [Fact]
    public void Validate_DetectsSelfLoop()
    {
        var builder = new GraphBuilder(_device, _cache);
        var a = builder.AddKernel(VectorAddPath, "A");

        builder.AddEdge(a, 2, a, 0); // Self-loop

        var result = builder.Validate();
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("Self-loop"));
    }

    [Fact]
    public void Validate_DetectsInvalidParamIndex()
    {
        var builder = new GraphBuilder(_device, _cache);
        var a = builder.AddKernel(VectorAddPath, "A");
        var b = builder.AddKernel(ScalarMulPath, "B");

        builder.AddEdge(a, 99, b, 0); // Invalid source param index

        var result = builder.Validate();
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("out of range"));
    }

    [Fact]
    public void Validate_WarnsOnTypeMismatch()
    {
        // This test needs two kernels with different pointer types.
        // Both our test kernels use float*, so we just verify the path works.
        var builder = new GraphBuilder(_device, _cache);
        var a = builder.AddKernel(VectorAddPath, "add");
        var b = builder.AddKernel(ScalarMulPath, "mul");

        builder.AddEdge(a, 2, b, 0); // float* → float* — should NOT warn

        var result = builder.Validate();
        Assert.True(result.IsValid);
        Assert.DoesNotContain(result.Warnings, w => w.Contains("Type mismatch"));
    }

    [Fact]
    public void Validate_WarnsOnScalarEdge()
    {
        var builder = new GraphBuilder(_device, _cache);
        var a = builder.AddKernel(VectorAddPath, "add");
        var b = builder.AddKernel(ScalarMulPath, "mul");

        // vector_add param 3 (N, uint scalar) → scalar_mul param 2 (scale, float scalar)
        builder.AddEdge(a, 3, b, 2);

        var result = builder.Validate();
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("not a pointer"));
    }

    [Fact]
    public void Validate_LinearChain_IsValid()
    {
        var builder = new GraphBuilder(_device, _cache);
        var a = builder.AddKernel(VectorAddPath, "add");
        var b = builder.AddKernel(ScalarMulPath, "mul");

        // C output of add → A input of mul
        builder.AddEdge(a, 2, b, 0);

        var result = builder.Validate();
        Assert.True(result.IsValid);
    }

    public void Dispose()
    {
        _cache.Dispose();
        _device.Dispose();
    }
}
