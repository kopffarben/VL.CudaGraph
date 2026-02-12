using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.Graph;
using Xunit;

namespace VL.Cuda.Tests.Graph;

public class StreamCaptureTests : IDisposable
{
    private readonly DeviceContext _device = new(0);

    [Fact]
    public void CaptureToGraph_EmptyWork_ProducesValidGraph()
    {
        using var stream = new CudaStream();

        // Empty work — should produce a valid (empty) graph
        var graph = StreamCaptureHelper.CaptureToGraph(stream.Stream, s => { });

        Assert.NotEqual(IntPtr.Zero, graph.Pointer);

        StreamCaptureHelper.DestroyGraph(graph);
    }

    [Fact]
    public void CaptureToGraph_WithMemset_ProducesGraph()
    {
        using var stream = new CudaStream();
        using var buf = GpuBuffer<uint>.Allocate(_device, 16);

        var graph = StreamCaptureHelper.CaptureToGraph(stream.Stream, s =>
        {
            // Async memset during capture — gets recorded into the graph
            DriverAPINativeMethods.MemsetAsync.cuMemsetD32Async(buf.Pointer, 42u, 16, s);
        });

        Assert.NotEqual(IntPtr.Zero, graph.Pointer);

        StreamCaptureHelper.DestroyGraph(graph);
    }

    [Fact]
    public void CaptureToGraph_NullWork_Throws()
    {
        using var stream = new CudaStream();

        Assert.Throws<ArgumentNullException>(() =>
            StreamCaptureHelper.CaptureToGraph(stream.Stream, null!));
    }

    [Fact]
    public void DestroyGraph_ZeroPointer_DoesNotThrow()
    {
        // Destroying a zero-pointer graph should be safe
        StreamCaptureHelper.DestroyGraph(default);
    }

    [Fact]
    public void AddChildGraphNode_InsertsIntoParentGraph()
    {
        using var stream = new CudaStream();
        using var buf = GpuBuffer<uint>.Allocate(_device, 4);

        // Create child graph via capture
        var childGraph = StreamCaptureHelper.CaptureToGraph(stream.Stream, s =>
        {
            DriverAPINativeMethods.MemsetAsync.cuMemsetD32Async(buf.Pointer, 99u, 4, s);
        });

        try
        {
            // Create parent graph and add child as node
            using var parentGraph = new CudaGraph();
            var childNode = StreamCaptureHelper.AddChildGraphNode(parentGraph.Graph, null, childGraph);

            Assert.NotEqual(IntPtr.Zero, childNode.Pointer);

            // Instantiate and launch
            using var exec = parentGraph.Instantiate(CUgraphInstantiate_flags.None);
            exec.Launch(stream);
            stream.Synchronize();

            var result = buf.Download();
            Assert.Equal(99u, result[0]);
        }
        finally
        {
            StreamCaptureHelper.DestroyGraph(childGraph);
        }
    }

    [Fact]
    public void CapturedNode_Capture_ProducesGraph()
    {
        var desc = new CapturedNodeDescriptor("test",
            outputs: new[] { CapturedParam.Pointer("out", "uint*") });

        using var buf = GpuBuffer<uint>.Allocate(_device, 4);
        var bufPtr = buf.Pointer;

        var node = new CapturedNode(desc, (s, buffers) =>
        {
            DriverAPINativeMethods.MemsetAsync.cuMemsetD32Async(bufPtr, 77u, 4, s);
        });

        try
        {
            using var stream = new CudaStream();
            var graph = node.Capture(stream.Stream);

            Assert.True(node.HasCapturedGraph);
            Assert.NotEqual(IntPtr.Zero, graph.Pointer);
        }
        finally
        {
            node.Dispose();
        }
    }

    [Fact]
    public void CapturedNode_Recapture_DestroysOldGraph()
    {
        var desc = new CapturedNodeDescriptor("test");
        var captureCount = 0;

        var node = new CapturedNode(desc, (s, buffers) =>
        {
            captureCount++;
        });

        try
        {
            using var stream = new CudaStream();

            node.Capture(stream.Stream);
            Assert.Equal(1, captureCount);
            Assert.True(node.HasCapturedGraph);

            // Recapture — should destroy old graph and capture new
            node.Capture(stream.Stream);
            Assert.Equal(2, captureCount);
            Assert.True(node.HasCapturedGraph);
        }
        finally
        {
            node.Dispose();
        }
    }

    public void Dispose()
    {
        _device.Dispose();
    }
}
