using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// Thin helper for CUDA stream capture. Executes a work action between
/// cuStreamBeginCapture and cuStreamEndCapture, returning the captured graph handle.
/// </summary>
internal static class StreamCaptureHelper
{
    /// <summary>
    /// Execute a work action in stream capture mode and return the captured CUgraph handle.
    /// The work action receives the stream handle and should direct all library calls to it.
    /// The caller owns the returned CUgraph and must destroy it when done.
    /// </summary>
    public static CUgraph CaptureToGraph(CUstream stream, Action<CUstream> work)
    {
        if (work == null) throw new ArgumentNullException(nameof(work));

        // Begin capture in Relaxed mode (allows multi-threaded capture)
        var res = DriverAPINativeMethods.Streams.cuStreamBeginCapture(stream, CUstreamCaptureMode.Relaxed);
        if (res != CUResult.Success)
            throw new CudaException(res);

        try
        {
            work(stream);
        }
        catch
        {
            // If work fails, we must still end capture to leave the stream in a valid state.
            CUgraph discardGraph = new();
            DriverAPINativeMethods.Streams.cuStreamEndCapture(stream, ref discardGraph);
            if (discardGraph.Pointer != IntPtr.Zero)
                DriverAPINativeMethods.GraphManagment.cuGraphDestroy(discardGraph);
            throw;
        }

        CUgraph graphHandle = new();
        res = DriverAPINativeMethods.Streams.cuStreamEndCapture(stream, ref graphHandle);
        if (res != CUResult.Success)
            throw new CudaException(res);

        return graphHandle;
    }

    /// <summary>
    /// Destroy a CUgraph handle that was returned from CaptureToGraph.
    /// </summary>
    public static void DestroyGraph(CUgraph graph)
    {
        if (graph.Pointer != IntPtr.Zero)
        {
            DriverAPINativeMethods.GraphManagment.cuGraphDestroy(graph);
        }
    }

    /// <summary>
    /// Add a child graph node to a parent graph using a captured CUgraph.
    /// </summary>
    public static CUgraphNode AddChildGraphNode(CUgraph parentGraph, CUgraphNode[]? dependencies, CUgraph childGraph)
    {
        CUgraphNode node = new();
        SizeT numDeps = dependencies?.Length ?? 0;

        var res = DriverAPINativeMethods.GraphManagment.cuGraphAddChildGraphNode(
            ref node, parentGraph, dependencies, numDeps, childGraph);
        if (res != CUResult.Success)
            throw new CudaException(res);

        return node;
    }

    /// <summary>
    /// Update a child graph node in an executable graph (Recapture update).
    /// </summary>
    public static void UpdateChildGraphNode(CUgraphExec exec, CUgraphNode node, CUgraph newChildGraph)
    {
        var res = DriverAPINativeMethods.GraphManagment.cuGraphExecChildGraphNodeSetParams(
            exec, node, newChildGraph);
        if (res != CUResult.Success)
            throw new CudaException(res);
    }
}
