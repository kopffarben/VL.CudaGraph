using System;
using ManagedCuda.BasicTypes;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// A graph node created by stream capture of library calls (cuBLAS, cuFFT, etc.).
/// Inserted into the CUDA graph as a child graph node via AddChildGraphNode.
/// Supports Recapture: re-execute the capture action and update the child graph
/// without a full Cold Rebuild.
/// </summary>
public sealed class CapturedNode : IGraphNode, IDisposable
{
    private bool _disposed;

    public Guid Id { get; }
    public string DebugName { get; set; }
    public CapturedNodeDescriptor Descriptor { get; }

    /// <summary>
    /// The capture action executed during stream capture.
    /// Receives the stream handle and buffer bindings (one CUdeviceptr per descriptor param).
    /// Order: inputs first, then outputs, then scalars.
    /// </summary>
    internal Action<CUstream, CUdeviceptr[]> CaptureAction { get; }

    /// <summary>
    /// Buffer bindings set by CudaEngine before Capture(). Order matches descriptor:
    /// [inputs..., outputs..., scalars...].
    /// </summary>
    internal CUdeviceptr[] BufferBindings { get; set; }

    /// <summary>
    /// The most recently captured child graph handle. Replaced on Recapture.
    /// The CapturedNode owns this handle and destroys it on Dispose/Recapture.
    /// </summary>
    internal CUgraph CapturedGraph { get; private set; }

    /// <summary>
    /// Whether this node has been captured at least once.
    /// </summary>
    internal bool HasCapturedGraph => CapturedGraph.Pointer != IntPtr.Zero;

    public CapturedNode(
        CapturedNodeDescriptor descriptor,
        Action<CUstream, CUdeviceptr[]> captureAction,
        string? debugName = null)
    {
        Id = Guid.NewGuid();
        Descriptor = descriptor ?? throw new ArgumentNullException(nameof(descriptor));
        CaptureAction = captureAction ?? throw new ArgumentNullException(nameof(captureAction));
        DebugName = debugName ?? descriptor.DebugName;
        BufferBindings = new CUdeviceptr[descriptor.TotalParamCount];
    }

    /// <summary>
    /// Execute the capture action via stream capture and produce a child graph.
    /// Disposes the previous captured graph if any.
    /// </summary>
    internal CUgraph Capture(CUstream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // Destroy previous captured graph
        if (CapturedGraph.Pointer != IntPtr.Zero)
        {
            StreamCaptureHelper.DestroyGraph(CapturedGraph);
            CapturedGraph = default;
        }

        CapturedGraph = StreamCaptureHelper.CaptureToGraph(stream, s => CaptureAction(s, BufferBindings));
        return CapturedGraph;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (CapturedGraph.Pointer != IntPtr.Zero)
        {
            StreamCaptureHelper.DestroyGraph(CapturedGraph);
            CapturedGraph = default;
        }
    }
}
