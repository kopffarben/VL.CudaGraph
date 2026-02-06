using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Buffers;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// Wraps a CudaGraphExec (instantiated CUDA graph). Provides launch and
/// in-place parameter updates (Hot/Warm) without graph rebuild.
/// </summary>
public sealed class CompiledGraph : IDisposable
{
    private readonly CudaGraphExec _exec;
    private readonly ManagedCuda.CudaGraph _graph;
    private readonly Dictionary<Guid, CUgraphNode> _nodeHandles;
    private readonly Dictionary<Guid, KernelNode> _kernelNodes;
    private readonly List<GpuBuffer<byte>> _intermediateBuffers;
    private bool _disposed;

    internal CompiledGraph(
        CudaGraphExec exec,
        ManagedCuda.CudaGraph graph,
        Dictionary<Guid, CUgraphNode> nodeHandles,
        Dictionary<Guid, KernelNode> kernelNodes,
        List<GpuBuffer<byte>> intermediateBuffers)
    {
        _exec = exec;
        _graph = graph;
        _nodeHandles = nodeHandles;
        _kernelNodes = kernelNodes;
        _intermediateBuffers = intermediateBuffers;
    }

    /// <summary>
    /// Launch the graph on a stream.
    /// </summary>
    public void Launch(CudaStream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _exec.Launch(stream);
    }

    /// <summary>
    /// Launch the graph and synchronize (block until complete).
    /// </summary>
    public void LaunchAndSync(CudaStream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _exec.Launch(stream);
        stream.Synchronize();
    }

    /// <summary>
    /// Hot Update: change a scalar parameter value in-place (~0 cost).
    /// </summary>
    public unsafe void UpdateScalar<T>(Guid nodeId, int paramIndex, T value) where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_kernelNodes.TryGetValue(nodeId, out var node))
            throw new ArgumentException($"Node {nodeId} not found in compiled graph");

        // Write new value into the existing native slot
        node.SetScalar(paramIndex, value);

        // Rebuild params struct and update the executable graph
        var nodeParams = node.BuildNodeParams();
        var graphNode = _nodeHandles[nodeId];
        _exec.SetParams(graphNode, ref nodeParams);
    }

    /// <summary>
    /// Warm Update: change a pointer parameter (buffer swap).
    /// </summary>
    public void UpdatePointer(Guid nodeId, int paramIndex, CUdeviceptr newPointer)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_kernelNodes.TryGetValue(nodeId, out var node))
            throw new ArgumentException($"Node {nodeId} not found in compiled graph");

        node.SetPointer(paramIndex, newPointer);

        var nodeParams = node.BuildNodeParams();
        var graphNode = _nodeHandles[nodeId];
        _exec.SetParams(graphNode, ref nodeParams);
    }

    /// <summary>
    /// Warm Update: change grid dimensions for a node.
    /// </summary>
    public void UpdateGrid(Guid nodeId, uint gridX, uint gridY = 1, uint gridZ = 1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_kernelNodes.TryGetValue(nodeId, out var node))
            throw new ArgumentException($"Node {nodeId} not found in compiled graph");

        node.GridDimX = gridX;
        node.GridDimY = gridY;
        node.GridDimZ = gridZ;

        var nodeParams = node.BuildNodeParams();
        var graphNode = _nodeHandles[nodeId];
        _exec.SetParams(graphNode, ref nodeParams);
    }

    internal CudaGraphExec Exec => _exec;
    internal IReadOnlyDictionary<Guid, CUgraphNode> NodeHandles => _nodeHandles;
    internal IReadOnlyList<GpuBuffer<byte>> IntermediateBuffers => _intermediateBuffers;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _exec.Dispose();
        _graph.Dispose();

        foreach (var buf in _intermediateBuffers)
            buf.Dispose();
        _intermediateBuffers.Clear();
    }
}
