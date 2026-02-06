using System;
using System.Collections.Generic;
using System.Linq;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Device;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// Compiles a graph description into an executable CUDA graph.
/// Phases: validate → topological sort → allocate intermediates → build CUgraph → instantiate.
/// </summary>
public sealed class GraphCompiler
{
    private readonly DeviceContext _device;
    private readonly BufferPool _pool;

    public GraphCompiler(DeviceContext device, BufferPool pool)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));
        _pool = pool ?? throw new ArgumentNullException(nameof(pool));
    }

    /// <summary>
    /// Compile a GraphBuilder into a CompiledGraph (CUgraphExec).
    /// </summary>
    public CompiledGraph Compile(GraphBuilder builder)
    {
        // 1. Validate
        var validation = builder.Validate();
        if (!validation.IsValid)
            throw new InvalidOperationException($"Graph validation failed: {validation}");

        var desc = builder.Build();
        var nodeMap = desc.Nodes.ToDictionary(n => n.Id);

        // 2. Topological sort (Kahn's algorithm)
        var sortedNodes = TopologicalSort(desc.Nodes, desc.Edges);

        // 3. Allocate intermediate buffers for edges without external buffers
        var intermediateBuffers = new List<GpuBuffer<byte>>();
        var edgeBuffers = AllocateIntermediateBuffers(desc, intermediateBuffers);

        // 4. Wire all buffer pointers into kernel node params
        WireParameters(desc, edgeBuffers, nodeMap);

        // 5. Build CUgraph
        var graph = new ManagedCuda.CudaGraph();
        var graphNodeHandles = new Dictionary<Guid, CUgraphNode>();

        // Build dependency map: for each node, collect the CUDA graph nodes it depends on
        var nodeDependencies = BuildDependencyMap(desc.Edges, sortedNodes);

        foreach (var node in sortedNodes)
        {
            var nodeParams = node.BuildNodeParams();

            // Get the CUDA graph node handles this node depends on
            CUgraphNode[]? deps = null;
            if (nodeDependencies.TryGetValue(node.Id, out var depIds) && depIds.Count > 0)
            {
                deps = depIds
                    .Where(id => graphNodeHandles.ContainsKey(id))
                    .Select(id => graphNodeHandles[id])
                    .ToArray();
                if (deps.Length == 0) deps = null;
            }

            var graphNode = graph.AddKernelNode(deps, nodeParams);
            graphNodeHandles[node.Id] = graphNode;
        }

        // 6. Instantiate
        var exec = graph.Instantiate(CUgraphInstantiate_flags.None);

        return new CompiledGraph(exec, graph, graphNodeHandles, nodeMap, intermediateBuffers);
    }

    /// <summary>
    /// Topological sort using Kahn's algorithm.
    /// </summary>
    private List<KernelNode> TopologicalSort(IReadOnlyList<KernelNode> nodes, IReadOnlyList<Edge> edges)
    {
        var inDegree = new Dictionary<Guid, int>();
        var adjacency = new Dictionary<Guid, HashSet<Guid>>();

        foreach (var node in nodes)
        {
            inDegree[node.Id] = 0;
            adjacency[node.Id] = new HashSet<Guid>();
        }

        foreach (var edge in edges)
        {
            if (adjacency[edge.SourceNodeId].Add(edge.TargetNodeId))
                inDegree[edge.TargetNodeId]++;
        }

        var queue = new Queue<Guid>();
        foreach (var kvp in inDegree)
        {
            if (kvp.Value == 0)
                queue.Enqueue(kvp.Key);
        }

        var nodeMap = nodes.ToDictionary(n => n.Id);
        var sorted = new List<KernelNode>();

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            sorted.Add(nodeMap[current]);

            foreach (var neighbor in adjacency[current])
            {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0)
                    queue.Enqueue(neighbor);
            }
        }

        return sorted;
    }

    /// <summary>
    /// Build a map of node → set of nodes it depends on (predecessors).
    /// </summary>
    private Dictionary<Guid, HashSet<Guid>> BuildDependencyMap(
        IReadOnlyList<Edge> edges, List<KernelNode> sortedNodes)
    {
        var deps = new Dictionary<Guid, HashSet<Guid>>();
        foreach (var node in sortedNodes)
            deps[node.Id] = new HashSet<Guid>();

        foreach (var edge in edges)
        {
            deps[edge.TargetNodeId].Add(edge.SourceNodeId);
        }

        return deps;
    }

    /// <summary>
    /// Allocate intermediate buffers for edges that don't have external buffers.
    /// Returns a map from (sourceNodeId, sourceParamIndex) → buffer pointer.
    /// </summary>
    private Dictionary<(Guid, int), CUdeviceptr> AllocateIntermediateBuffers(
        GraphDescription desc, List<GpuBuffer<byte>> intermediateBuffers)
    {
        var edgeBuffers = new Dictionary<(Guid, int), CUdeviceptr>();

        // Group edges by source output — multiple edges from the same source output share one buffer
        var edgesBySource = desc.Edges
            .GroupBy(e => (e.SourceNodeId, e.SourceParamIndex))
            .ToList();

        var nodeMap = desc.Nodes.ToDictionary(n => n.Id);

        foreach (var group in edgesBySource)
        {
            var (sourceNodeId, sourceParamIndex) = group.Key;

            // Check if source already has an external buffer
            if (desc.ExternalBuffers.ContainsKey((sourceNodeId, sourceParamIndex)))
            {
                edgeBuffers[(sourceNodeId, sourceParamIndex)] = desc.ExternalBuffers[(sourceNodeId, sourceParamIndex)];
                continue;
            }

            // Also check if any target already has an external buffer we should use
            // (This handles the case where the output buffer is externally managed)
            bool foundExternal = false;
            foreach (var edge in group)
            {
                if (desc.ExternalBuffers.TryGetValue((edge.TargetNodeId, edge.TargetParamIndex), out var extPtr))
                {
                    edgeBuffers[(sourceNodeId, sourceParamIndex)] = extPtr;
                    foundExternal = true;
                    break;
                }
            }
            if (foundExternal) continue;

            // Need to allocate an intermediate buffer
            // Determine size from the source node's parameter type
            var sourceNode = nodeMap[sourceNodeId];
            var sourceParam = sourceNode.LoadedKernel.Descriptor.Parameters[sourceParamIndex];

            // Use the grid dimensions to estimate element count.
            // For now, we use GridDimX * BlockDimX as the element count (1D assumption).
            int elementCount = (int)(sourceNode.GridDimX * sourceNode.BlockDimX);
            int elementSize = GetElementSize(sourceParam.Type);
            int totalBytes = elementCount * elementSize;

            if (totalBytes <= 0)
                totalBytes = 1024; // fallback minimum

            var buffer = _pool.Acquire<byte>(totalBytes);
            intermediateBuffers.Add(buffer);
            edgeBuffers[(sourceNodeId, sourceParamIndex)] = buffer.Pointer;
        }

        return edgeBuffers;
    }

    /// <summary>
    /// Wire buffer pointers into kernel node parameters:
    /// - External buffers → set directly
    /// - Edge buffers → set source output and target input to same pointer
    /// </summary>
    private void WireParameters(
        GraphDescription desc,
        Dictionary<(Guid, int), CUdeviceptr> edgeBuffers,
        Dictionary<Guid, KernelNode> nodeMap)
    {
        // First, set all external buffers
        foreach (var kvp in desc.ExternalBuffers)
        {
            var (nodeId, paramIndex) = kvp.Key;
            nodeMap[nodeId].SetPointer(paramIndex, kvp.Value);
        }

        // Then, wire edge buffers: source output and all target inputs share the same pointer
        foreach (var edge in desc.Edges)
        {
            var key = (edge.SourceNodeId, edge.SourceParamIndex);
            if (edgeBuffers.TryGetValue(key, out var ptr))
            {
                // Set source output (if not already set by external buffer)
                if (!desc.ExternalBuffers.ContainsKey(key))
                    nodeMap[edge.SourceNodeId].SetPointer(edge.SourceParamIndex, ptr);

                // Set target input (if not already set by external buffer)
                var targetKey = (edge.TargetNodeId, edge.TargetParamIndex);
                if (!desc.ExternalBuffers.ContainsKey(targetKey))
                    nodeMap[edge.TargetNodeId].SetPointer(edge.TargetParamIndex, ptr);
            }
        }
    }

    private static int GetElementSize(string typeStr)
    {
        var baseType = typeStr.TrimEnd('*', ' ').ToLowerInvariant();
        return baseType switch
        {
            "float" or "f32" or "int" or "int32" or "s32" or "uint" or "uint32" or "u32" => 4,
            "double" or "f64" or "long" or "int64" or "s64" or "ulong" or "uint64" or "u64" => 8,
            "short" or "int16" or "s16" or "ushort" or "uint16" or "u16" => 2,
            "byte" or "uint8" or "u8" or "sbyte" or "int8" or "s8" => 1,
            _ => 4, // default to 4 bytes
        };
    }
}
