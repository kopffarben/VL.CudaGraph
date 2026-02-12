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
/// Supports both KernelNodes (PTX kernels) and CapturedNodes (stream-captured library calls).
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

        // 2. Topological sort (Kahn's algorithm) — kernel nodes only
        var sortedNodes = desc.Nodes.Count > 0
            ? TopologicalSort(desc.Nodes, desc.Edges)
            : new List<KernelNode>();

        // 3. Allocate intermediate buffers for edges without external buffers
        var intermediateBuffers = new List<GpuBuffer<byte>>();
        var edgeBuffers = desc.Nodes.Count > 0
            ? AllocateIntermediateBuffers(desc, intermediateBuffers)
            : new Dictionary<(Guid, int), CUdeviceptr>();

        // 4. Wire all buffer pointers into kernel node params
        if (desc.Nodes.Count > 0)
            WireParameters(desc, edgeBuffers, nodeMap);

        // 5. Build CUgraph
        var graph = new ManagedCuda.CudaGraph();
        var graphNodeHandles = new Dictionary<Guid, CUgraphNode>();
        var memsetNodeHandles = new Dictionary<Guid, CUgraphNode>();
        var capturedNodeHandles = new Dictionary<Guid, CUgraphNode>();

        // 5a. Insert memset nodes (e.g., AppendBuffer counter resets)
        foreach (var memset in desc.MemsetDescriptors)
        {
            var memsetParams = new CudaMemsetNodeParams
            {
                dst = memset.Destination,
                pitch = 0,
                value = memset.Value,
                elementSize = memset.ElementSize,
                width = (ManagedCuda.BasicTypes.SizeT)memset.Width,
                height = 1,
            };

            var memsetNode = graph.AddMemsetNode(null, memsetParams, _device.Context);
            memsetNodeHandles[memset.Id] = memsetNode;
        }

        // Build a lookup: kernelNodeId → list of memset CUgraphNodes it depends on
        var memsetDepsForKernel = new Dictionary<Guid, List<CUgraphNode>>();
        foreach (var memset in desc.MemsetDescriptors)
        {
            if (!memsetNodeHandles.TryGetValue(memset.Id, out var memsetHandle)) continue;
            foreach (var kernelId in memset.DependentKernelNodeIds)
            {
                if (!memsetDepsForKernel.TryGetValue(kernelId, out var list))
                {
                    list = new List<CUgraphNode>();
                    memsetDepsForKernel[kernelId] = list;
                }
                list.Add(memsetHandle);
            }
        }

        // 5b. Capture and insert CapturedNodes as child graph nodes
        using var captureStream = new CudaStream();
        foreach (var capturedNode in desc.CapturedNodes)
        {
            // Perform stream capture
            var childGraph = capturedNode.Capture(captureStream.Stream);

            // Collect dependencies for this captured node from CapturedDependencies
            var deps = new List<CUgraphNode>();
            foreach (var dep in desc.CapturedDependencies)
            {
                if (dep.TargetNodeId == capturedNode.Id)
                {
                    // Source must complete before this captured node
                    if (graphNodeHandles.TryGetValue(dep.SourceNodeId, out var srcHandle))
                        deps.Add(srcHandle);
                    if (capturedNodeHandles.TryGetValue(dep.SourceNodeId, out var capSrcHandle))
                        deps.Add(capSrcHandle);
                }
            }

            CUgraphNode[]? depArray = deps.Count > 0 ? deps.ToArray() : null;
            var childNode = StreamCaptureHelper.AddChildGraphNode(graph.Graph, depArray, childGraph);
            capturedNodeHandles[capturedNode.Id] = childNode;
        }

        // 5c. Build dependency map for kernel nodes: for each node, collect the CUDA graph nodes it depends on
        var nodeDependencies = BuildDependencyMap(desc.Edges, sortedNodes);

        foreach (var node in sortedNodes)
        {
            var nodeParams = node.BuildNodeParams();

            // Collect all dependencies: edge-based + memset-based + captured-node-based
            var allDeps = new List<CUgraphNode>();

            if (nodeDependencies.TryGetValue(node.Id, out var depIds) && depIds.Count > 0)
            {
                allDeps.AddRange(depIds
                    .Where(id => graphNodeHandles.ContainsKey(id))
                    .Select(id => graphNodeHandles[id]));
            }

            if (memsetDepsForKernel.TryGetValue(node.Id, out var memsetDeps))
            {
                allDeps.AddRange(memsetDeps);
            }

            // Check CapturedDependencies: captured node → this kernel node
            foreach (var dep in desc.CapturedDependencies)
            {
                if (dep.TargetNodeId == node.Id && capturedNodeHandles.TryGetValue(dep.SourceNodeId, out var capHandle))
                {
                    allDeps.Add(capHandle);
                }
            }

            CUgraphNode[]? depsArr = allDeps.Count > 0 ? allDeps.ToArray() : null;

            var graphNode = graph.AddKernelNode(depsArr, nodeParams);
            graphNodeHandles[node.Id] = graphNode;
        }

        // 6. Instantiate
        var exec = graph.Instantiate(CUgraphInstantiate_flags.None);

        return new CompiledGraph(exec, graph, graphNodeHandles, nodeMap, intermediateBuffers,
            memsetNodeHandles, capturedNodeHandles);
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
            var sourceNode = nodeMap[sourceNodeId];
            var sourceParam = sourceNode.LoadedKernel.Descriptor.Parameters[sourceParamIndex];

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
    /// Wire buffer pointers into kernel node parameters.
    /// </summary>
    private void WireParameters(
        GraphDescription desc,
        Dictionary<(Guid, int), CUdeviceptr> edgeBuffers,
        Dictionary<Guid, KernelNode> nodeMap)
    {
        // First, set all external buffers (for kernel nodes only)
        foreach (var kvp in desc.ExternalBuffers)
        {
            var (nodeId, paramIndex) = kvp.Key;
            if (nodeMap.TryGetValue(nodeId, out var node))
                node.SetPointer(paramIndex, kvp.Value);
        }

        // Then, wire edge buffers
        foreach (var edge in desc.Edges)
        {
            var key = (edge.SourceNodeId, edge.SourceParamIndex);
            if (edgeBuffers.TryGetValue(key, out var ptr))
            {
                if (!desc.ExternalBuffers.ContainsKey(key))
                    nodeMap[edge.SourceNodeId].SetPointer(edge.SourceParamIndex, ptr);

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
