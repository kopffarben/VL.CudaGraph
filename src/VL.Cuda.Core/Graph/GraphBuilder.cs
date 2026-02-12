using System;
using System.Collections.Generic;
using System.Linq;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.PTX;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// Fluent API for constructing a CUDA graph description before compilation.
/// Supports both KernelNodes (PTX kernels) and CapturedNodes (stream-captured library calls).
/// </summary>
public sealed class GraphBuilder
{
    private readonly DeviceContext _device;
    private readonly ModuleCache _moduleCache;
    private readonly List<KernelNode> _nodes = new();
    private readonly List<CapturedNode> _capturedNodes = new();
    private readonly List<Edge> _edges = new();
    private readonly Dictionary<(Guid NodeId, int ParamIndex), CUdeviceptr> _externalBuffers = new();
    private readonly List<MemsetDescriptor> _memsetDescriptors = new();
    private readonly List<CapturedDependency> _capturedDependencies = new();

    public GraphBuilder(DeviceContext device, ModuleCache moduleCache)
    {
        _device = device ?? throw new ArgumentNullException(nameof(device));
        _moduleCache = moduleCache ?? throw new ArgumentNullException(nameof(moduleCache));
    }

    /// <summary>
    /// Add a kernel node from a PTX file path. Uses ModuleCache for deduplication.
    /// </summary>
    public KernelNode AddKernel(string ptxPath, string? debugName = null)
    {
        var loaded = _moduleCache.GetOrLoad(ptxPath);
        var node = new KernelNode(loaded, debugName);
        _nodes.Add(node);
        return node;
    }

    /// <summary>
    /// Add a kernel node from an already-loaded kernel.
    /// </summary>
    public KernelNode AddKernel(LoadedKernel loaded, string? debugName = null)
    {
        var node = new KernelNode(loaded, debugName);
        _nodes.Add(node);
        return node;
    }

    /// <summary>
    /// Add a captured library operation node. Created via stream capture during graph compilation.
    /// </summary>
    public CapturedNode AddCaptured(CapturedNodeDescriptor descriptor, Action<CUstream, CUdeviceptr[]> captureAction, string? debugName = null)
    {
        var node = new CapturedNode(descriptor, captureAction, debugName);
        _capturedNodes.Add(node);
        return node;
    }

    /// <summary>
    /// Declare that a kernel node depends on a captured node completing first.
    /// </summary>
    public void AddCapturedDependency(CapturedNode captured, KernelNode kernel)
    {
        _capturedDependencies.Add(new CapturedDependency(captured.Id, kernel.Id));
    }

    /// <summary>
    /// Declare that a captured node depends on a kernel node completing first.
    /// </summary>
    public void AddCapturedDependency(KernelNode kernel, CapturedNode captured)
    {
        _capturedDependencies.Add(new CapturedDependency(kernel.Id, captured.Id));
    }

    /// <summary>
    /// Assign an external buffer to a specific captured node parameter.
    /// Uses the same external buffer dictionary as kernel nodes.
    /// </summary>
    public void SetExternalBuffer(CapturedNode node, int paramIndex, CUdeviceptr ptr)
    {
        _externalBuffers[(node.Id, paramIndex)] = ptr;
    }

    /// <summary>
    /// Add a data dependency edge between two nodes.
    /// The target node will depend on the source node completing first.
    /// </summary>
    public void AddEdge(KernelNode source, int sourceParam, KernelNode target, int targetParam)
    {
        _edges.Add(new Edge(source.Id, sourceParam, target.Id, targetParam));
    }

    /// <summary>
    /// Assign an external buffer to a specific node parameter.
    /// Used for graph inputs and outputs that are managed by the caller.
    /// </summary>
    public void SetExternalBuffer(KernelNode node, int paramIndex, CUdeviceptr ptr)
    {
        _externalBuffers[(node.Id, paramIndex)] = ptr;
    }

    /// <summary>
    /// Validate the graph description: cycle detection, type compatibility, completeness.
    /// </summary>
    public ValidationResult Validate()
    {
        var result = new ValidationResult();
        var nodeMap = _nodes.ToDictionary(n => n.Id);

        // Check for empty graph (no kernel nodes AND no captured nodes)
        if (_nodes.Count == 0 && _capturedNodes.Count == 0)
        {
            result.AddError("Graph has no nodes");
            return result;
        }

        // Validate edge references
        foreach (var edge in _edges)
        {
            if (!nodeMap.ContainsKey(edge.SourceNodeId))
                result.AddError($"Edge references unknown source node {edge.SourceNodeId}");
            if (!nodeMap.ContainsKey(edge.TargetNodeId))
                result.AddError($"Edge references unknown target node {edge.TargetNodeId}");

            if (nodeMap.TryGetValue(edge.SourceNodeId, out var src))
            {
                if (edge.SourceParamIndex < 0 || edge.SourceParamIndex >= src.ParameterCount)
                    result.AddError($"Edge source param index {edge.SourceParamIndex} out of range for node '{src.DebugName}' (has {src.ParameterCount} params)");
            }

            if (nodeMap.TryGetValue(edge.TargetNodeId, out var tgt))
            {
                if (edge.TargetParamIndex < 0 || edge.TargetParamIndex >= tgt.ParameterCount)
                    result.AddError($"Edge target param index {edge.TargetParamIndex} out of range for node '{tgt.DebugName}' (has {tgt.ParameterCount} params)");
            }

            if (edge.SourceNodeId == edge.TargetNodeId)
                result.AddError($"Self-loop detected on node {edge.SourceNodeId}");
        }

        // Type compatibility: source must be a pointer (output buffer) and target must be a pointer (input buffer)
        foreach (var edge in _edges)
        {
            if (nodeMap.TryGetValue(edge.SourceNodeId, out var srcNode) &&
                nodeMap.TryGetValue(edge.TargetNodeId, out var tgtNode))
            {
                // Skip type checks if indices are out of range (already reported above)
                if (edge.SourceParamIndex < 0 || edge.SourceParamIndex >= srcNode.ParameterCount ||
                    edge.TargetParamIndex < 0 || edge.TargetParamIndex >= tgtNode.ParameterCount)
                    continue;

                var srcParam = srcNode.LoadedKernel.Descriptor.Parameters[edge.SourceParamIndex];
                var tgtParam = tgtNode.LoadedKernel.Descriptor.Parameters[edge.TargetParamIndex];

                if (!srcParam.IsPointer)
                    result.AddError($"Edge source '{srcNode.DebugName}' param '{srcParam.Name}' is not a pointer — edges must connect buffer parameters");
                if (!tgtParam.IsPointer)
                    result.AddError($"Edge target '{tgtNode.DebugName}' param '{tgtParam.Name}' is not a pointer — edges must connect buffer parameters");

                // Type compat: check that underlying types match
                if (srcParam.IsPointer && tgtParam.IsPointer)
                {
                    var srcType = NormalizeType(srcParam.Type);
                    var tgtType = NormalizeType(tgtParam.Type);
                    if (srcType != tgtType)
                        result.AddWarning($"Type mismatch on edge: '{srcNode.DebugName}'.{srcParam.Name} ({srcParam.Type}) → '{tgtNode.DebugName}'.{tgtParam.Name} ({tgtParam.Type})");
                }
            }
        }

        // Cycle detection via Kahn's algorithm
        if (!DetectCyclesKahn(nodeMap, result))
        {
            // Error already added by DetectCyclesKahn
        }

        // Check that all pointer parameters are connected (either edge or external buffer)
        var connectedTargets = new HashSet<(Guid, int)>();
        foreach (var edge in _edges)
            connectedTargets.Add((edge.TargetNodeId, edge.TargetParamIndex));

        foreach (var node in _nodes)
        {
            for (int i = 0; i < node.LoadedKernel.Descriptor.Parameters.Count; i++)
            {
                var param = node.LoadedKernel.Descriptor.Parameters[i];
                if (!param.IsPointer) continue;

                bool hasEdge = connectedTargets.Contains((node.Id, i)) ||
                               _edges.Any(e => e.SourceNodeId == node.Id && e.SourceParamIndex == i);
                bool hasExternal = _externalBuffers.ContainsKey((node.Id, i));

                if (!hasEdge && !hasExternal)
                    result.AddWarning($"Node '{node.DebugName}' param '{param.Name}' (index {i}) has no connection or external buffer");
            }
        }

        return result;
    }

    /// <summary>
    /// Build an immutable graph description for the compiler.
    /// </summary>
    internal GraphDescription Build()
    {
        return new GraphDescription(
            _nodes.ToList().AsReadOnly(),
            _edges.ToList().AsReadOnly(),
            new Dictionary<(Guid, int), CUdeviceptr>(_externalBuffers),
            _memsetDescriptors.ToList().AsReadOnly(),
            _capturedNodes.ToList().AsReadOnly(),
            _capturedDependencies.ToList().AsReadOnly());
    }

    /// <summary>
    /// Add a memset operation to the graph (e.g., for resetting AppendBuffer counters).
    /// </summary>
    internal MemsetDescriptor AddMemset(CUdeviceptr dst, uint value, uint elemSize, ulong width, string debugName)
    {
        var desc = new MemsetDescriptor(dst, value, elemSize, width, debugName);
        _memsetDescriptors.Add(desc);
        return desc;
    }

    /// <summary>
    /// Declare that a kernel node depends on a memset completing first.
    /// </summary>
    internal void AddMemsetDependency(MemsetDescriptor memset, KernelNode kernelNode)
    {
        memset.DependentKernelNodeIds.Add(kernelNode.Id);
    }

    public IReadOnlyList<KernelNode> Nodes => _nodes;
    public IReadOnlyList<CapturedNode> CapturedNodes => _capturedNodes;
    public IReadOnlyList<Edge> Edges => _edges;
    internal IReadOnlyList<MemsetDescriptor> MemsetDescriptors => _memsetDescriptors;
    internal IReadOnlyList<CapturedDependency> CapturedDependencies => _capturedDependencies;

    /// <summary>
    /// Kahn's algorithm for cycle detection. Returns true if the graph is acyclic.
    /// </summary>
    private bool DetectCyclesKahn(Dictionary<Guid, KernelNode> nodeMap, ValidationResult result)
    {
        // Build adjacency from edges (node-level, not param-level)
        var inDegree = new Dictionary<Guid, int>();
        var adjacency = new Dictionary<Guid, List<Guid>>();

        foreach (var node in _nodes)
        {
            inDegree[node.Id] = 0;
            adjacency[node.Id] = new List<Guid>();
        }

        foreach (var edge in _edges)
        {
            if (!adjacency.ContainsKey(edge.SourceNodeId) || !inDegree.ContainsKey(edge.TargetNodeId))
                continue;

            // Only count unique node-to-node edges for cycle detection
            if (!adjacency[edge.SourceNodeId].Contains(edge.TargetNodeId))
            {
                adjacency[edge.SourceNodeId].Add(edge.TargetNodeId);
                inDegree[edge.TargetNodeId]++;
            }
        }

        var queue = new Queue<Guid>();
        foreach (var kvp in inDegree)
        {
            if (kvp.Value == 0)
                queue.Enqueue(kvp.Key);
        }

        int visited = 0;
        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            visited++;

            foreach (var neighbor in adjacency[current])
            {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0)
                    queue.Enqueue(neighbor);
            }
        }

        if (visited != _nodes.Count)
        {
            result.AddError("Graph contains a cycle");
            return false;
        }

        return true;
    }

    private static string NormalizeType(string type)
    {
        // Strip pointer suffix and normalize
        return type.TrimEnd('*', ' ').ToLowerInvariant();
    }
}
