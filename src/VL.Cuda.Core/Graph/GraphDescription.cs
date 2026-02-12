using System;
using System.Collections.Generic;
using ManagedCuda.BasicTypes;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// Immutable snapshot of a graph description ready for compilation.
/// Produced by GraphBuilder.Build().
/// </summary>
internal sealed class GraphDescription
{
    public IReadOnlyList<KernelNode> Nodes { get; }
    public IReadOnlyList<Edge> Edges { get; }
    public IReadOnlyDictionary<(Guid NodeId, int ParamIndex), CUdeviceptr> ExternalBuffers { get; }
    public IReadOnlyList<MemsetDescriptor> MemsetDescriptors { get; }

    public GraphDescription(
        IReadOnlyList<KernelNode> nodes,
        IReadOnlyList<Edge> edges,
        IReadOnlyDictionary<(Guid NodeId, int ParamIndex), CUdeviceptr> externalBuffers,
        IReadOnlyList<MemsetDescriptor>? memsetDescriptors = null)
    {
        Nodes = nodes;
        Edges = edges;
        ExternalBuffers = externalBuffers;
        MemsetDescriptors = memsetDescriptors ?? Array.Empty<MemsetDescriptor>();
    }
}
