using System;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// A directed edge from one node's output parameter to another node's input parameter.
/// Defines a data dependency: the target node depends on the source node completing first.
/// </summary>
public sealed class Edge
{
    public Guid SourceNodeId { get; }
    public int SourceParamIndex { get; }
    public Guid TargetNodeId { get; }
    public int TargetParamIndex { get; }

    public Edge(Guid sourceNodeId, int sourceParamIndex, Guid targetNodeId, int targetParamIndex)
    {
        SourceNodeId = sourceNodeId;
        SourceParamIndex = sourceParamIndex;
        TargetNodeId = targetNodeId;
        TargetParamIndex = targetParamIndex;
    }
}
