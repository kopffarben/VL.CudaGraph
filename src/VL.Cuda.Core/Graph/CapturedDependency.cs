using System;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// Declares a dependency between a captured node and a kernel node (or vice versa).
/// SourceNodeId must complete before TargetNodeId can begin.
/// Both IDs can refer to either KernelNode or CapturedNode.
/// </summary>
internal readonly record struct CapturedDependency(Guid SourceNodeId, Guid TargetNodeId);
