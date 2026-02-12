using System;

namespace VL.Cuda.Core.Graph;

/// <summary>
/// Common interface for all graph node types: KernelNode and CapturedNode.
/// Allows GraphCompiler and CudaEngine to work polymorphically with both.
/// </summary>
internal interface IGraphNode : IDisposable
{
    Guid Id { get; }
    string DebugName { get; }
}
