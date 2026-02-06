using System;

namespace VL.Cuda.Core.Context.Services;

/// <summary>
/// A connection between two block ports. Source block output → target block input.
/// </summary>
public sealed record Connection(
    Guid SourceBlockId,
    string SourcePort,
    Guid TargetBlockId,
    string TargetPort);
