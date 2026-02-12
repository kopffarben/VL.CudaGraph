using System;

namespace VL.Cuda.Core.Context.Services;

/// <summary>
/// Identifies a captured node that needs recapture: which block and which captured operation.
/// </summary>
public readonly record struct DirtyCapturedNode(Guid BlockId, Guid CapturedHandleId);
