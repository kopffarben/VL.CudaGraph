using System;

namespace VL.Cuda.Core.Context.Services;

/// <summary>
/// Identifies a dirty parameter: which block and which parameter name changed.
/// </summary>
public readonly record struct DirtyParameter(Guid BlockId, string ParamName);
