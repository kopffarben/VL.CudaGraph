using System;
using System.Collections.Generic;

namespace VL.Cuda.Core.Blocks;

/// <summary>
/// Mutable debug info, written by CudaEngine, read by blocks.
/// </summary>
public sealed class BlockDebugInfo : IBlockDebugInfo
{
    public BlockState State { get; set; } = BlockState.NotCompiled;
    public string? StateMessage { get; set; }
    public TimeSpan LastExecutionTime { get; set; }
    public IReadOnlyDictionary<string, int>? AppendCounts { get; set; }
}
