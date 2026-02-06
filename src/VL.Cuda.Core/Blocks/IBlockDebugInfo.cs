using System;

namespace VL.Cuda.Core.Blocks;

/// <summary>
/// Debug information written by CudaEngine after each frame launch.
/// Read by blocks for VL tooltip display.
/// </summary>
public interface IBlockDebugInfo
{
    BlockState State { get; }
    string? StateMessage { get; }
    TimeSpan LastExecutionTime { get; }
}
