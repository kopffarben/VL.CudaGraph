using System;
using System.Collections.Generic;

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

    /// <summary>
    /// Append buffer counts per port name, written after auto-readback.
    /// Block authors read: DebugInfo?.AppendCounts?["PortName"] to expose as VL output pin.
    /// </summary>
    IReadOnlyDictionary<string, int>? AppendCounts { get; }
}
