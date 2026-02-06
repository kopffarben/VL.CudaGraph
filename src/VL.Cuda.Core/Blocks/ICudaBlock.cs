using System;
using System.Collections.Generic;
using VL.Core;

namespace VL.Cuda.Core.Blocks;

/// <summary>
/// A passive block that describes GPU compute work. Blocks never execute GPU work
/// directly — they provide descriptions that CudaEngine compiles and launches.
/// </summary>
public interface ICudaBlock : IDisposable
{
    Guid Id { get; }
    string TypeName { get; }
    NodeContext NodeContext { get; }

    IReadOnlyList<IBlockPort> Inputs { get; }
    IReadOnlyList<IBlockPort> Outputs { get; }
    IReadOnlyList<IBlockParameter> Parameters { get; }

    /// <summary>
    /// Debug info written by CudaEngine after each frame. Read by blocks for tooltips.
    /// </summary>
    IBlockDebugInfo DebugInfo { get; set; }
}
