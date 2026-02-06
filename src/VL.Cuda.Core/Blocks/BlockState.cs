namespace VL.Cuda.Core.Blocks;

/// <summary>
/// Runtime state of a block, used for VL diagnostic coloring.
/// </summary>
public enum BlockState
{
    OK,
    Warning,
    Error,
    NotCompiled,
}
