namespace VL.Cuda.Core.Buffers;

/// <summary>
/// Who owns/manages the buffer's lifetime.
/// </summary>
public enum BufferLifetime
{
    /// <summary>Managed externally (user-created, explicit dispose).</summary>
    External = 0,

    /// <summary>Managed by the graph compiler (allocated during graph build, freed on rebuild).</summary>
    Graph = 1,

    /// <summary>Scoped to a region (conditional/loop body).</summary>
    Region = 2,
}
