namespace VL.Cuda.Core.Buffers;

/// <summary>
/// Tracks the validity state of GPU buffer contents.
/// </summary>
public enum BufferState
{
    /// <summary>Buffer contains valid data.</summary>
    Valid = 0,

    /// <summary>Buffer is allocated but contents are undefined.</summary>
    Uninitialized = 1,

    /// <summary>Buffer has been released back to the pool or freed.</summary>
    Released = 2,
}
