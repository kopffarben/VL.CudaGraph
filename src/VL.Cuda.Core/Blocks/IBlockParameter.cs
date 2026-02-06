using System;

namespace VL.Cuda.Core.Blocks;

/// <summary>
/// A scalar parameter on a block. Changes trigger Hot Update (no graph rebuild).
/// </summary>
public interface IBlockParameter
{
    string Name { get; }
    Type ValueType { get; }
    object Value { get; set; }
    bool IsDirty { get; }
    void ClearDirty();
}
