using System;

namespace VL.Cuda.Core.Blocks;

/// <summary>
/// A block's input or output port. Maps to a kernel parameter in the graph.
/// </summary>
public interface IBlockPort
{
    Guid BlockId { get; }
    string Name { get; }
    PortDirection Direction { get; }
    PinType Type { get; }
}
