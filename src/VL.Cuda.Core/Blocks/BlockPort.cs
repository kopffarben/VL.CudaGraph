using System;

namespace VL.Cuda.Core.Blocks;

/// <summary>
/// Concrete block port: maps a named port to a specific kernel node's parameter.
/// </summary>
public sealed class BlockPort : IBlockPort
{
    public Guid BlockId { get; }
    public string Name { get; }
    public PortDirection Direction { get; }
    public PinType Type { get; }

    /// <summary>
    /// Which kernel node this port maps to (set by BlockBuilder).
    /// </summary>
    internal Guid KernelNodeId { get; set; }

    /// <summary>
    /// Which parameter index on the kernel node this port maps to.
    /// </summary>
    internal int KernelParamIndex { get; set; }

    public BlockPort(Guid blockId, string name, PortDirection direction, PinType type)
    {
        BlockId = blockId;
        Name = name;
        Direction = direction;
        Type = type;
    }

    public override string ToString() => $"{Direction} {Name}: {Type}";
}
