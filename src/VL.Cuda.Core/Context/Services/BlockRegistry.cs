using System;
using System.Collections.Generic;
using VL.Cuda.Core.Blocks;

namespace VL.Cuda.Core.Context.Services;

/// <summary>
/// Registry of all blocks participating in the CUDA pipeline.
/// Fires StructureChanged when blocks are registered or unregistered.
/// </summary>
public sealed class BlockRegistry
{
    private readonly Dictionary<Guid, ICudaBlock> _blocks = new();

    /// <summary>
    /// Fired when a block is registered or unregistered. DirtyTracker subscribes.
    /// </summary>
    public event Action? StructureChanged;

    public IReadOnlyDictionary<Guid, ICudaBlock> Blocks => _blocks;
    public int Count => _blocks.Count;

    public void Register(ICudaBlock block)
    {
        ArgumentNullException.ThrowIfNull(block);
        _blocks[block.Id] = block;
        StructureChanged?.Invoke();
    }

    public bool Unregister(Guid blockId)
    {
        if (_blocks.Remove(blockId))
        {
            StructureChanged?.Invoke();
            return true;
        }
        return false;
    }

    public ICudaBlock? Get(Guid blockId)
    {
        _blocks.TryGetValue(blockId, out var block);
        return block;
    }

    public IEnumerable<ICudaBlock> All => _blocks.Values;
}
