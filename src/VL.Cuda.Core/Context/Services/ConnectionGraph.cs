using System;
using System.Collections.Generic;
using System.Linq;

namespace VL.Cuda.Core.Context.Services;

/// <summary>
/// Tracks inter-block connections (block port → block port).
/// Fires StructureChanged when connections are added or removed.
/// </summary>
public sealed class ConnectionGraph
{
    private readonly List<Connection> _connections = new();

    /// <summary>
    /// Fired when connections change. DirtyTracker subscribes.
    /// </summary>
    public event Action? StructureChanged;

    public IReadOnlyList<Connection> Connections => _connections;
    public int Count => _connections.Count;

    public void Connect(Guid srcBlockId, string srcPort, Guid tgtBlockId, string tgtPort)
    {
        var conn = new Connection(srcBlockId, srcPort, tgtBlockId, tgtPort);
        if (!_connections.Contains(conn))
        {
            _connections.Add(conn);
            StructureChanged?.Invoke();
        }
    }

    public bool Disconnect(Guid srcBlockId, string srcPort, Guid tgtBlockId, string tgtPort)
    {
        var conn = new Connection(srcBlockId, srcPort, tgtBlockId, tgtPort);
        if (_connections.Remove(conn))
        {
            StructureChanged?.Invoke();
            return true;
        }
        return false;
    }

    /// <summary>
    /// Remove all connections involving a given block.
    /// </summary>
    public int RemoveBlock(Guid blockId)
    {
        int removed = _connections.RemoveAll(c =>
            c.SourceBlockId == blockId || c.TargetBlockId == blockId);
        if (removed > 0)
            StructureChanged?.Invoke();
        return removed;
    }

    /// <summary>
    /// Get all connections where the given block is the source.
    /// </summary>
    public IEnumerable<Connection> GetOutgoing(Guid blockId)
        => _connections.Where(c => c.SourceBlockId == blockId);

    /// <summary>
    /// Get all connections where the given block is the target.
    /// </summary>
    public IEnumerable<Connection> GetIncoming(Guid blockId)
        => _connections.Where(c => c.TargetBlockId == blockId);
}
