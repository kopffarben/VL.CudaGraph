using System;
using System.Collections.Generic;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Context.Services;
using VL.Cuda.Core.Device;
using VL.Cuda.Core.Libraries;
using VL.Cuda.Core.PTX;

namespace VL.Cuda.Core.Context;

/// <summary>
/// Facade over all CUDA pipeline services. Owns DeviceContext, BufferPool,
/// ModuleCache, BlockRegistry, ConnectionGraph, DirtyTracker.
/// Created and owned by CudaEngine.
/// </summary>
public sealed class CudaContext : IDisposable
{
    private readonly Dictionary<Guid, BlockDescription> _blockDescriptions = new();
    private readonly Dictionary<(Guid BlockId, string PortName), CUdeviceptr> _externalBuffers = new();
    private bool _disposed;

    public DeviceContext Device { get; }
    public BufferPool Pool { get; }
    public ModuleCache ModuleCache { get; }
    public BlockRegistry Registry { get; }
    public ConnectionGraph Connections { get; }
    public DirtyTracker Dirty { get; }
    public LibraryHandleCache Libraries { get; }

    public CudaContext(CudaEngineOptions options)
    {
        Device = new DeviceContext(options.DeviceId);
        Pool = new BufferPool(Device);
        ModuleCache = new ModuleCache(Device);
        Registry = new BlockRegistry();
        Connections = new ConnectionGraph();
        Dirty = new DirtyTracker();
        Libraries = new LibraryHandleCache();

        Dirty.Subscribe(Registry, Connections);
    }

    /// <summary>
    /// Test-friendly constructor: inject a pre-existing DeviceContext.
    /// </summary>
    internal CudaContext(DeviceContext device)
    {
        Device = device;
        Pool = new BufferPool(Device);
        ModuleCache = new ModuleCache(Device);
        Registry = new BlockRegistry();
        Connections = new ConnectionGraph();
        Dirty = new DirtyTracker();
        Libraries = new LibraryHandleCache();

        Dirty.Subscribe(Registry, Connections);
    }

    // --- Public facade methods ---

    public void RegisterBlock(ICudaBlock block)
    {
        Registry.Register(block);
    }

    public void UnregisterBlock(Guid blockId)
    {
        Registry.Unregister(blockId);
        Connections.RemoveBlock(blockId);
        _blockDescriptions.Remove(blockId);
    }

    public void Connect(Guid srcBlockId, string srcPort, Guid tgtBlockId, string tgtPort)
    {
        Connections.Connect(srcBlockId, srcPort, tgtBlockId, tgtPort);
    }

    public void Disconnect(Guid srcBlockId, string srcPort, Guid tgtBlockId, string tgtPort)
    {
        Connections.Disconnect(srcBlockId, srcPort, tgtBlockId, tgtPort);
    }

    /// <summary>
    /// Called by BlockParameter.ValueChanged → BlockBuilder wiring.
    /// Marks the parameter dirty for Hot/Warm Update.
    /// </summary>
    public void OnParameterChanged(Guid blockId, string paramName)
    {
        Dirty.MarkParameterDirty(new DirtyParameter(blockId, paramName));
    }

    /// <summary>
    /// Called when a captured node's parameters change and it needs recapture.
    /// </summary>
    public void OnCapturedNodeChanged(Guid blockId, Guid capturedHandleId)
    {
        Dirty.MarkCapturedNodeDirty(new DirtyCapturedNode(blockId, capturedHandleId));
    }

    /// <summary>
    /// Store the current block description (for structure change detection).
    /// </summary>
    public void SetBlockDescription(Guid blockId, BlockDescription description)
    {
        _blockDescriptions[blockId] = description;
    }

    /// <summary>
    /// Get the previous block description for change detection.
    /// </summary>
    public BlockDescription? GetBlockDescription(Guid blockId)
    {
        _blockDescriptions.TryGetValue(blockId, out var desc);
        return desc;
    }

    /// <summary>
    /// Bind an external buffer to a specific block port.
    /// Used for graph inputs/outputs managed by the user.
    /// </summary>
    public void SetExternalBuffer(Guid blockId, string portName, CUdeviceptr pointer)
    {
        _externalBuffers[(blockId, portName)] = pointer;
        // External buffer change is a structural change (needs rewiring)
        Dirty.MarkParameterDirty(new DirtyParameter(blockId, portName));
    }

    /// <summary>
    /// Get all external buffer bindings.
    /// </summary>
    public IReadOnlyDictionary<(Guid BlockId, string PortName), CUdeviceptr> ExternalBuffers => _externalBuffers;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        Libraries.Dispose();
        Pool.Dispose();
        ModuleCache.Dispose();
        Device.Dispose();
    }
}
