using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using ManagedCuda.BasicTypes;
using VL.Core;
using VL.Cuda.Core.Blocks;
using VL.Cuda.Core.Blocks.Builder;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Context;
using VL.Cuda.Core.Context.Services;
using VL.Core.Import;
using VL.Cuda.Core.Graph;

namespace VL.Cuda.Core.Engine;

/// <summary>
/// The active component of the CUDA pipeline. Owns the CudaContext,
/// compiles and launches the CUDA graph, and distributes debug info.
/// One CudaEngine per pipeline. Supports both KernelNodes and CapturedNodes.
/// </summary>
[ProcessNode(HasStateOutput = true)]
public sealed class CudaEngine : IDisposable
{
    private readonly ManagedCuda.CudaStream _stream;
    private readonly Stopwatch _stopwatch = new();

    /// <summary>
    /// All KernelNodes created during ColdRebuild. We own their lifetime
    /// since CompiledGraph.Dispose() does NOT dispose KernelNodes.
    /// </summary>
    private readonly List<KernelNode> _ownedKernelNodes = new();

    /// <summary>
    /// All CapturedNodes created during ColdRebuild. We own their lifetime.
    /// </summary>
    private readonly List<CapturedNode> _ownedCapturedNodes = new();

    /// <summary>
    /// AppendBuffers allocated during ColdRebuild. Disposed on rebuild.
    /// </summary>
    private readonly List<IAppendBuffer> _ownedAppendBuffers = new();

    /// <summary>
    /// Maps (blockId, portName) → IAppendBuffer for auto-readback after launch.
    /// </summary>
    private readonly Dictionary<(Guid BlockId, string PortName), IAppendBuffer> _appendBufferMapping = new();

    /// <summary>
    /// Maps (blockId, paramName) → (kernelNodeId, paramIndex) for Hot/Warm updates.
    /// Built during ColdRebuild.
    /// </summary>
    private readonly Dictionary<(Guid BlockId, string ParamName), (Guid KernelNodeId, int ParamIndex)> _paramMapping = new();

    /// <summary>
    /// Maps KernelHandle.Id / CapturedHandle.Id → graph node Id for wiring block ports to graph nodes.
    /// </summary>
    private readonly Dictionary<Guid, Guid> _handleToNodeId = new();

    /// <summary>
    /// Maps (blockId, capturedHandleId) → CapturedNode.Id for Recapture updates.
    /// </summary>
    private readonly Dictionary<(Guid BlockId, Guid HandleId), Guid> _capturedHandleToNodeId = new();

    private CompiledGraph? _compiledGraph;
    private bool _disposed;

    public CudaContext Context { get; }

    /// <summary>
    /// The VL NodeContext for this engine. Null when created via test constructor.
    /// </summary>
    public NodeContext? NodeContext { get; }

    /// <summary>
    /// Whether a compiled graph is ready for launch.
    /// </summary>
    public bool IsCompiled => _compiledGraph != null;

    /// <summary>
    /// Last Cold Rebuild duration.
    /// </summary>
    public TimeSpan LastRebuildTime { get; private set; }

    /// <summary>
    /// Production constructor. NodeContext is injected by VL as first parameter.
    /// </summary>
    public CudaEngine(NodeContext nodeContext, CudaEngineOptions? options = null)
    {
        NodeContext = nodeContext;
        Context = new CudaContext(options ?? new CudaEngineOptions());
        _stream = new ManagedCuda.CudaStream();
    }

    /// <summary>
    /// Test-friendly constructor with injected CudaContext.
    /// </summary>
    internal CudaEngine(CudaContext context)
    {
        Context = context;
        _stream = new ManagedCuda.CudaStream();
    }

    /// <summary>
    /// Main frame update. Call once per frame.
    /// Priority: Structure > Recapture > Parameters > Launch.
    /// </summary>
    public void Update()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (Context.Dirty.IsStructureDirty)
        {
            ColdRebuild();
            Context.Dirty.ClearStructureDirty();
        }
        else if (Context.Dirty.AreCapturedNodesDirty)
        {
            RecaptureNodes();
            Context.Dirty.ClearCapturedNodesDirty();
        }
        else if (Context.Dirty.AreParametersDirty)
        {
            UpdateParameters();
            Context.Dirty.ClearParametersDirty();
        }

        if (_compiledGraph != null)
        {
            _compiledGraph.Launch(_stream);
            _stream.Synchronize();
            ReadbackAppendCounters();
            DistributeDebugInfo(BlockState.OK);
        }
    }

    /// <summary>
    /// Full graph rebuild: dispose old graph, create new from block descriptions.
    /// </summary>
    private void ColdRebuild()
    {
        _stopwatch.Restart();

        // 1. Dispose old resources
        _compiledGraph?.Dispose();
        _compiledGraph = null;

        foreach (var node in _ownedKernelNodes)
            node.Dispose();
        _ownedKernelNodes.Clear();

        foreach (var node in _ownedCapturedNodes)
            node.Dispose();
        _ownedCapturedNodes.Clear();

        foreach (var ab in _ownedAppendBuffers)
            ab.Dispose();
        _ownedAppendBuffers.Clear();
        _appendBufferMapping.Clear();

        _paramMapping.Clear();
        _handleToNodeId.Clear();
        _capturedHandleToNodeId.Clear();

        // 2. Check if there are any blocks
        if (Context.Registry.Count == 0)
        {
            DistributeDebugInfo(BlockState.NotCompiled);
            _stopwatch.Stop();
            LastRebuildTime = _stopwatch.Elapsed;
            return;
        }

        try
        {
            // 3. Build the graph from block descriptions
            var graphBuilder = new GraphBuilder(Context.Device, Context.ModuleCache);

            foreach (var block in Context.Registry.All)
            {
                BuildBlockNodes(graphBuilder, block);
            }

            // 4. Add internal connections within blocks (using kernel indices → node IDs)
            foreach (var block in Context.Registry.All)
            {
                var desc = Context.GetBlockDescription(block.Id);
                if (desc == null) continue;

                foreach (var (srcIdx, srcParam, tgtIdx, tgtParam) in desc.InternalConnections)
                {
                    if (srcIdx >= desc.KernelEntries.Count || tgtIdx >= desc.KernelEntries.Count) continue;

                    var srcHandleId = desc.KernelEntries[srcIdx].HandleId;
                    var tgtHandleId = desc.KernelEntries[tgtIdx].HandleId;

                    if (_handleToNodeId.TryGetValue(srcHandleId, out var srcNodeId) &&
                        _handleToNodeId.TryGetValue(tgtHandleId, out var tgtNodeId))
                    {
                        var srcNode = FindKernelNode(srcNodeId);
                        var tgtNode = FindKernelNode(tgtNodeId);
                        if (srcNode != null && tgtNode != null)
                            graphBuilder.AddEdge(srcNode, srcParam, tgtNode, tgtParam);
                    }
                }
            }

            // 5. Add inter-block connections from ConnectionGraph
            foreach (var conn in Context.Connections.Connections)
            {
                var srcBlock = Context.Registry.Get(conn.SourceBlockId);
                var tgtBlock = Context.Registry.Get(conn.TargetBlockId);
                if (srcBlock == null || tgtBlock == null) continue;

                var srcPort = srcBlock.Outputs.OfType<BlockPort>().FirstOrDefault(p => p.Name == conn.SourcePort);
                var tgtPort = tgtBlock.Inputs.OfType<BlockPort>().FirstOrDefault(p => p.Name == conn.TargetPort);
                if (srcPort == null || tgtPort == null) continue;

                if (_handleToNodeId.TryGetValue(srcPort.KernelNodeId, out var srcNodeId) &&
                    _handleToNodeId.TryGetValue(tgtPort.KernelNodeId, out var tgtNodeId))
                {
                    var srcNode = FindKernelNode(srcNodeId);
                    var tgtNode = FindKernelNode(tgtNodeId);
                    if (srcNode != null && tgtNode != null)
                        graphBuilder.AddEdge(srcNode, srcPort.KernelParamIndex, tgtNode, tgtPort.KernelParamIndex);
                }
            }

            // 6. Apply external buffer bindings
            foreach (var ((blockId, portName), pointer) in Context.ExternalBuffers)
            {
                var block = Context.Registry.Get(blockId);
                if (block == null) continue;

                var port = block.Inputs.OfType<BlockPort>().FirstOrDefault(p => p.Name == portName)
                    ?? block.Outputs.OfType<BlockPort>().FirstOrDefault(p => p.Name == portName);
                if (port == null) continue;

                if (_handleToNodeId.TryGetValue(port.KernelNodeId, out var nodeId))
                {
                    var kernelNode = FindKernelNode(nodeId);
                    if (kernelNode != null)
                    {
                        graphBuilder.SetExternalBuffer(kernelNode, port.KernelParamIndex, pointer);
                    }
                    else
                    {
                        // Wire buffer to captured node's BufferBindings array
                        var capturedNode = FindCapturedNode(nodeId);
                        if (capturedNode != null)
                        {
                            var paramIndex = port.KernelParamIndex;
                            if (paramIndex >= 0 && paramIndex < capturedNode.BufferBindings.Length)
                                capturedNode.BufferBindings[paramIndex] = pointer;
                        }
                    }
                }
            }

            // 7. Apply current parameter values
            foreach (var block in Context.Registry.All)
            {
                ApplyBlockParameters(block);
            }

            // 7a. Allocate AppendBuffers and wire counter memset nodes
            foreach (var block in Context.Registry.All)
            {
                var desc = Context.GetBlockDescription(block.Id);
                if (desc == null) continue;

                foreach (var appendInfo in desc.AppendBuffers)
                {
                    AllocateAndWireAppendBuffer(graphBuilder, appendInfo);
                }
            }

            // 8. Compile
            var compiler = new GraphCompiler(Context.Device, Context.Pool);
            _compiledGraph = compiler.Compile(graphBuilder);

            DistributeDebugInfo(BlockState.OK);
        }
        catch (Exception ex)
        {
            DistributeDebugInfo(BlockState.Error, ex.Message);
        }

        _stopwatch.Stop();
        LastRebuildTime = _stopwatch.Elapsed;
    }

    /// <summary>
    /// Create KernelNodes and CapturedNodes from block description.
    /// </summary>
    private void BuildBlockNodes(GraphBuilder graphBuilder, ICudaBlock block)
    {
        var desc = Context.GetBlockDescription(block.Id);
        if (desc == null) return;

        // Create a KernelNode for each kernel entry
        foreach (var entry in desc.KernelEntries)
        {
            if (_handleToNodeId.ContainsKey(entry.HandleId)) continue;

            var loaded = Context.ModuleCache.GetOrLoad(entry.PtxPath);
            var kernelNode = graphBuilder.AddKernel(loaded, $"{block.TypeName}.{entry.EntryPoint}");

            kernelNode.GridDimX = entry.GridDimX;
            kernelNode.GridDimY = entry.GridDimY;
            kernelNode.GridDimZ = entry.GridDimZ;

            _ownedKernelNodes.Add(kernelNode);
            _handleToNodeId[entry.HandleId] = kernelNode.Id;
        }

        // Create a CapturedNode for each captured entry
        foreach (var entry in desc.CapturedEntries)
        {
            if (_handleToNodeId.ContainsKey(entry.HandleId)) continue;

            var capturedNode = graphBuilder.AddCaptured(
                entry.Descriptor, entry.CaptureAction,
                $"{block.TypeName}.{entry.Descriptor.DebugName}");

            _ownedCapturedNodes.Add(capturedNode);
            _handleToNodeId[entry.HandleId] = capturedNode.Id;
            _capturedHandleToNodeId[(block.Id, entry.HandleId)] = capturedNode.Id;
        }

        // Build parameter mapping: (blockId, paramName) → (kernelNodeId, paramIndex)
        foreach (var param in block.Parameters)
        {
            var (handleId, paramIndex) = GetParamKernelMapping(param);
            if (handleId != Guid.Empty && _handleToNodeId.TryGetValue(handleId, out var nodeId))
            {
                _paramMapping[(block.Id, param.Name)] = (nodeId, paramIndex);
            }
        }
    }

    /// <summary>
    /// Extract kernel mapping from a BlockParameter without reflection.
    /// </summary>
    private static (Guid HandleId, int ParamIndex) GetParamKernelMapping(IBlockParameter param)
    {
        switch (param)
        {
            case BlockParameter<float> p: return (p.KernelNodeId, p.KernelParamIndex);
            case BlockParameter<uint> p: return (p.KernelNodeId, p.KernelParamIndex);
            case BlockParameter<int> p: return (p.KernelNodeId, p.KernelParamIndex);
            case BlockParameter<double> p: return (p.KernelNodeId, p.KernelParamIndex);
            case BlockParameter<long> p: return (p.KernelNodeId, p.KernelParamIndex);
            case BlockParameter<ulong> p: return (p.KernelNodeId, p.KernelParamIndex);
            case BlockParameter<short> p: return (p.KernelNodeId, p.KernelParamIndex);
            case BlockParameter<ushort> p: return (p.KernelNodeId, p.KernelParamIndex);
            case BlockParameter<byte> p: return (p.KernelNodeId, p.KernelParamIndex);
            case BlockParameter<sbyte> p: return (p.KernelNodeId, p.KernelParamIndex);
        }

        var pt = param.GetType();
        if (pt.IsGenericType && pt.GetGenericTypeDefinition() == typeof(BlockParameter<>))
        {
            var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            var handleId = (Guid)(pt.GetProperty("KernelNodeId", flags)?.GetValue(param) ?? Guid.Empty);
            var paramIndex = (int)(pt.GetProperty("KernelParamIndex", flags)?.GetValue(param) ?? 0);
            return (handleId, paramIndex);
        }

        return (Guid.Empty, 0);
    }

    /// <summary>
    /// Apply all parameter values from a block to the corresponding KernelNodes.
    /// </summary>
    private void ApplyBlockParameters(ICudaBlock block)
    {
        foreach (var param in block.Parameters)
        {
            if (!_paramMapping.TryGetValue((block.Id, param.Name), out var mapping)) continue;
            var node = FindKernelNode(mapping.KernelNodeId);
            if (node == null) continue;
            ApplyScalar(node, mapping.ParamIndex, param);
        }
    }

    /// <summary>
    /// Apply a scalar parameter value to a KernelNode.
    /// </summary>
    private static void ApplyScalar(KernelNode node, int paramIndex, IBlockParameter param)
    {
        switch (param.Value)
        {
            case float f: node.SetScalar(paramIndex, f); break;
            case uint u: node.SetScalar(paramIndex, u); break;
            case int i: node.SetScalar(paramIndex, i); break;
            case double d: node.SetScalar(paramIndex, d); break;
            case long l: node.SetScalar(paramIndex, l); break;
            case ulong ul: node.SetScalar(paramIndex, ul); break;
            case short s: node.SetScalar(paramIndex, s); break;
            case ushort us: node.SetScalar(paramIndex, us); break;
            case byte b: node.SetScalar(paramIndex, b); break;
            case sbyte sb: node.SetScalar(paramIndex, sb); break;
        }
    }

    /// <summary>
    /// Hot/Warm Update: apply only changed parameters without graph rebuild.
    /// </summary>
    private void UpdateParameters()
    {
        if (_compiledGraph == null) return;

        foreach (var dirty in Context.Dirty.GetDirtyParameters())
        {
            var block = Context.Registry.Get(dirty.BlockId);
            if (block == null) continue;

            var param = block.Parameters.FirstOrDefault(p => p.Name == dirty.ParamName);
            if (param == null) continue;

            if (!_paramMapping.TryGetValue((dirty.BlockId, dirty.ParamName), out var mapping)) continue;

            switch (param.Value)
            {
                case float f: _compiledGraph.UpdateScalar(mapping.KernelNodeId, mapping.ParamIndex, f); break;
                case uint u: _compiledGraph.UpdateScalar(mapping.KernelNodeId, mapping.ParamIndex, u); break;
                case int i: _compiledGraph.UpdateScalar(mapping.KernelNodeId, mapping.ParamIndex, i); break;
                case double d: _compiledGraph.UpdateScalar(mapping.KernelNodeId, mapping.ParamIndex, d); break;
                case long l: _compiledGraph.UpdateScalar(mapping.KernelNodeId, mapping.ParamIndex, l); break;
                case ulong ul: _compiledGraph.UpdateScalar(mapping.KernelNodeId, mapping.ParamIndex, ul); break;
                case short s: _compiledGraph.UpdateScalar(mapping.KernelNodeId, mapping.ParamIndex, s); break;
                case ushort us: _compiledGraph.UpdateScalar(mapping.KernelNodeId, mapping.ParamIndex, us); break;
                case byte b: _compiledGraph.UpdateScalar(mapping.KernelNodeId, mapping.ParamIndex, b); break;
                case sbyte sb: _compiledGraph.UpdateScalar(mapping.KernelNodeId, mapping.ParamIndex, sb); break;
            }
        }
    }

    /// <summary>
    /// Recapture Update: re-stream-capture affected CapturedNodes and update executable graph.
    /// </summary>
    private void RecaptureNodes()
    {
        if (_compiledGraph == null) return;

        foreach (var dirty in Context.Dirty.GetDirtyCapturedNodes())
        {
            if (!_capturedHandleToNodeId.TryGetValue((dirty.BlockId, dirty.CapturedHandleId), out var nodeId))
                continue;

            var capturedNode = FindCapturedNode(nodeId);
            if (capturedNode == null) continue;

            // Re-capture the operation
            var newGraph = capturedNode.Capture(_stream.Stream);

            // Update the executable graph's child graph node
            _compiledGraph.RecaptureNode(nodeId, newGraph);
        }
    }

    /// <summary>
    /// Allocate an AppendBuffer for the given info, wire data+counter pointers
    /// as external buffers, and register a counter-reset memset node.
    /// </summary>
    private void AllocateAndWireAppendBuffer(GraphBuilder graphBuilder, AppendBufferInfo info)
    {
        int totalBytes = info.MaxCapacity * info.ElementSize;
        var dataBuffer = Context.Pool.Acquire<byte>(totalBytes);
        var counterBuffer = Context.Pool.Acquire<uint>(1);

        var appendBuffer = new AppendBuffer<byte>(dataBuffer, counterBuffer, info.MaxCapacity,
            BufferLifetime.Graph, onDispose: ab =>
            {
                ab.Data.Dispose();
                ab.Counter.Dispose();
            });
        _ownedAppendBuffers.Add(appendBuffer);
        _appendBufferMapping[(info.BlockId, info.PortName)] = appendBuffer;

        // Wire data pointer to the kernel parameter
        if (_handleToNodeId.TryGetValue(info.DataKernelHandleId, out var dataNodeId))
        {
            var dataNode = FindKernelNode(dataNodeId);
            if (dataNode != null)
                graphBuilder.SetExternalBuffer(dataNode, info.DataParamIndex, appendBuffer.DataPointer);
        }

        // Wire counter pointer to the kernel parameter
        if (_handleToNodeId.TryGetValue(info.CounterKernelHandleId, out var counterNodeId))
        {
            var counterNode = FindKernelNode(counterNodeId);
            if (counterNode != null)
                graphBuilder.SetExternalBuffer(counterNode, info.CounterParamIndex, appendBuffer.CounterPointer);
        }

        // Register memset node to reset counter to 0 before each launch
        var memset = graphBuilder.AddMemset(
            appendBuffer.CounterPointer,
            value: 0,
            elemSize: 4,
            width: 1,
            debugName: $"Reset {info.PortName} Counter");

        if (_handleToNodeId.TryGetValue(info.DataKernelHandleId, out var depNodeId))
        {
            var depNode = FindKernelNode(depNodeId);
            if (depNode != null)
                graphBuilder.AddMemsetDependency(memset, depNode);
        }

        if (info.CounterKernelHandleId != info.DataKernelHandleId &&
            _handleToNodeId.TryGetValue(info.CounterKernelHandleId, out var counterDepNodeId))
        {
            var counterDepNode = FindKernelNode(counterDepNodeId);
            if (counterDepNode != null)
                graphBuilder.AddMemsetDependency(memset, counterDepNode);
        }
    }

    /// <summary>
    /// Auto-readback: read all append buffer counters after launch+sync.
    /// </summary>
    private void ReadbackAppendCounters()
    {
        foreach (var ab in _ownedAppendBuffers)
        {
            ab.ReadCount();
        }
    }

    private KernelNode? FindKernelNode(Guid nodeId)
        => _ownedKernelNodes.FirstOrDefault(n => n.Id == nodeId);

    private CapturedNode? FindCapturedNode(Guid nodeId)
        => _ownedCapturedNodes.FirstOrDefault(n => n.Id == nodeId);

    private void DistributeDebugInfo(BlockState state, string? message = null)
    {
        foreach (var block in Context.Registry.All)
        {
            Dictionary<string, int>? appendCounts = null;
            foreach (var kvp in _appendBufferMapping)
            {
                if (kvp.Key.BlockId != block.Id) continue;
                appendCounts ??= new Dictionary<string, int>();
                appendCounts[kvp.Key.PortName] = kvp.Value.LastReadCount;
            }

            block.DebugInfo = new BlockDebugInfo
            {
                State = state,
                StateMessage = BuildStateMessage(message, block.Id),
                LastExecutionTime = _stopwatch.Elapsed,
                AppendCounts = appendCounts,
            };
        }
    }

    private string? BuildStateMessage(string? baseMessage, Guid blockId)
    {
        var overflows = new List<string>();
        foreach (var kvp in _appendBufferMapping)
        {
            if (kvp.Key.BlockId != blockId) continue;
            if (kvp.Value.DidOverflow)
                overflows.Add($"{kvp.Key.PortName}: overflow ({kvp.Value.LastRawCount}/{kvp.Value.MaxCapacity})");
        }

        if (overflows.Count == 0) return baseMessage;

        var overflowMsg = string.Join("; ", overflows);
        return baseMessage != null ? $"{baseMessage}; {overflowMsg}" : overflowMsg;
    }

    public override string ToString()
    {
        var blockCount = Context.Registry.Count;
        var status = _compiledGraph != null ? "compiled" : "not compiled";
        return $"CudaEngine: {blockCount} blocks, {status}";
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _compiledGraph?.Dispose();
        _compiledGraph = null;

        foreach (var node in _ownedKernelNodes)
            node.Dispose();
        _ownedKernelNodes.Clear();

        foreach (var node in _ownedCapturedNodes)
            node.Dispose();
        _ownedCapturedNodes.Clear();

        foreach (var ab in _ownedAppendBuffers)
            ab.Dispose();
        _ownedAppendBuffers.Clear();
        _appendBufferMapping.Clear();

        _stream.Dispose();
        Context.Dispose();
    }
}
