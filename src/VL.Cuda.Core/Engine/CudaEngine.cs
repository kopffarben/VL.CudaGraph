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
using VL.Cuda.Core.Graph;

namespace VL.Cuda.Core.Engine;

/// <summary>
/// The active component of the CUDA pipeline. Owns the CudaContext,
/// compiles and launches the CUDA graph, and distributes debug info.
/// One CudaEngine per pipeline.
/// </summary>
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
    /// Maps (blockId, paramName) → (kernelNodeId, paramIndex) for Hot/Warm updates.
    /// Built during ColdRebuild.
    /// </summary>
    private readonly Dictionary<(Guid BlockId, string ParamName), (Guid KernelNodeId, int ParamIndex)> _paramMapping = new();

    /// <summary>
    /// Maps KernelHandle.Id → KernelNode.Id for wiring block ports to graph nodes.
    /// </summary>
    private readonly Dictionary<Guid, Guid> _handleToNodeId = new();

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
    /// Priority: Structure > Parameters > Launch.
    /// </summary>
    public void Update()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (Context.Dirty.IsStructureDirty)
        {
            ColdRebuild();
            Context.Dirty.ClearStructureDirty();
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
        _paramMapping.Clear();
        _handleToNodeId.Clear();

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
                        var srcNode = FindNode(srcNodeId);
                        var tgtNode = FindNode(tgtNodeId);
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
                    var srcNode = FindNode(srcNodeId);
                    var tgtNode = FindNode(tgtNodeId);
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
                    var node = FindNode(nodeId);
                    if (node != null)
                        graphBuilder.SetExternalBuffer(node, port.KernelParamIndex, pointer);
                }
            }

            // 7. Apply current parameter values
            foreach (var block in Context.Registry.All)
            {
                ApplyBlockParameters(block);
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
    /// Create KernelNodes from block description's ordered kernel entries.
    /// No reflection, no HashSet — uses the deterministic kernel list.
    /// </summary>
    private void BuildBlockNodes(GraphBuilder graphBuilder, ICudaBlock block)
    {
        var desc = Context.GetBlockDescription(block.Id);
        if (desc == null) return;

        // Create a KernelNode for each kernel entry (deterministic order from AddKernel calls)
        foreach (var entry in desc.KernelEntries)
        {
            if (_handleToNodeId.ContainsKey(entry.HandleId)) continue;

            var loaded = Context.ModuleCache.GetOrLoad(entry.PtxPath);
            var kernelNode = graphBuilder.AddKernel(loaded, $"{block.TypeName}.{entry.EntryPoint}");

            // Apply grid dimensions from block description
            kernelNode.GridDimX = entry.GridDimX;
            kernelNode.GridDimY = entry.GridDimY;
            kernelNode.GridDimZ = entry.GridDimZ;

            _ownedKernelNodes.Add(kernelNode);
            _handleToNodeId[entry.HandleId] = kernelNode.Id;
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
    /// Tries known types first, falls back to reflection for others.
    /// </summary>
    private static (Guid HandleId, int ParamIndex) GetParamKernelMapping(IBlockParameter param)
    {
        // Fast path: check common concrete types directly
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

        // Fallback: reflection for unknown BlockParameter<T> types
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
            var node = FindNode(mapping.KernelNodeId);
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

            // Hot Update via CompiledGraph
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

    private KernelNode? FindNode(Guid nodeId)
        => _ownedKernelNodes.FirstOrDefault(n => n.Id == nodeId);

    private void DistributeDebugInfo(BlockState state, string? message = null)
    {
        foreach (var block in Context.Registry.All)
        {
            block.DebugInfo = new BlockDebugInfo
            {
                State = state,
                StateMessage = message,
                LastExecutionTime = _stopwatch.Elapsed,
            };
        }
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

        _stream.Dispose();
        Context.Dispose();
    }
}
