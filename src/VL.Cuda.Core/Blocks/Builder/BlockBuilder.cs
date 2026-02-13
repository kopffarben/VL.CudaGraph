using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;
using VL.Cuda.Core.Buffers;
using VL.Cuda.Core.Context;
using VL.Cuda.Core.Graph;
using VL.Cuda.Core.PTX;
using VL.Cuda.Core.PTX.Compilation;

namespace VL.Cuda.Core.Blocks.Builder;

/// <summary>
/// DSL for describing a block's kernels, ports, parameters, and internal connections.
/// Used in block constructors. Call Commit() when done to finalize the description.
/// </summary>
public sealed class BlockBuilder
{
    private readonly CudaContext _context;
    private readonly ICudaBlock _block;
    private readonly List<KernelHandle> _kernels = new();
    private readonly List<CapturedHandle> _capturedHandles = new();
    private readonly List<BlockPort> _inputs = new();
    private readonly List<BlockPort> _outputs = new();
    private readonly List<IBlockParameter> _parameters = new();
    private readonly List<(Guid SrcKernel, int SrcParam, Guid TgtKernel, int TgtParam)> _internalConnections = new();
    private readonly List<AppendBufferInfo> _appendBuffers = new();
    private bool _committed;

    public BlockBuilder(CudaContext context, ICudaBlock block)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _block = block ?? throw new ArgumentNullException(nameof(block));
    }

    /// <summary>
    /// Add a kernel from a PTX file path. Returns a handle for binding ports.
    /// </summary>
    public KernelHandle AddKernel(string ptxPath)
    {
        EnsureNotCommitted();
        var loaded = _context.ModuleCache.GetOrLoad(ptxPath);
        var handle = new KernelHandle(ptxPath, loaded.Descriptor);
        _kernels.Add(handle);
        return handle;
    }

    /// <summary>
    /// Add a kernel from an ILGPU-compiled C# method. Returns a handle for binding ports.
    /// The method must follow ILGPU kernel conventions (Index1D first parameter, etc.).
    /// The user-provided descriptor is stored for recompilation; the In()/Out() indices
    /// are auto-remapped to account for ILGPU's (pointer, length) parameter expansion.
    /// </summary>
    public KernelHandle AddKernel(MethodInfo kernelMethod, KernelDescriptor descriptor)
    {
        EnsureNotCommitted();
        // Compile and load (IlgpuCompiler auto-expands the descriptor internally)
        _context.IlgpuCompiler.GetOrCompile(kernelMethod, descriptor);
        var methodHash = IlgpuCompiler.ComputeMethodHash(kernelMethod);
        var source = new KernelSource.IlgpuMethod(methodHash, kernelMethod);
        // Compute index remap: user-facing indices → expanded PTX indices
        var indexRemap = IlgpuCompiler.ComputeIndexRemap(descriptor);
        // Store the ORIGINAL descriptor (for recompilation in KernelEntry)
        var handle = new KernelHandle(source, descriptor, indexRemap);
        _kernels.Add(handle);
        return handle;
    }

    /// <summary>
    /// Add a kernel from CUDA C++ source compiled via NVRTC. Returns a handle for binding ports.
    /// </summary>
    public KernelHandle AddKernelFromCuda(string cudaSource, string entryPoint, KernelDescriptor descriptor)
    {
        EnsureNotCommitted();
        var loaded = _context.NvrtcCache.GetOrCompile(cudaSource, entryPoint, descriptor);
        var sourceHash = NvrtcCache.ComputeSourceKey(cudaSource);
        var source = new KernelSource.NvrtcSource(sourceHash, cudaSource, entryPoint);
        var handle = new KernelHandle(source, loaded.Descriptor);
        _kernels.Add(handle);
        return handle;
    }

    /// <summary>
    /// Add a captured library operation. Returns a handle for binding ports.
    /// The captureAction is called during stream capture to record library calls.
    /// </summary>
    public CapturedHandle AddCaptured(Action<CUstream, CUdeviceptr[]> captureAction, CapturedNodeDescriptor descriptor)
    {
        EnsureNotCommitted();
        var handle = new CapturedHandle(descriptor, captureAction);
        _capturedHandles.Add(handle);
        return handle;
    }

    /// <summary>
    /// Define a buffer input port bound to a captured operation parameter.
    /// </summary>
    public BlockPort Input<T>(string name, CapturedPin pin) where T : unmanaged
    {
        EnsureNotCommitted();
        var port = new BlockPort(_block.Id, name, PortDirection.Input, PinType.Buffer<T>())
        {
            KernelNodeId = pin.CapturedHandleId,
            KernelParamIndex = pin.ParamIndex,
        };
        _inputs.Add(port);
        return port;
    }

    /// <summary>
    /// Define a buffer output port bound to a captured operation parameter.
    /// </summary>
    public BlockPort Output<T>(string name, CapturedPin pin) where T : unmanaged
    {
        EnsureNotCommitted();
        var port = new BlockPort(_block.Id, name, PortDirection.Output, PinType.Buffer<T>())
        {
            KernelNodeId = pin.CapturedHandleId,
            KernelParamIndex = pin.ParamIndex,
        };
        _outputs.Add(port);
        return port;
    }

    /// <summary>
    /// Define a buffer input port bound to a kernel parameter.
    /// </summary>
    public BlockPort Input<T>(string name, KernelPin pin) where T : unmanaged
    {
        EnsureNotCommitted();
        var port = new BlockPort(_block.Id, name, PortDirection.Input, PinType.Buffer<T>())
        {
            KernelNodeId = pin.KernelHandleId,
            KernelParamIndex = pin.ParamIndex,
        };
        _inputs.Add(port);
        return port;
    }

    /// <summary>
    /// Define a buffer output port bound to a kernel parameter.
    /// </summary>
    public BlockPort Output<T>(string name, KernelPin pin) where T : unmanaged
    {
        EnsureNotCommitted();
        var port = new BlockPort(_block.Id, name, PortDirection.Output, PinType.Buffer<T>())
        {
            KernelNodeId = pin.KernelHandleId,
            KernelParamIndex = pin.ParamIndex,
        };
        _outputs.Add(port);
        return port;
    }

    /// <summary>
    /// Define an append buffer output. Creates a data output port and records
    /// the data+counter kernel parameter bindings. The engine will allocate the
    /// AppendBuffer, wire both pointers, insert a memset node to reset the counter,
    /// and auto-read the count after each launch. A "{name} Count" entry will appear
    /// in BlockDebugInfo.AppendCounts for the block to expose as a VL output pin.
    /// </summary>
    public AppendOutputPort AppendOutput<T>(string name, KernelPin dataPin, KernelPin counterPin, int maxCapacity)
        where T : unmanaged
    {
        EnsureNotCommitted();

        // Create the data output port (downstream sees a normal buffer)
        var dataPort = new BlockPort(_block.Id, name, PortDirection.Output, PinType.Buffer<T>())
        {
            KernelNodeId = dataPin.KernelHandleId,
            KernelParamIndex = dataPin.ParamIndex,
        };
        _outputs.Add(dataPort);

        var info = new AppendBufferInfo(
            _block.Id, name,
            dataPin.KernelHandleId, dataPin.ParamIndex,
            counterPin.KernelHandleId, counterPin.ParamIndex,
            maxCapacity, Marshal.SizeOf<T>());
        _appendBuffers.Add(info);

        return new AppendOutputPort(dataPort, info);
    }

    /// <summary>
    /// Define a scalar input parameter bound to a kernel parameter.
    /// Changes trigger Hot Update, not graph rebuild.
    /// </summary>
    public BlockParameter<T> InputScalar<T>(string name, KernelPin pin, T defaultValue = default)
        where T : unmanaged
    {
        EnsureNotCommitted();
        var param = new BlockParameter<T>(name, defaultValue)
        {
            KernelNodeId = pin.KernelHandleId,
            KernelParamIndex = pin.ParamIndex,
        };
        _parameters.Add(param);
        return param;
    }

    /// <summary>
    /// Add an internal connection between two kernel parameters within this block.
    /// </summary>
    public void Connect(KernelPin source, KernelPin target)
    {
        EnsureNotCommitted();
        _internalConnections.Add((source.KernelHandleId, source.ParamIndex,
                                   target.KernelHandleId, target.ParamIndex));
    }

    /// <summary>
    /// Finalize the block description. Wires parameter change events,
    /// performs structure change detection, and stores the description.
    /// </summary>
    public void Commit()
    {
        EnsureNotCommitted();
        _committed = true;

        // Wire parameter change events → CudaContext dirty tracking
        foreach (var param in _parameters)
        {
            WireParameterEvents(param);
        }

        // Build and store the description for change detection
        var description = BuildDescription();
        var oldDescription = _context.GetBlockDescription(_block.Id);

        if (oldDescription != null && !description.StructuralEquals(oldDescription))
        {
            // Structure changed → triggers Cold Rebuild via BlockRegistry event
            // (already handled by Register/Unregister; this catches in-place changes)
        }

        _context.SetBlockDescription(_block.Id, description);
    }

    private void WireParameterEvents(IBlockParameter param)
    {
        // Use reflection to subscribe to the generic ValueChanged event.
        // Works via contravariance: OnChanged(object) matches Action<BlockParameter<T>>
        // because BlockParameter<T> is a reference type that derives from object.
        var paramType = param.GetType();
        var eventInfo = paramType.GetEvent("ValueChanged");
        if (eventInfo == null) return;

        var blockId = _block.Id;
        var paramName = param.Name;
        var context = _context;

        var handlerType = eventInfo.EventHandlerType!;
        var handler = Delegate.CreateDelegate(handlerType,
            new ParameterChangedHandler(context, blockId, paramName),
            typeof(ParameterChangedHandler).GetMethod(nameof(ParameterChangedHandler.OnChanged))!);
        eventInfo.AddEventHandler(param, handler);
    }

    public IReadOnlyList<KernelHandle> Kernels => _kernels;
    public IReadOnlyList<CapturedHandle> CapturedHandles => _capturedHandles;
    public IReadOnlyList<BlockPort> Inputs => _inputs;
    public IReadOnlyList<BlockPort> Outputs => _outputs;
    public IReadOnlyList<IBlockParameter> Parameters => _parameters;
    public IReadOnlyList<(Guid SrcKernel, int SrcParam, Guid TgtKernel, int TgtParam)> InternalConnections
        => _internalConnections;

    private BlockDescription BuildDescription()
    {
        // Build kernel entries with HandleId and grid dims.
        // Store the descriptor for non-filesystem sources (ILGPU/NVRTC) so
        // CudaEngine can recompile on cache invalidation.
        var kernelEntries = _kernels.Select(k => new KernelEntry(
            k.Id, k.Source, k.Descriptor.EntryPoint,
            k.GridDimX, k.GridDimY, k.GridDimZ,
            k.Source is KernelSource.FilesystemPtx ? null : k.Descriptor)).ToList();

        // Build captured entries
        var capturedEntries = _capturedHandles.Select(c => new CapturedEntry(
            c.Id, c.Descriptor, c.CaptureAction)).ToList();

        // Build handle-to-index map for converting internal connections
        var handleToIndex = new Dictionary<Guid, int>();
        for (int i = 0; i < _kernels.Count; i++)
            handleToIndex[_kernels[i].Id] = i;

        // Convert internal connections from HandleId → kernel index
        var indexedConnections = _internalConnections
            .Where(c => handleToIndex.ContainsKey(c.SrcKernel) && handleToIndex.ContainsKey(c.TgtKernel))
            .Select(c => (handleToIndex[c.SrcKernel], c.SrcParam,
                          handleToIndex[c.TgtKernel], c.TgtParam))
            .ToList();

        var ports = _inputs.Select(p => (p.Name, p.Direction, p.Type))
            .Concat(_outputs.Select(p => (p.Name, p.Direction, p.Type)))
            .ToList();

        return new BlockDescription(kernelEntries, ports, indexedConnections,
            _appendBuffers.Count > 0 ? _appendBuffers.ToList() : null,
            capturedEntries.Count > 0 ? capturedEntries : null);
    }

    private void EnsureNotCommitted()
    {
        if (_committed)
            throw new InvalidOperationException("BlockBuilder has already been committed");
    }

    /// <summary>
    /// Helper class to avoid closure allocations in event wiring.
    /// </summary>
    private sealed class ParameterChangedHandler
    {
        private readonly CudaContext _context;
        private readonly Guid _blockId;
        private readonly string _paramName;

        public ParameterChangedHandler(CudaContext context, Guid blockId, string paramName)
        {
            _context = context;
            _blockId = blockId;
            _paramName = paramName;
        }

        public void OnChanged(object sender)
        {
            _context.OnParameterChanged(_blockId, _paramName);
        }
    }
}
