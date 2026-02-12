# C# API Reference

> **Status**: Phase 0 + Phase 1 + Phase 2 + Phase 3.1 + Phase 4a implemented and tested (211 tests).
> Types marked *(Phase 3+)* are planned but not yet implemented.

## Module Structure

```
VL.Cuda.Core
  ├── Engine
  │     CudaEngine
  │
  ├── Context
  │     CudaContext
  │     CudaEngineOptions
  │
  ├── Context.Services
  │     BlockRegistry
  │     ConnectionGraph
  │     Connection
  │     DirtyTracker
  │     DirtyParameter
  │     DirtyCapturedNode
  │
  ├── Buffers
  │     GpuBuffer<T>
  │     BufferPool
  │     IAppendBuffer
  │     AppendBuffer<T>
  │     BufferLifetime
  │     DType
  │
  ├── Blocks
  │     ICudaBlock
  │     IBlockPort
  │     BlockPort
  │     IBlockParameter
  │     BlockParameter<T>
  │     IBlockDebugInfo
  │     BlockDebugInfo
  │     PortDirection
  │     PinType
  │     PinKind
  │     BlockState
  │
  ├── Blocks.Builder
  │     BlockBuilder
  │     KernelHandle
  │     KernelPin
  │     KernelEntry
  │     CapturedHandle
  │     CapturedPin
  │     CapturedPinCategory
  │     CapturedEntry
  │     BlockDescription
  │     AppendBufferInfo
  │     AppendOutputPort
  │
  ├── Graph
  │     IGraphNode (internal)
  │     KernelNode
  │     CapturedNode
  │     CapturedNodeDescriptor
  │     CapturedParam
  │     StreamCaptureHelper (internal)
  │     CapturedDependency (internal)
  │     Edge
  │     GraphBuilder
  │     GraphCompiler
  │     CompiledGraph
  │     GraphDescription
  │     MemsetDescriptor (internal)
  │     ValidationResult
  │
  ├── PTX
  │     PtxLoader
  │     ModuleCache
  │     PtxMetadata
  │     KernelDescriptor
  │     KernelParamDescriptor
  │     LoadedKernel
  │
  ├── Device
  │     DeviceContext
  │
  ├── Handles (Phase 3+)
  │     IHandle, IInputHandle, IOutputHandle
  │     InputHandle<T>, OutputHandle<T>
  │
  ├── Libraries
  │     LibraryHandleCache
  │
  ├── Libraries.Blas
  │     BlasOperations
  │
  ├── Libraries.FFT
  │     FftOperations
  │
  ├── Libraries.Sparse
  │     SparseOperations
  │
  ├── Libraries.Rand
  │     RandOperations
  │
  ├── Libraries.Solve
  │     SolveOperations
  │
  ├── Debug (Phase 3+)
  │     IDebugInfo, ProfilingPipeline
  │     RegionTiming, KernelTiming, KernelDebugInfo
  │     GraphStructure, NodeInfo, EdgeInfo
  │     DiagnosticMessage, DiagnosticLevel
  │
  └── Interop (Phase 5)
        IDX11SharedResource
        SharedBuffer<T>, SharedTexture2D<T>
        CudaDX11Interop
```

---

## Engine

### CudaEngine

The single active ProcessNode. The ONLY component that does GPU work.

> See `EXECUTION-MODEL.md` for the full execution model.

```csharp
[ProcessNode(HasStateOutput = true)]
public sealed class CudaEngine : IDisposable
{
    // === Creation ===

    /// <summary>
    /// Production constructor. NodeContext is injected by VL as first parameter.
    /// CudaEngineOptions becomes a visible VL input pin.
    /// </summary>
    public CudaEngine(NodeContext nodeContext, CudaEngineOptions? options = null);

    /// <summary>
    /// Test-friendly constructor with injected CudaContext.
    /// </summary>
    internal CudaEngine(CudaContext context);

    // === Properties ===

    /// <summary>
    /// The CudaContext managed by this engine.
    /// Exposed as state output — flows to all blocks via VL links.
    /// </summary>
    public CudaContext Context { get; }

    /// <summary>
    /// The VL NodeContext for this engine. Null when created via test constructor.
    /// </summary>
    public NodeContext? NodeContext { get; }

    /// <summary>
    /// Whether a compiled graph is ready for launch.
    /// </summary>
    public bool IsCompiled { get; }

    /// <summary>
    /// Duration of the last Cold Rebuild.
    /// </summary>
    public TimeSpan LastRebuildTime { get; }

    // === Execution (called every frame) ===

    /// <summary>
    /// Main update method. Call once per frame.
    /// Priority: Structure dirty → Cold Rebuild; Captured dirty → Recapture; Parameters dirty → Hot Update; then Launch.
    ///
    /// 1. If IsStructureDirty → ColdRebuild (GraphBuilder → Compile → new CompiledGraph)
    /// 2. Else if AreCapturedNodesDirty → RecaptureNodes (re-stream-capture → UpdateChildGraphNode)
    /// 3. Else if AreParametersDirty → Hot/Warm Update (CompiledGraph.UpdateScalar)
    /// 4. If compiled graph exists → Launch + Synchronize + Distribute DebugInfo
    /// </summary>
    public void Update();

    // === Debug ===

    /// <summary>
    /// VL tooltip display. Shows block count and compilation status.
    /// </summary>
    public override string ToString();
    // → "CudaEngine: 3 blocks, compiled"
    // → "CudaEngine: 0 blocks, not compiled"

    // === Dispose ===

    /// <summary>
    /// Disposes compiled graph, owned kernel nodes, owned captured nodes, stream, and CudaContext.
    /// </summary>
    public void Dispose();
}
```

**ColdRebuild pipeline** (internal):
1. Dispose old CompiledGraph + KernelNodes + CapturedNodes + owned AppendBuffers
2. For each registered block: iterate `BlockDescription.KernelEntries` → create KernelNodes (via GraphBuilder) with grid dims
3. For each registered block: iterate `BlockDescription.CapturedEntries` → create CapturedNodes (via GraphBuilder)
4. Add intra-block connections (from `BlockDescription.InternalConnections` using kernel indices)
5. Add inter-block connections (from `ConnectionGraph` using BlockPort → KernelNode/CapturedNode mapping)
6. Allocate and wire AppendBuffers (`AllocateAndWireAppendBuffer()` for each `AppendBufferInfo`)
7. Add memset nodes for AppendBuffer counters (via `GraphBuilder.AddMemset()` + `AddMemsetDependency()`)
8. Apply external buffer bindings (for both KernelNodes and CapturedNodes)
9. Apply current parameter values to KernelNodes
10. Compile via GraphCompiler → new CompiledGraph (stream-captures CapturedNodes → ChildGraphNodes)
11. Distribute debug info (BlockState.OK or BlockState.Error) to all blocks

**Post-Launch** (internal):
- `ReadbackAppendCounters()` — synchronously reads append counter values from GPU
- `BuildStateMessage()` — constructs state message including append counts for debug info

**RecaptureNodes** (internal):
- Iterates `DirtyTracker.GetDirtyCapturedNodes()`
- For each dirty captured node: maps `(blockId, capturedHandleId)` → `CapturedNode.Id`
- Calls `CapturedNode.Capture(stream)` → re-stream-capture → new CUgraph
- Calls `CompiledGraph.RecaptureNode(nodeId, newChildGraph)` → `cuGraphExecChildGraphNodeSetParams`

**Hot/Warm Update** (internal):
- Iterates `DirtyTracker.GetDirtyParameters()`
- For each dirty param: finds block → finds parameter → looks up `_paramMapping` → calls `CompiledGraph.UpdateScalar()`

---

## Context

### CudaContext

Facade over all CUDA pipeline services. Created and owned by CudaEngine.

```csharp
public sealed class CudaContext : IDisposable
{
    // === Creation ===

    /// <summary>
    /// Standard constructor from engine options.
    /// </summary>
    public CudaContext(CudaEngineOptions options);

    /// <summary>
    /// Test-friendly constructor: inject a pre-existing DeviceContext.
    /// </summary>
    internal CudaContext(DeviceContext device);

    // === Service Properties ===

    public DeviceContext Device { get; }
    public BufferPool Pool { get; }
    public ModuleCache ModuleCache { get; }
    public BlockRegistry Registry { get; }
    public ConnectionGraph Connections { get; }
    public DirtyTracker Dirty { get; }
    public LibraryHandleCache Libraries { get; }

    // === Block Registry Facade ===

    /// <summary>
    /// Register a block. Called by block constructors.
    /// Delegates to Registry.Register → fires StructureChanged → DirtyTracker marks structure dirty.
    /// </summary>
    public void RegisterBlock(ICudaBlock block);

    /// <summary>
    /// Unregister a block. Called by block Dispose().
    /// Also removes connections and block description.
    /// </summary>
    public void UnregisterBlock(Guid blockId);

    // === Connection Facade ===

    public void Connect(Guid srcBlockId, string srcPort, Guid tgtBlockId, string tgtPort);
    public void Disconnect(Guid srcBlockId, string srcPort, Guid tgtBlockId, string tgtPort);

    // === Parameter Tracking ===

    /// <summary>
    /// Called by BlockParameter.ValueChanged → BlockBuilder event wiring.
    /// Marks the parameter dirty for Hot/Warm Update.
    /// </summary>
    public void OnParameterChanged(Guid blockId, string paramName);

    /// <summary>
    /// Called when a captured node's parameters change and it needs recapture.
    /// Marks the captured node dirty for Recapture Update.
    /// </summary>
    public void OnCapturedNodeChanged(Guid blockId, Guid capturedHandleId);

    // === Block Descriptions (for structure change detection) ===

    public void SetBlockDescription(Guid blockId, BlockDescription description);
    public BlockDescription? GetBlockDescription(Guid blockId);

    // === External Buffer Bindings ===

    /// <summary>
    /// Bind an external buffer to a block port. Used for graph inputs managed by the user.
    /// </summary>
    public void SetExternalBuffer(Guid blockId, string portName, CUdeviceptr pointer);
    public IReadOnlyDictionary<(Guid BlockId, string PortName), CUdeviceptr> ExternalBuffers { get; }

    // === Dispose ===
    public void Dispose();
}
```

### CudaEngineOptions

```csharp
public sealed class CudaEngineOptions
{
    /// <summary>
    /// CUDA device ordinal (default: 0 = first GPU).
    /// </summary>
    public int DeviceId { get; init; }
}
```

---

## Context Services

### BlockRegistry

```csharp
public sealed class BlockRegistry
{
    /// <summary>
    /// Fired when a block is registered or unregistered. DirtyTracker subscribes.
    /// </summary>
    public event Action? StructureChanged;

    public IReadOnlyDictionary<Guid, ICudaBlock> Blocks { get; }
    public int Count { get; }
    public IEnumerable<ICudaBlock> All { get; }

    public void Register(ICudaBlock block);
    public bool Unregister(Guid blockId);
    public ICudaBlock? Get(Guid blockId);
}
```

### ConnectionGraph

```csharp
public sealed class ConnectionGraph
{
    public event Action? StructureChanged;

    public IReadOnlyList<Connection> Connections { get; }
    public int Count { get; }

    public void Connect(Guid srcBlockId, string srcPort, Guid tgtBlockId, string tgtPort);
    public bool Disconnect(Guid srcBlockId, string srcPort, Guid tgtBlockId, string tgtPort);
    public int RemoveBlock(Guid blockId);
    public IEnumerable<Connection> GetOutgoing(Guid blockId);
    public IEnumerable<Connection> GetIncoming(Guid blockId);
}
```

### Connection

```csharp
public sealed record Connection(
    Guid SourceBlockId,
    string SourcePort,
    Guid TargetBlockId,
    string TargetPort);
```

### DirtyTracker

```csharp
public sealed class DirtyTracker
{
    public bool IsStructureDirty { get; }       // starts true → first build
    public bool AreParametersDirty { get; }     // true if any parameters dirty
    public bool AreCapturedNodesDirty { get; }  // true if any captured nodes need recapture

    public void Subscribe(BlockRegistry registry, ConnectionGraph connectionGraph);
    public void MarkParameterDirty(DirtyParameter param);
    public IReadOnlySet<DirtyParameter> GetDirtyParameters();

    /// <summary>
    /// Mark a captured node dirty for Recapture Update.
    /// </summary>
    public void MarkCapturedNodeDirty(DirtyCapturedNode node);

    /// <summary>
    /// Get the set of captured nodes that need recapture.
    /// </summary>
    public IReadOnlySet<DirtyCapturedNode> GetDirtyCapturedNodes();

    /// <summary>
    /// Clear structure dirty. Also clears parameters and captured nodes since rebuild applies all values.
    /// </summary>
    public void ClearStructureDirty();

    /// <summary>
    /// Clear parameter dirty flags after Hot/Warm Update.
    /// </summary>
    public void ClearParametersDirty();

    /// <summary>
    /// Clear captured node dirty flags after Recapture Update.
    /// </summary>
    public void ClearCapturedNodesDirty();
}
```

### DirtyParameter

```csharp
public readonly record struct DirtyParameter(Guid BlockId, string ParamName);
```

### DirtyCapturedNode

```csharp
/// <summary>
/// Identifies a captured node that needs recapture: which block and which captured operation.
/// </summary>
public readonly record struct DirtyCapturedNode(Guid BlockId, Guid CapturedHandleId);
```

---

## Buffers

### GpuBuffer\<T\>

```csharp
public sealed class GpuBuffer<T> : IDisposable where T : unmanaged
{
    public Guid Id { get; }
    public CUdeviceptr Pointer { get; }
    public long SizeInBytes { get; }
    public int ElementCount { get; }

    public void Upload(ReadOnlySpan<T> data);
    public void Download(Span<T> destination);

    public void Dispose();
}
```

### BufferPool

```csharp
public sealed class BufferPool : IDisposable
{
    public BufferPool(DeviceContext device);

    public GpuBuffer<T> Acquire<T>(int elementCount) where T : unmanaged;
    public void Release<T>(GpuBuffer<T> buffer) where T : unmanaged;

    /// <summary>
    /// Acquire an AppendBuffer with the given max capacity and lifetime.
    /// Allocates both the data buffer (maxCapacity elements) and a 1-element uint counter buffer.
    /// </summary>
    public AppendBuffer<T> AcquireAppend<T>(int maxCapacity, BufferLifetime lifetime) where T : unmanaged;

    public long TotalAllocatedBytes { get; }
    public int RentedCount { get; }

    public void Dispose();
}
```

### IAppendBuffer

```csharp
/// <summary>
/// Non-generic interface for AppendBuffer. Used for type-erased engine operations.
/// </summary>
public interface IAppendBuffer
{
    /// <summary>Device pointer to the data buffer.</summary>
    CUdeviceptr DataPointer { get; }

    /// <summary>Device pointer to the GPU atomic counter (single uint).</summary>
    CUdeviceptr CounterPointer { get; }

    /// <summary>Maximum number of elements the buffer can hold.</summary>
    int MaxCapacity { get; }

    /// <summary>
    /// Read the counter value from GPU (synchronous download).
    /// Clamps to MaxCapacity and stores result in LastReadCount.
    /// Returns the clamped count.
    /// </summary>
    int ReadCount();

    /// <summary>Last value returned by ReadCount() (clamped to MaxCapacity).</summary>
    int LastReadCount { get; }

    /// <summary>Raw counter value from GPU before clamping. Useful for overflow detection.</summary>
    int LastRawCount { get; }

    /// <summary>True if LastRawCount exceeded MaxCapacity on the last ReadCount().</summary>
    bool DidOverflow { get; }
}
```

### AppendBuffer\<T\>

```csharp
/// <summary>
/// GPU buffer with an atomic append counter. Wraps a data GpuBuffer and a counter GpuBuffer.
/// Kernels atomicAdd to the counter and write data at the returned index.
/// The counter is reset to zero via a CUDA memset node before each graph launch.
/// </summary>
public sealed class AppendBuffer<T> : IAppendBuffer, IDisposable where T : unmanaged
{
    /// <summary>The underlying data buffer.</summary>
    public GpuBuffer<T> Data { get; }

    /// <summary>The 1-element uint buffer used as an atomic counter on GPU.</summary>
    public GpuBuffer<uint> Counter { get; }

    /// <summary>Maximum number of elements the data buffer can hold.</summary>
    public int MaxCapacity { get; }

    /// <summary>The CLR element type (typeof(T)).</summary>
    public Type ElementType { get; }

    /// <summary>Lifetime policy for this buffer.</summary>
    public BufferLifetime Lifetime { get; }

    // --- IAppendBuffer ---
    public CUdeviceptr DataPointer { get; }
    public CUdeviceptr CounterPointer { get; }

    /// <summary>
    /// Read counter from GPU, clamp to MaxCapacity.
    /// Sets LastReadCount, LastRawCount, and DidOverflow.
    /// </summary>
    public int ReadCount();

    public int LastReadCount { get; }
    public int LastRawCount { get; }
    public bool DidOverflow { get; }

    /// <summary>
    /// Download only the valid elements (0..LastReadCount) from the data buffer.
    /// Must call ReadCount() first.
    /// </summary>
    public T[] DownloadValid();

    /// <summary>
    /// Set the read count manually (internal, used for testing).
    /// </summary>
    internal void SetReadCount(int count);

    public void Dispose();
}
```

### BufferLifetime

```csharp
/// <summary>
/// Controls how long an AppendBuffer lives relative to graph rebuilds.
/// </summary>
public enum BufferLifetime
{
    /// <summary>Buffer is re-allocated on every cold rebuild.</summary>
    PerBuild,

    /// <summary>Buffer persists across cold rebuilds (re-used if compatible).</summary>
    Persistent
}
```

### DType

```csharp
public enum DType
{
    F32, F64, S32, U32, S64, U64, S16, U16, S8, U8
}

public static class DTypeExtensions
{
    public static int SizeInBytes(this DType dtype);
    public static DType FromClrType<T>() where T : unmanaged;
}
```

---

## Blocks

### ICudaBlock

```csharp
public interface ICudaBlock : IDisposable
{
    Guid Id { get; }
    string TypeName { get; }

    /// <summary>
    /// VL NodeContext — auto-injected by VL as first constructor parameter.
    /// Provides identity, logging, and app access.
    /// </summary>
    NodeContext NodeContext { get; }

    IReadOnlyList<IBlockPort> Inputs { get; }
    IReadOnlyList<IBlockPort> Outputs { get; }
    IReadOnlyList<IBlockParameter> Parameters { get; }

    /// <summary>
    /// Debug info written by CudaEngine after each frame.
    /// </summary>
    IBlockDebugInfo DebugInfo { get; set; }
}
```

### IBlockPort

```csharp
public interface IBlockPort
{
    Guid BlockId { get; }
    string Name { get; }
    PortDirection Direction { get; }
    PinType Type { get; }
}
```

### BlockPort

```csharp
public sealed class BlockPort : IBlockPort
{
    public Guid BlockId { get; }
    public string Name { get; }
    public PortDirection Direction { get; }
    public PinType Type { get; }

    /// <summary>
    /// Which kernel handle (→ node) this port maps to. Set by BlockBuilder.
    /// </summary>
    internal Guid KernelNodeId { get; set; }

    /// <summary>
    /// Which parameter index on the kernel this port maps to.
    /// </summary>
    internal int KernelParamIndex { get; set; }

    public BlockPort(Guid blockId, string name, PortDirection direction, PinType type);
}
```

### IBlockParameter

```csharp
public interface IBlockParameter
{
    string Name { get; }
    Type ValueType { get; }
    object Value { get; set; }
    bool IsDirty { get; }
    void ClearDirty();
}
```

### BlockParameter\<T\>

```csharp
public sealed class BlockParameter<T> : IBlockParameter where T : unmanaged
{
    public string Name { get; }
    public Type ValueType => typeof(T);

    /// <summary>
    /// Typed value with change tracking. Setting fires ValueChanged event.
    /// </summary>
    public T TypedValue { get; set; }

    /// <summary>
    /// Fired when value changes. BlockBuilder wires this → CudaContext.OnParameterChanged.
    /// </summary>
    public event Action<BlockParameter<T>>? ValueChanged;

    public bool IsDirty { get; }
    public void ClearDirty();

    /// <summary>
    /// IBlockParameter explicit implementation (boxes).
    /// </summary>
    object IBlockParameter.Value { get; set; }

    internal Guid KernelNodeId { get; set; }
    internal int KernelParamIndex { get; set; }

    public BlockParameter(string name, T defaultValue = default);
}
```

### IBlockDebugInfo

```csharp
public interface IBlockDebugInfo
{
    BlockState State { get; }
    string? StateMessage { get; }
    TimeSpan LastExecutionTime { get; }

    /// <summary>
    /// Per-port append counts after the last launch. Null if the block has no append outputs.
    /// Key = port name, Value = last read count (clamped to MaxCapacity).
    /// </summary>
    IReadOnlyDictionary<string, int>? AppendCounts { get; }
}
```

### BlockDebugInfo

```csharp
public sealed class BlockDebugInfo : IBlockDebugInfo
{
    public BlockState State { get; set; } = BlockState.NotCompiled;
    public string? StateMessage { get; set; }
    public TimeSpan LastExecutionTime { get; set; }

    /// <summary>
    /// Per-port append counts. Null if the block has no append outputs.
    /// </summary>
    public IReadOnlyDictionary<string, int>? AppendCounts { get; set; }
}
```

### Enums

```csharp
public enum PortDirection { Input, Output }

public enum PinKind { Buffer, Scalar }

public enum BlockState { OK, Warning, Error, NotCompiled }
```

### PinType

```csharp
public sealed class PinType : IEquatable<PinType>
{
    public PinKind Kind { get; }
    public DType DataType { get; }

    public PinType(PinKind kind, DType dataType);

    public static PinType Buffer(DType dataType);
    public static PinType Scalar(DType dataType);
    public static PinType Buffer<T>() where T : unmanaged;
    public static PinType Scalar<T>() where T : unmanaged;

    /// <summary>
    /// Both must be same kind and same data type.
    /// </summary>
    public bool IsCompatible(PinType other);

    public bool Equals(PinType? other);
    public override int GetHashCode();
}
```

---

## Blocks.Builder

### BlockBuilder

DSL for describing a block's kernels, ports, parameters, and internal connections. Used in block constructors. Call `Commit()` when done.

```csharp
public sealed class BlockBuilder
{
    public BlockBuilder(CudaContext context, ICudaBlock block);

    // === Add Kernel (Filesystem PTX) ===

    /// <summary>
    /// Add a kernel from a PTX file path. Returns a handle for binding ports.
    /// Automatically loads PTX and extracts entry point from metadata.
    /// </summary>
    public KernelHandle AddKernel(string ptxPath);

    // === Add Captured (Library Call via Stream Capture) ===

    /// <summary>
    /// Add a captured library operation. Returns a handle for binding ports.
    /// The captureAction is called during stream capture to record library calls.
    /// </summary>
    public CapturedHandle AddCaptured(Action<CUstream, CUdeviceptr[]> captureAction,
        CapturedNodeDescriptor descriptor);

    // === Buffer Ports (KernelPin) ===

    /// <summary>
    /// Define a buffer input port bound to a kernel parameter.
    /// </summary>
    public BlockPort Input<T>(string name, KernelPin pin) where T : unmanaged;

    /// <summary>
    /// Define a buffer output port bound to a kernel parameter.
    /// </summary>
    public BlockPort Output<T>(string name, KernelPin pin) where T : unmanaged;

    // === Buffer Ports (CapturedPin) ===

    /// <summary>
    /// Define a buffer input port bound to a captured operation parameter.
    /// </summary>
    public BlockPort Input<T>(string name, CapturedPin pin) where T : unmanaged;

    /// <summary>
    /// Define a buffer output port bound to a captured operation parameter.
    /// </summary>
    public BlockPort Output<T>(string name, CapturedPin pin) where T : unmanaged;

    // === Scalar Parameters ===

    /// <summary>
    /// Define a scalar input parameter bound to a kernel parameter.
    /// Changes trigger Hot Update, not graph rebuild.
    /// </summary>
    public BlockParameter<T> InputScalar<T>(string name, KernelPin pin, T defaultValue = default)
        where T : unmanaged;

    // === Append Buffer Outputs ===

    /// <summary>
    /// Define an append buffer output. Binds both a data pointer pin and a counter pointer pin
    /// to kernel parameters. The engine will allocate an AppendBuffer, wire both pointers,
    /// and add a memset node to reset the counter before each launch.
    /// </summary>
    public AppendOutputPort AppendOutput<T>(string name, KernelPin dataPin, KernelPin counterPin,
        int maxCapacity) where T : unmanaged;

    // === Internal Connections ===

    /// <summary>
    /// Connect two kernel parameters within this block (output of one kernel → input of another).
    /// </summary>
    public void Connect(KernelPin source, KernelPin target);

    // === Finalize ===

    /// <summary>
    /// Finalize the block description. Wires parameter change events to CudaContext dirty tracking.
    /// Stores BlockDescription for structure change detection. Throws on double-commit.
    /// </summary>
    public void Commit();

    // === Read-only access ===

    public IReadOnlyList<KernelHandle> Kernels { get; }
    public IReadOnlyList<CapturedHandle> CapturedHandles { get; }
    public IReadOnlyList<BlockPort> Inputs { get; }
    public IReadOnlyList<BlockPort> Outputs { get; }
    public IReadOnlyList<IBlockParameter> Parameters { get; }
    public IReadOnlyList<(Guid SrcKernel, int SrcParam, Guid TgtKernel, int TgtParam)> InternalConnections { get; }
}
```

### KernelHandle

```csharp
public sealed class KernelHandle
{
    public Guid Id { get; }
    public string PtxPath { get; }
    public KernelDescriptor Descriptor { get; }

    /// <summary>
    /// Grid dimensions (default 1×1×1). Set before Commit() to configure launch grid.
    /// </summary>
    public uint GridDimX { get; set; }
    public uint GridDimY { get; set; }
    public uint GridDimZ { get; set; }

    /// <summary>
    /// Reference to an input parameter by index.
    /// </summary>
    public KernelPin In(int index);

    /// <summary>
    /// Reference to an output parameter by index.
    /// </summary>
    public KernelPin Out(int index);

    public KernelHandle(string ptxPath, KernelDescriptor descriptor);
}
```

### KernelPin

```csharp
public readonly struct KernelPin
{
    public Guid KernelHandleId { get; }
    public int ParamIndex { get; }

    public KernelPin(Guid kernelHandleId, int paramIndex);
}
```

### KernelEntry

```csharp
/// <summary>
/// A kernel entry in a BlockDescription. Stores everything the engine
/// needs to recreate the KernelNode during ColdRebuild.
/// </summary>
public sealed class KernelEntry
{
    public Guid HandleId { get; }
    public string PtxPath { get; }
    public string EntryPoint { get; }
    public uint GridDimX { get; }
    public uint GridDimY { get; }
    public uint GridDimZ { get; }

    public KernelEntry(Guid handleId, string ptxPath, string entryPoint,
        uint gridDimX, uint gridDimY, uint gridDimZ);

    /// <summary>
    /// Structural equality ignores HandleId (changes every construction).
    /// Compares PTX path, entry point, and grid dimensions.
    /// </summary>
    public bool StructuralEquals(KernelEntry? other);
}
```

### BlockDescription

```csharp
/// <summary>
/// Immutable snapshot of a block's structural description.
/// Used for change detection (Hot-Swap) and graph rebuilding.
/// </summary>
public sealed class BlockDescription
{
    /// <summary>
    /// Kernel entries in AddKernel order. Deterministic, ordered list.
    /// </summary>
    public IReadOnlyList<KernelEntry> KernelEntries { get; }

    /// <summary>
    /// Captured library operation entries in AddCaptured order.
    /// </summary>
    public IReadOnlyList<CapturedEntry> CapturedEntries { get; }

    /// <summary>
    /// Port entries: (Name, Direction, PinType).
    /// </summary>
    public IReadOnlyList<(string Name, PortDirection Direction, PinType Type)> Ports { get; }

    /// <summary>
    /// Internal connections using kernel indices (not GUIDs) for stable comparison.
    /// (SrcKernelIndex, SrcParam, TgtKernelIndex, TgtParam)
    /// </summary>
    public IReadOnlyList<(int SrcKernelIndex, int SrcParam, int TgtKernelIndex, int TgtParam)> InternalConnections { get; }

    /// <summary>
    /// Append buffer descriptors. One per AppendOutput port.
    /// Used by CudaEngine to allocate AppendBuffers and wire memset nodes.
    /// </summary>
    public IReadOnlyList<AppendBufferInfo> AppendBuffers { get; }

    public BlockDescription(
        IReadOnlyList<KernelEntry> kernelEntries,
        IReadOnlyList<(string Name, PortDirection Direction, PinType Type)> ports,
        IReadOnlyList<(int SrcKernelIndex, int SrcParam, int TgtKernelIndex, int TgtParam)> internalConnections,
        IReadOnlyList<AppendBufferInfo>? appendBuffers = null,
        IReadOnlyList<CapturedEntry>? capturedEntries = null);

    /// <summary>
    /// Structural equality: same kernels (path + grid), captured ops, ports, connections,
    /// and append buffers. HandleIds are ignored — they change every construction.
    /// </summary>
    public bool StructuralEquals(BlockDescription? other);
}
```

### AppendBufferInfo

```csharp
/// <summary>
/// Describes an append buffer binding within a block. Stored in BlockDescription
/// for structural comparison and used by CudaEngine during ColdRebuild to allocate
/// and wire AppendBuffers.
/// </summary>
public sealed class AppendBufferInfo
{
    /// <summary>Block that owns this append buffer.</summary>
    public Guid BlockId { get; }

    /// <summary>Name of the append output port (data port).</summary>
    public string PortName { get; }

    /// <summary>Name of the counter port (derived from data port name).</summary>
    public string CountPortName { get; }

    /// <summary>KernelHandle ID for the data pointer parameter.</summary>
    public Guid DataKernelHandleId { get; }

    /// <summary>Parameter index for the data pointer on the kernel.</summary>
    public int DataParamIndex { get; }

    /// <summary>KernelHandle ID for the counter pointer parameter.</summary>
    public Guid CounterKernelHandleId { get; }

    /// <summary>Parameter index for the counter pointer on the kernel.</summary>
    public int CounterParamIndex { get; }

    /// <summary>Maximum number of elements the append buffer can hold.</summary>
    public int MaxCapacity { get; }

    /// <summary>Size in bytes of each element.</summary>
    public int ElementSize { get; }

    /// <summary>
    /// Structural equality: compares port names, param indices, max capacity, and element size.
    /// Ignores BlockId and KernelHandleIds (change every construction).
    /// </summary>
    public bool StructuralEquals(AppendBufferInfo? other);
}
```

### AppendOutputPort

```csharp
/// <summary>
/// Returned by BlockBuilder.AppendOutput(). Provides access to the data port
/// and append buffer metadata.
/// </summary>
public sealed class AppendOutputPort
{
    /// <summary>The data output port (can be connected to downstream block inputs).</summary>
    public BlockPort DataPort { get; }

    /// <summary>The append buffer descriptor for this port.</summary>
    public AppendBufferInfo Info { get; }

    /// <summary>Name of the counter port.</summary>
    public string CountPortName { get; }
}
```

### CapturedHandle

```csharp
/// <summary>
/// Represents a captured library operation added to a block via BlockBuilder.AddCaptured().
/// Provides In()/Out()/Scalar() to reference specific parameters for port binding.
/// Analogous to KernelHandle for kernel operations.
/// </summary>
public sealed class CapturedHandle
{
    public Guid Id { get; }
    public CapturedNodeDescriptor Descriptor { get; }
    public Action<CUstream, CUdeviceptr[]> CaptureAction { get; }

    public CapturedHandle(CapturedNodeDescriptor descriptor, Action<CUstream, CUdeviceptr[]> captureAction);

    /// <summary>
    /// Reference to an input parameter by index. Returns the flat buffer binding index.
    /// Buffer bindings layout: [inputs..., outputs..., scalars...].
    /// </summary>
    public CapturedPin In(int index);

    /// <summary>
    /// Reference to an output parameter by index. Returns the flat buffer binding index.
    /// Buffer bindings layout: [inputs..., outputs..., scalars...].
    /// </summary>
    public CapturedPin Out(int index);

    /// <summary>
    /// Reference to a scalar parameter by index. Returns the flat buffer binding index.
    /// Buffer bindings layout: [inputs..., outputs..., scalars...].
    /// </summary>
    public CapturedPin Scalar(int index);
}
```

### CapturedPin

```csharp
/// <summary>
/// Reference to a specific parameter on a captured handle within a block.
/// Analogous to KernelPin for KernelHandle.
/// </summary>
public readonly struct CapturedPin
{
    public Guid CapturedHandleId { get; }

    /// <summary>
    /// Flat index into the CapturedNode.BufferBindings array.
    /// Layout: [inputs..., outputs..., scalars...].
    /// </summary>
    public int ParamIndex { get; }

    /// <summary>
    /// Which category this pin references: Input, Output, or Scalar.
    /// </summary>
    public CapturedPinCategory Category { get; }

    public CapturedPin(Guid capturedHandleId, int paramIndex, CapturedPinCategory category);
}
```

### CapturedPinCategory

```csharp
/// <summary>
/// Identifies which parameter list a CapturedPin references.
/// </summary>
public enum CapturedPinCategory
{
    Input,
    Output,
    Scalar,
}
```

### CapturedEntry

```csharp
/// <summary>
/// A captured library operation entry in a block description.
/// Stores the handle ID, descriptor, and capture action needed to recreate the CapturedNode.
/// </summary>
public sealed class CapturedEntry
{
    public Guid HandleId { get; }
    public CapturedNodeDescriptor Descriptor { get; }
    public Action<CUstream, CUdeviceptr[]> CaptureAction { get; }

    public CapturedEntry(Guid handleId, CapturedNodeDescriptor descriptor,
        Action<CUstream, CUdeviceptr[]> captureAction);

    /// <summary>
    /// Structural equality ignores HandleId (changes every construction).
    /// Compares descriptor debug name and parameter counts.
    /// </summary>
    public bool StructuralEquals(CapturedEntry? other);
}
```

---

## Graph (Phase 0 + Phase 1 + Phase 4a)

### IGraphNode

```csharp
/// <summary>
/// Common interface for all graph node types: KernelNode and CapturedNode.
/// Allows GraphCompiler and CudaEngine to work polymorphically with both.
/// </summary>
internal interface IGraphNode : IDisposable
{
    Guid Id { get; }
    string DebugName { get; }
}
```

### KernelNode

```csharp
public sealed class KernelNode : IGraphNode, IDisposable
{
    public Guid Id { get; }
    public string DebugName { get; }
    public CUfunction Function { get; }

    public uint GridDimX { get; set; }
    public uint GridDimY { get; set; }
    public uint GridDimZ { get; set; }
    public uint BlockDimX { get; set; }
    public uint BlockDimY { get; set; }
    public uint BlockDimZ { get; set; }
    public uint SharedMemBytes { get; set; }

    public void SetPointer(int index, CUdeviceptr pointer);
    public void SetScalar<T>(int index, T value) where T : unmanaged;
    public CUDA_KERNEL_NODE_PARAMS BuildNodeParams();

    public void Dispose(); // frees pinned native memory
}
```

### CapturedNode

```csharp
/// <summary>
/// A graph node created by stream capture of library calls (cuBLAS, cuFFT, etc.).
/// Inserted into the CUDA graph as a child graph node via AddChildGraphNode.
/// Supports Recapture: re-execute the capture action and update the child graph
/// without a full Cold Rebuild.
/// </summary>
public sealed class CapturedNode : IGraphNode, IDisposable
{
    public Guid Id { get; }
    public string DebugName { get; set; }
    public CapturedNodeDescriptor Descriptor { get; }

    /// <summary>
    /// The capture action executed during stream capture.
    /// Receives the stream handle and buffer bindings (one CUdeviceptr per descriptor param).
    /// Order: inputs first, then outputs, then scalars.
    /// </summary>
    internal Action<CUstream, CUdeviceptr[]> CaptureAction { get; }

    /// <summary>
    /// Buffer bindings set by CudaEngine before Capture(). Order matches descriptor:
    /// [inputs..., outputs..., scalars...].
    /// </summary>
    internal CUdeviceptr[] BufferBindings { get; set; }

    /// <summary>
    /// The most recently captured child graph handle. Replaced on Recapture.
    /// The CapturedNode owns this handle and destroys it on Dispose/Recapture.
    /// </summary>
    internal CUgraph CapturedGraph { get; }

    /// <summary>
    /// Whether this node has been captured at least once.
    /// </summary>
    internal bool HasCapturedGraph { get; }

    public CapturedNode(
        CapturedNodeDescriptor descriptor,
        Action<CUstream, CUdeviceptr[]> captureAction,
        string? debugName = null);

    /// <summary>
    /// Execute the capture action via stream capture and produce a child graph.
    /// Disposes the previous captured graph if any.
    /// </summary>
    internal CUgraph Capture(CUstream stream);

    public void Dispose();
}
```

### CapturedNodeDescriptor

```csharp
/// <summary>
/// Describes the input/output/scalar parameters of a captured library operation.
/// Analogous to KernelDescriptor for KernelNodes, but for stream-captured operations.
/// </summary>
public sealed class CapturedNodeDescriptor
{
    public string DebugName { get; }
    public IReadOnlyList<CapturedParam> Inputs { get; }
    public IReadOnlyList<CapturedParam> Outputs { get; }
    public IReadOnlyList<CapturedParam> Scalars { get; }

    /// <summary>
    /// Total parameter count across all categories.
    /// </summary>
    public int TotalParamCount { get; }

    public CapturedNodeDescriptor(
        string debugName,
        IReadOnlyList<CapturedParam>? inputs = null,
        IReadOnlyList<CapturedParam>? outputs = null,
        IReadOnlyList<CapturedParam>? scalars = null);
}
```

### CapturedParam

```csharp
/// <summary>
/// Describes a single parameter of a captured library operation.
/// </summary>
public sealed class CapturedParam
{
    public string Name { get; }
    public string Type { get; }
    public bool IsPointer { get; }

    public CapturedParam(string name, string type, bool isPointer = true);

    /// <summary>
    /// Create a pointer parameter (buffer input/output).
    /// </summary>
    public static CapturedParam Pointer(string name, string type);

    /// <summary>
    /// Create a scalar parameter (e.g., alpha/beta for BLAS).
    /// </summary>
    public static CapturedParam Scalar(string name, string type);
}
```

### StreamCaptureHelper

```csharp
/// <summary>
/// Thin helper for CUDA stream capture. Executes a work action between
/// cuStreamBeginCapture and cuStreamEndCapture, returning the captured graph handle.
/// </summary>
internal static class StreamCaptureHelper
{
    /// <summary>
    /// Execute a work action in stream capture mode and return the captured CUgraph handle.
    /// The work action receives the stream handle and should direct all library calls to it.
    /// The caller owns the returned CUgraph and must destroy it when done.
    /// Uses CUstreamCaptureMode.Relaxed for multi-threaded capture support.
    /// </summary>
    public static CUgraph CaptureToGraph(CUstream stream, Action<CUstream> work);

    /// <summary>
    /// Destroy a CUgraph handle that was returned from CaptureToGraph.
    /// </summary>
    public static void DestroyGraph(CUgraph graph);

    /// <summary>
    /// Add a child graph node to a parent graph using a captured CUgraph.
    /// </summary>
    public static CUgraphNode AddChildGraphNode(CUgraph parentGraph,
        CUgraphNode[]? dependencies, CUgraph childGraph);

    /// <summary>
    /// Update a child graph node in an executable graph (Recapture update).
    /// Calls cuGraphExecChildGraphNodeSetParams.
    /// </summary>
    public static void UpdateChildGraphNode(CUgraphExec exec,
        CUgraphNode node, CUgraph newChildGraph);
}
```

### CapturedDependency

```csharp
/// <summary>
/// Declares a dependency between a captured node and a kernel node (or vice versa).
/// SourceNodeId must complete before TargetNodeId can begin.
/// Both IDs can refer to either KernelNode or CapturedNode.
/// </summary>
internal readonly record struct CapturedDependency(Guid SourceNodeId, Guid TargetNodeId);
```

### Edge

```csharp
public sealed record Edge(
    Guid SourceNodeId,
    int SourceParamIndex,
    Guid TargetNodeId,
    int TargetParamIndex);
```

### GraphBuilder

```csharp
public sealed class GraphBuilder
{
    public GraphBuilder(DeviceContext device, ModuleCache moduleCache);

    // === Kernel Nodes ===

    public KernelNode AddKernel(LoadedKernel loaded, string? debugName = null);
    public void AddEdge(KernelNode source, int sourceParam, KernelNode target, int targetParam);
    public void SetExternalBuffer(KernelNode node, int paramIndex, CUdeviceptr pointer);

    // === Captured Nodes (Phase 4a) ===

    /// <summary>
    /// Add a captured library operation node. Created via stream capture during graph compilation.
    /// </summary>
    public CapturedNode AddCaptured(CapturedNodeDescriptor descriptor,
        Action<CUstream, CUdeviceptr[]> captureAction, string? debugName = null);

    /// <summary>
    /// Declare that a kernel node depends on a captured node completing first.
    /// </summary>
    public void AddCapturedDependency(CapturedNode captured, KernelNode kernel);

    /// <summary>
    /// Declare that a captured node depends on a kernel node completing first.
    /// </summary>
    public void AddCapturedDependency(KernelNode kernel, CapturedNode captured);

    /// <summary>
    /// Assign an external buffer to a specific captured node parameter.
    /// </summary>
    public void SetExternalBuffer(CapturedNode node, int paramIndex, CUdeviceptr ptr);

    // === Memset Nodes (Phase 3.1) ===

    /// <summary>
    /// Add a memset node to the graph. Used to zero-initialize append buffer counters
    /// before kernel execution. Returns a descriptor for dependency wiring.
    /// </summary>
    internal MemsetDescriptor AddMemset(CUdeviceptr dst, uint value, uint elemSize,
        ulong width, string? debugName = null);

    /// <summary>
    /// Add a dependency edge: the memset must complete before the kernel node executes.
    /// </summary>
    internal void AddMemsetDependency(MemsetDescriptor memset, KernelNode kernel);

    // === Read-only access ===

    public IReadOnlyList<KernelNode> Nodes { get; }
    public IReadOnlyList<Edge> Edges { get; }
    public IReadOnlyList<CapturedNode> CapturedNodes { get; }
    internal IReadOnlyList<CapturedDependency> CapturedDependencies { get; }

    /// <summary>
    /// All memset descriptors added via AddMemset().
    /// </summary>
    internal IReadOnlyList<MemsetDescriptor> MemsetDescriptors { get; }

    public ValidationResult Validate();
}
```

### MemsetDescriptor

```csharp
/// <summary>
/// Describes a memset node in the CUDA graph. Used to zero-initialize
/// append buffer counters before kernel execution.
/// </summary>
internal sealed class MemsetDescriptor
{
    public Guid Id { get; }

    /// <summary>Device pointer to the memory to be set.</summary>
    public CUdeviceptr Destination { get; }

    /// <summary>Value to fill (typically 0 for counter reset).</summary>
    public uint Value { get; }

    /// <summary>Size of each element in bytes (e.g., 4 for uint).</summary>
    public uint ElementSize { get; }

    /// <summary>Number of elements to set.</summary>
    public ulong Width { get; }

    /// <summary>Debug name for diagnostics.</summary>
    public string? DebugName { get; }

    /// <summary>
    /// Kernel nodes that depend on this memset (must execute after memset completes).
    /// </summary>
    public IReadOnlyList<Guid> DependentKernelNodeIds { get; }
}
```

### GraphDescription

```csharp
/// <summary>
/// Immutable snapshot of a graph description ready for compilation.
/// Produced by GraphBuilder.Build().
/// </summary>
internal sealed class GraphDescription
{
    public IReadOnlyList<KernelNode> Nodes { get; }
    public IReadOnlyList<Edge> Edges { get; }
    public IReadOnlyDictionary<(Guid NodeId, int ParamIndex), CUdeviceptr> ExternalBuffers { get; }

    /// <summary>
    /// Memset descriptors for append buffer counter resets.
    /// </summary>
    public IReadOnlyList<MemsetDescriptor> MemsetDescriptors { get; }

    public GraphDescription(
        IReadOnlyList<KernelNode> nodes,
        IReadOnlyList<Edge> edges,
        IReadOnlyDictionary<(Guid NodeId, int ParamIndex), CUdeviceptr> externalBuffers,
        IReadOnlyList<MemsetDescriptor>? memsetDescriptors = null);
}
```

### GraphCompiler

```csharp
public sealed class GraphCompiler
{
    public GraphCompiler(DeviceContext device, BufferPool pool);

    /// <summary>
    /// Validates, topologically sorts, allocates intermediate buffers, compiles to CUDA Graph.
    /// For CapturedNodes: executes stream capture on each, inserts as ChildGraphNodes,
    /// wires CapturedDependency edges, and applies external buffer bindings.
    /// </summary>
    public CompiledGraph Compile(GraphBuilder builder);
}
```

### CompiledGraph

```csharp
public sealed class CompiledGraph : IDisposable
{
    public Guid Id { get; }

    public void Launch(ManagedCuda.CudaStream stream);

    /// <summary>
    /// Hot Update: change scalar value without rebuild.
    /// </summary>
    public void UpdateScalar<T>(Guid nodeId, int paramIndex, T value) where T : unmanaged;

    /// <summary>
    /// Warm Update: change buffer pointer without rebuild.
    /// </summary>
    public void UpdatePointer(Guid nodeId, int paramIndex, CUdeviceptr newPointer);

    /// <summary>
    /// Warm Update: change grid dimensions without rebuild.
    /// </summary>
    public void UpdateGrid(Guid nodeId, uint gridX, uint gridY, uint gridZ);

    /// <summary>
    /// Recapture Update: replace a captured node's child graph in the executable graph.
    /// Used when a CapturedNode's parameters change and it needs re-stream-capture.
    /// </summary>
    internal void RecaptureNode(Guid nodeId, CUgraph newChildGraph);

    /// <summary>
    /// Native handles for memset nodes in the compiled CUDA graph.
    /// Used internally to track memset → kernel dependencies.
    /// </summary>
    internal IReadOnlyDictionary<Guid, CUgraphNode> MemsetNodeHandles { get; }

    /// <summary>
    /// Native handles for captured (child graph) nodes in the compiled CUDA graph.
    /// Used internally for Recapture updates.
    /// </summary>
    internal IReadOnlyDictionary<Guid, CUgraphNode> CapturedNodeHandles { get; }

    /// <summary>
    /// Disposes CUgraph and CUgraphExec. Does NOT dispose KernelNodes or CapturedNodes.
    /// Node lifetime is owned by CudaEngine.
    /// </summary>
    public void Dispose();
}
```

---

## PTX

### PtxLoader

```csharp
public static class PtxLoader
{
    /// <summary>
    /// Load a PTX file + companion JSON metadata. Returns LoadedKernel.
    /// </summary>
    public static LoadedKernel Load(DeviceContext device, string ptxPath);
}
```

### ModuleCache

```csharp
public sealed class ModuleCache : IDisposable
{
    public ModuleCache(DeviceContext device);

    public LoadedKernel GetOrLoad(string ptxPath);
    public bool Contains(string ptxPath);

    public void Dispose();
}
```

### KernelDescriptor

```csharp
public sealed class KernelDescriptor
{
    public string EntryPoint { get; }
    public IReadOnlyList<KernelParamDescriptor> Parameters { get; }
    public int BlockSizeX { get; }
    public int SharedMemoryBytes { get; }

    // ... constructed from JSON metadata
}
```

### KernelParamDescriptor

```csharp
public sealed class KernelParamDescriptor
{
    public int Index { get; }
    public string PtxType { get; }
    public bool IsPointer { get; }
    public int? Alignment { get; }
    public string? Name { get; }
}
```

---

## Device

### DeviceContext

```csharp
public sealed class DeviceContext : IDisposable
{
    public DeviceContext(int deviceId = 0);

    public ManagedCuda.CudaContext CudaContext { get; }
    public int DeviceId { get; }
    public string DeviceName { get; }
    public long TotalMemory { get; }

    public CUmodule LoadModule(string ptxPath);
    public CUfunction GetFunction(CUmodule module, string entryPoint);

    public void Dispose();
}
```

---

## Constructor Pattern (Blocks)

All blocks follow this pattern. In VL, `NodeContext` is injected automatically as the first constructor parameter.

```csharp
public class VectorAddBlock : ICudaBlock, IDisposable
{
    private readonly CudaContext _ctx;

    public Guid Id { get; } = Guid.NewGuid();
    public string TypeName => "VectorAdd";
    public NodeContext NodeContext { get; }

    public IReadOnlyList<IBlockPort> Inputs => _inputs;
    public IReadOnlyList<IBlockPort> Outputs => _outputs;
    public IReadOnlyList<IBlockParameter> Parameters => _parameters;
    public IBlockDebugInfo DebugInfo { get; set; }

    private readonly List<IBlockPort> _inputs = new();
    private readonly List<IBlockPort> _outputs = new();
    private readonly List<IBlockParameter> _parameters = new();

    public VectorAddBlock(NodeContext nodeContext, CudaContext ctx)
    {
        _ctx = ctx;
        NodeContext = nodeContext;

        var builder = new BlockBuilder(ctx, this);

        var kernel = builder.AddKernel("kernels/vector_add.ptx");

        _inputs.Add(builder.Input<float>("A", kernel.In(0)));
        _inputs.Add(builder.Input<float>("B", kernel.In(1)));
        _outputs.Add(builder.Output<float>("C", kernel.Out(2)));
        _parameters.Add(builder.InputScalar<uint>("N", kernel.In(3), 1024u));

        builder.Commit();
        ctx.RegisterBlock(this);
    }

    public void Dispose() => _ctx.UnregisterBlock(Id);
}
```

---

## Libraries (Phase 4a)

### LibraryHandleCache

```csharp
/// <summary>
/// Lazy-initialized cache for CUDA library handles. One instance per CudaContext.
/// Handles are expensive to create, so we cache them and reuse across captures.
/// Each handle is created on first access and disposed when the cache is disposed.
/// </summary>
public sealed class LibraryHandleCache : IDisposable
{
    /// <summary>
    /// Get or create the cuBLAS handle.
    /// </summary>
    public CudaBlas GetOrCreateBlas();

    /// <summary>
    /// Get or create a cuFFT 1D plan with the given configuration.
    /// Plans are cached by (nx, type, batch) since they are configuration-dependent.
    /// </summary>
    public CudaFFTPlan1D GetOrCreateFFT1D(int nx, cufftType type, int batch = 1);

    /// <summary>
    /// Get or create the cuSPARSE context.
    /// </summary>
    public CudaSparseContext GetOrCreateSparse();

    /// <summary>
    /// Get or create the cuRAND device generator.
    /// </summary>
    public CudaRandDevice GetOrCreateRand(GeneratorType type = GeneratorType.PseudoDefault);

    /// <summary>
    /// Get or create the cuSOLVER dense handle.
    /// </summary>
    public CudaSolveDense GetOrCreateSolveDense();

    public void Dispose();
}
```

### BlasOperations

Static wrappers for cuBLAS. Each method returns a `CapturedHandle` registered on the `BlockBuilder`.

```csharp
/// <summary>
/// High-level cuBLAS wrappers that produce CapturedHandle entries for BlockBuilder.
/// Each operation wraps a stream-captured cuBLAS call as a CapturedNode.
/// Buffer bindings follow descriptor order: [inputs..., outputs..., scalars...].
/// </summary>
public static class BlasOperations  // VL.Cuda.Core.Libraries.Blas
{
    /// <summary>
    /// Single-precision matrix multiply: C = alpha*A*B + beta*C.
    /// A is (M x K), B is (K x N), C is (M x N). Column-major layout.
    /// Buffers: In[0]=A, In[1]=B, Out[0]=C.
    /// </summary>
    public static CapturedHandle Sgemm(
        BlockBuilder builder, LibraryHandleCache libs,
        int m, int n, int k,
        float alpha = 1.0f, float beta = 0.0f,
        Operation transA = Operation.NonTranspose,
        Operation transB = Operation.NonTranspose);

    /// <summary>
    /// Double-precision matrix multiply: C = alpha*A*B + beta*C.
    /// Buffers: In[0]=A, In[1]=B, Out[0]=C.
    /// </summary>
    public static CapturedHandle Dgemm(
        BlockBuilder builder, LibraryHandleCache libs,
        int m, int n, int k,
        double alpha = 1.0, double beta = 0.0,
        Operation transA = Operation.NonTranspose,
        Operation transB = Operation.NonTranspose);

    /// <summary>
    /// Single-precision matrix-vector multiply: y = alpha*A*x + beta*y.
    /// A is (M x N), x is (N), y is (M).
    /// Buffers: In[0]=A, In[1]=x, Out[0]=y.
    /// </summary>
    public static CapturedHandle Sgemv(
        BlockBuilder builder, LibraryHandleCache libs,
        int m, int n,
        float alpha = 1.0f, float beta = 0.0f,
        Operation transA = Operation.NonTranspose);

    /// <summary>
    /// Single-precision vector scaling: x = alpha * x.
    /// Buffers: Out[0]=x (in-place).
    /// </summary>
    public static CapturedHandle Sscal(
        BlockBuilder builder, LibraryHandleCache libs,
        int n, float alpha = 1.0f);
}
```

### FftOperations

Static wrappers for cuFFT. Each method returns a `CapturedHandle` registered on the `BlockBuilder`.

```csharp
/// <summary>
/// High-level cuFFT wrappers that produce CapturedHandle entries for BlockBuilder.
/// Each operation wraps a stream-captured cuFFT plan execution as a CapturedNode.
/// </summary>
public static class FftOperations  // VL.Cuda.Core.Libraries.FFT
{
    /// <summary>
    /// 1D FFT Forward transform (out-of-place).
    /// Buffers: In[0]=input, Out[0]=output.
    /// </summary>
    public static CapturedHandle Forward1D(
        BlockBuilder builder, LibraryHandleCache libs,
        int nx, cufftType type = cufftType.C2C, int batch = 1);

    /// <summary>
    /// 1D FFT Inverse transform (out-of-place).
    /// Buffers: In[0]=input, Out[0]=output.
    /// </summary>
    public static CapturedHandle Inverse1D(
        BlockBuilder builder, LibraryHandleCache libs,
        int nx, cufftType type = cufftType.C2C, int batch = 1);

    /// <summary>
    /// 1D Real-to-Complex FFT (out-of-place).
    /// Input: N real values, Output: N/2+1 complex values.
    /// Buffers: In[0]=input (float*), Out[0]=output (float2*).
    /// </summary>
    public static CapturedHandle R2C1D(
        BlockBuilder builder, LibraryHandleCache libs,
        int nx, int batch = 1);

    /// <summary>
    /// 1D Complex-to-Real inverse FFT (out-of-place).
    /// Input: N/2+1 complex values, Output: N real values.
    /// Buffers: In[0]=input (float2*), Out[0]=output (float*).
    /// </summary>
    public static CapturedHandle C2R1D(
        BlockBuilder builder, LibraryHandleCache libs,
        int nx, int batch = 1);
}
```

### SparseOperations

Static wrappers for cuSPARSE. Each method returns a `CapturedHandle` registered on the `BlockBuilder`.

```csharp
/// <summary>
/// High-level cuSPARSE wrappers that produce CapturedHandle entries for BlockBuilder.
/// Sparse matrix operations via stream capture.
/// </summary>
public static class SparseOperations  // VL.Cuda.Core.Libraries.Sparse
{
    /// <summary>
    /// Sparse matrix-vector multiply (SpMV): y = alpha*A*x + beta*y.
    /// A is sparse CSR (M x N, nnz non-zeros), x and y are dense vectors.
    /// Buffers: In[0]=csrValues, In[1]=csrRowOffsets, In[2]=csrColInd, In[3]=x,
    ///          Out[0]=y, Out[1]=workspace.
    /// </summary>
    public static CapturedHandle SpMV(
        BlockBuilder builder, LibraryHandleCache libs,
        int m, int n, int nnz, int workspaceSizeBytes);
}
```

### RandOperations

Static wrappers for cuRAND. Each method returns a `CapturedHandle` registered on the `BlockBuilder`.

```csharp
/// <summary>
/// High-level cuRAND wrappers that produce CapturedHandle entries for BlockBuilder.
/// Random number generation via stream capture.
/// </summary>
public static class RandOperations  // VL.Cuda.Core.Libraries.Rand
{
    /// <summary>
    /// Generate uniform random float values in [0, 1).
    /// Buffers: Out[0]=output.
    /// </summary>
    public static CapturedHandle GenerateUniform(
        BlockBuilder builder, LibraryHandleCache libs, int count);

    /// <summary>
    /// Generate normal-distributed random float values.
    /// Buffers: Out[0]=output.
    /// </summary>
    public static CapturedHandle GenerateNormal(
        BlockBuilder builder, LibraryHandleCache libs,
        int count, float mean = 0.0f, float stddev = 1.0f);
}
```

### SolveOperations

Static wrappers for cuSOLVER. Each method returns a `CapturedHandle` registered on the `BlockBuilder`.

```csharp
/// <summary>
/// High-level cuSOLVER wrappers that produce CapturedHandle entries for BlockBuilder.
/// Dense linear algebra solvers via stream capture.
/// </summary>
public static class SolveOperations  // VL.Cuda.Core.Libraries.Solve
{
    /// <summary>
    /// LU factorization of an M x N matrix: A = P * L * U.
    /// Buffers: Out[0]=A (in-place LU), Out[1]=workspace, Out[2]=devIpiv, Out[3]=devInfo.
    /// </summary>
    public static CapturedHandle Sgetrf(
        BlockBuilder builder, LibraryHandleCache libs, int m, int n);

    /// <summary>
    /// Solve a linear system A*X = B using LU factorization.
    /// A must have been factored by Sgetrf first.
    /// Buffers: In[0]=A (LU), In[1]=Ipiv, Out[0]=B (in-place solution X), Out[1]=Info.
    /// </summary>
    public static CapturedHandle Sgetrs(
        BlockBuilder builder, LibraryHandleCache libs,
        int n, int nrhs, Operation trans = Operation.NonTranspose);
}
```

---

## Constructor Pattern (Captured Block)

Blocks that use library calls via stream capture follow this pattern. The `CapturedHandle` is used the same way as `KernelHandle`.

```csharp
public class MatMulBlock : ICudaBlock, IDisposable
{
    private readonly CudaContext _ctx;

    public Guid Id { get; } = Guid.NewGuid();
    public string TypeName => "MatMul";
    public NodeContext NodeContext { get; }

    public IReadOnlyList<IBlockPort> Inputs => _inputs;
    public IReadOnlyList<IBlockPort> Outputs => _outputs;
    public IReadOnlyList<IBlockParameter> Parameters => Array.Empty<IBlockParameter>();
    public IBlockDebugInfo DebugInfo { get; set; }

    private readonly List<IBlockPort> _inputs = new();
    private readonly List<IBlockPort> _outputs = new();

    public MatMulBlock(NodeContext nodeContext, CudaContext ctx, int m, int n, int k)
    {
        _ctx = ctx;
        NodeContext = nodeContext;

        var builder = new BlockBuilder(ctx, this);

        // Use cuBLAS Sgemm via stream capture
        var sgemm = BlasOperations.Sgemm(builder, ctx.Libraries, m, n, k);

        _inputs.Add(builder.Input<float>("A", sgemm.In(0)));
        _inputs.Add(builder.Input<float>("B", sgemm.In(1)));
        _outputs.Add(builder.Output<float>("C", sgemm.Out(0)));

        builder.Commit();
        ctx.RegisterBlock(this);
    }

    public void Dispose() => _ctx.UnregisterBlock(Id);
}
```

---

## Planned (Phase 3+)

The following types are designed but not yet implemented:

- **InputHandle\<T\> / OutputHandle\<T\>** — VL handle-flow for typed links (Phase 3)
- **GridConfig / GridSizeMode** — Auto-grid from buffer size (Phase 3)
- **Regions** — If/While/For conditional graph execution (Phase 3)
- **Composite blocks** — AddChild / ConnectChildren / ExposeInput/Output (Phase 3)
- **ProfilingPipeline** — Async GPU event readback (Phase 3+)
- **IDebugInfo** — Full engine debug info (Phase 3+)
- **IlgpuCompiler** — ILGPU IR → PTX compilation (Phase 4b)
- **NvrtcCache** — User CUDA C++ compilation (Phase 4b)
- **CudaDX11Interop** — DX11/Stride graphics sharing (Phase 5)
- **ProfilingLevel** — None/Summary/PerBlock/PerKernel/DeepAsync/DeepSync (Phase 3+)
- **IsCodeDirty / MarkCodeDirty** — Patchable kernel recompile tracking (Phase 4b)

See `PHASES.md` for the implementation roadmap.
