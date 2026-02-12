# C# API Reference

> **Status**: Phase 0 + Phase 1 + Phase 2 + Phase 3.1 implemented and tested (167 tests).
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
  │     BlockDescription
  │     AppendBufferInfo
  │     AppendOutputPort
  │
  ├── Graph
  │     KernelNode
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
  ├── Captured (Phase 4a)
  │     CapturedHandle, CapturedNodeDescriptor
  │     StreamCaptureHelper, LibraryHandleCache
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
    /// Priority: Structure dirty → Cold Rebuild; Parameters dirty → Hot Update; then Launch.
    ///
    /// 1. If IsStructureDirty → ColdRebuild (GraphBuilder → Compile → new CompiledGraph)
    /// 2. Else if AreParametersDirty → Hot/Warm Update (CompiledGraph.UpdateScalar)
    /// 3. If compiled graph exists → Launch + Synchronize + Distribute DebugInfo
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
    /// Disposes compiled graph, owned kernel nodes, stream, and CudaContext.
    /// </summary>
    public void Dispose();
}
```

**ColdRebuild pipeline** (internal):
1. Dispose old CompiledGraph + KernelNodes + owned AppendBuffers
2. For each registered block: iterate `BlockDescription.KernelEntries` → create KernelNodes (via GraphBuilder) with grid dims
3. Add intra-block connections (from `BlockDescription.InternalConnections` using kernel indices)
4. Add inter-block connections (from `ConnectionGraph` using BlockPort → KernelNode mapping)
5. Allocate and wire AppendBuffers (`AllocateAndWireAppendBuffer()` for each `AppendBufferInfo`)
6. Add memset nodes for AppendBuffer counters (via `GraphBuilder.AddMemset()` + `AddMemsetDependency()`)
7. Apply external buffer bindings
8. Apply current parameter values to KernelNodes
9. Compile via GraphCompiler → new CompiledGraph
10. Distribute debug info (BlockState.OK or BlockState.Error) to all blocks

**Post-Launch** (internal):
- `ReadbackAppendCounters()` — synchronously reads append counter values from GPU
- `BuildStateMessage()` — constructs state message including append counts for debug info

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

    public void Subscribe(BlockRegistry registry, ConnectionGraph connectionGraph);
    public void MarkParameterDirty(DirtyParameter param);
    public IReadOnlySet<DirtyParameter> GetDirtyParameters();

    /// <summary>
    /// Clear structure dirty. Also clears parameters since rebuild applies all values.
    /// </summary>
    public void ClearStructureDirty();

    /// <summary>
    /// Clear parameter dirty flags after Hot/Warm Update.
    /// </summary>
    public void ClearParametersDirty();
}
```

### DirtyParameter

```csharp
public readonly record struct DirtyParameter(Guid BlockId, string ParamName);
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

    // === Buffer Ports ===

    /// <summary>
    /// Define a buffer input port bound to a kernel parameter.
    /// </summary>
    public BlockPort Input<T>(string name, KernelPin pin) where T : unmanaged;

    /// <summary>
    /// Define a buffer output port bound to a kernel parameter.
    /// </summary>
    public BlockPort Output<T>(string name, KernelPin pin) where T : unmanaged;

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
        IReadOnlyList<AppendBufferInfo>? appendBuffers = null);

    /// <summary>
    /// Structural equality: same kernels (path + grid), ports, connections, and append buffers.
    /// HandleIds are ignored — they change every construction.
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

---

## Graph (Phase 0 + Phase 1)

### KernelNode

```csharp
public sealed class KernelNode : IDisposable
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

    public KernelNode AddKernel(LoadedKernel loaded, string? debugName = null);
    public void AddEdge(KernelNode source, int sourceParam, KernelNode target, int targetParam);
    public void SetExternalBuffer(KernelNode node, int paramIndex, CUdeviceptr pointer);

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

    public IReadOnlyList<KernelNode> Nodes { get; }
    public IReadOnlyList<Edge> Edges { get; }

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
    /// Native handles for memset nodes in the compiled CUDA graph.
    /// Used internally to track memset → kernel dependencies.
    /// </summary>
    internal IReadOnlyDictionary<Guid, CUgraphNode> MemsetNodeHandles { get; }

    /// <summary>
    /// Disposes CUgraph and CUgraphExec. Does NOT dispose KernelNodes.
    /// KernelNode lifetime is owned by CudaEngine.
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

## Planned (Phase 3+)

The following types are designed but not yet implemented:

- **InputHandle\<T\> / OutputHandle\<T\>** — VL handle-flow for typed links (Phase 3)
- **GridConfig / GridSizeMode** — Auto-grid from buffer size (Phase 3)
- **Regions** — If/While/For conditional graph execution (Phase 3)
- **Composite blocks** — AddChild / ConnectChildren / ExposeInput/Output (Phase 3)
- **ProfilingPipeline** — Async GPU event readback (Phase 3+)
- **IDebugInfo** — Full engine debug info (Phase 3+)
- **CapturedHandle / StreamCaptureHelper** — cuBLAS/cuFFT via Stream Capture (Phase 4a)
- **IlgpuCompiler** — ILGPU IR → PTX compilation (Phase 4b)
- **NvrtcCache** — User CUDA C++ compilation (Phase 4b)
- **CudaDX11Interop** — DX11/Stride graphics sharing (Phase 5)
- **ProfilingLevel** — None/Summary/PerBlock/PerKernel/DeepAsync/DeepSync (Phase 3+)
- **IsCodeDirty / MarkCodeDirty** — Patchable kernel recompile tracking (Phase 4b)

See `PHASES.md` for the implementation roadmap.
