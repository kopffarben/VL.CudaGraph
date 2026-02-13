# C# API Reference

> **Status**: Phase 0 + Phase 1 + Phase 2 + Phase 3.1 + Phase 4a + Phase 4b implemented and tested (248 passed, 9 skipped/NVRTC; 257 total).
> Types marked *(Phase 3+)* or *(Phase 5)* are planned but not yet implemented.

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
  │     DirtyCodeEntry
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
  │     KernelSource
  │       ├── FilesystemPtx
  │       ├── IlgpuMethod
  │       └── NvrtcSource
  │     PtxLoader
  │     ModuleCache
  │     PtxMetadata
  │     KernelDescriptor
  │     KernelParamDescriptor
  │     ParamDirection
  │     LoadedKernel
  │
  ├── PTX.Compilation (internal)
  │     IlgpuCompiler
  │     NvrtcCache
  │
  ├── Device
  │     DeviceContext
  │
  ├── Region (Phase 3+ — planned, see VL-UX.md)
  │     CudaGraphRegion
  │     CudaFunction
  │     CudaFunctionInvoke
  │     UploadNode
  │     DownloadNode
  │     FrameDelayNode
  │     GridSizeMode (Auto, Fixed)
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
    /// Priority: Structure > Code > Recapture > Parameters > Launch.
    ///
    /// 1. If IsStructureDirty → ColdRebuild (GraphBuilder → Compile → new CompiledGraph)
    /// 2. Else if IsCodeDirty → CodeRebuild (invalidate caches) → ColdRebuild
    /// 3. Else if AreCapturedNodesDirty → RecaptureNodes (re-stream-capture → UpdateChildGraphNode)
    /// 4. Else if AreParametersDirty → Hot/Warm Update (CompiledGraph.UpdateScalar)
    /// 5. If compiled graph exists → Launch + Synchronize + Distribute DebugInfo
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
2. For each registered block: iterate `BlockDescription.KernelEntries` → load kernel via `LoadKernelFromSource()` → create KernelNodes (via GraphBuilder) with grid dims
3. For ILGPU kernels: call `InitializeIlgpuParams()` to set implicit `_kernel_length` and ArrayView struct length fields
4. For each registered block: iterate `BlockDescription.CapturedEntries` → create CapturedNodes (via GraphBuilder)
5. Add intra-block connections (from `BlockDescription.InternalConnections` using kernel indices)
6. Add inter-block connections (from `ConnectionGraph` using BlockPort → KernelNode/CapturedNode mapping)
7. Allocate and wire AppendBuffers (`AllocateAndWireAppendBuffer()` for each `AppendBufferInfo`)
8. Add memset nodes for AppendBuffer counters (via `GraphBuilder.AddMemset()` + `AddMemsetDependency()`)
9. Apply external buffer bindings (for both KernelNodes and CapturedNodes)
10. Apply current parameter values to KernelNodes
11. Compile via GraphCompiler → new CompiledGraph (stream-captures CapturedNodes → ChildGraphNodes)
12. Distribute debug info (BlockState.OK or BlockState.Error) to all blocks

**CodeRebuild** (internal):
- Iterates `DirtyTracker.GetDirtyCodeEntries()`
- For each dirty entry: invalidates the matching compiler cache based on `KernelSource` type:
  - `IlgpuMethod` → `IlgpuCompiler.Invalidate(methodHash)`
  - `NvrtcSource` → `NvrtcCache.Invalidate(sourceHash)`
  - `FilesystemPtx` → `ModuleCache.Evict(ptxPath)`
- Clears code dirty flags, then triggers ColdRebuild

**LoadKernelFromSource** (internal):
- Dispatches on `KernelEntry.Source` type:
  - `FilesystemPtx` → `ModuleCache.GetOrLoad(ptxPath)`
  - `IlgpuMethod` → `IlgpuCompiler.GetOrCompile(method, descriptor)`
  - `NvrtcSource` → `NvrtcCache.GetOrCompile(cudaSource, entryPoint, descriptor)`

**InitializeIlgpuParams** (internal, static):
- Called during ColdRebuild for ILGPU kernels only
- Sets param 0 (`_kernel_length`) to `GridDimX * BlockDimX` (total thread count for Index1D)
- For each ArrayView struct param (SizeBytes=16): calls `SetArrayView(i, default, long.MaxValue)`
- Pointer fields are overwritten later when external buffers are bound

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

    /// <summary>
    /// ILGPU IR → PTX compiler. Caches compiled kernels by method hash.
    /// Internal: accessed by CudaEngine during ColdRebuild/CodeRebuild.
    /// </summary>
    internal IlgpuCompiler IlgpuCompiler { get; }

    /// <summary>
    /// NVRTC CUDA C++ → PTX compiler. Caches compiled kernels by source hash.
    /// Internal: accessed by CudaEngine during ColdRebuild/CodeRebuild.
    /// </summary>
    internal NvrtcCache NvrtcCache { get; }

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

    /// <summary>
    /// Called when a kernel's source code changes and needs recompilation.
    /// Triggers Code Rebuild → Cold Rebuild in CudaEngine.
    /// </summary>
    public void OnCodeChanged(Guid blockId, Guid handleId, KernelSource newSource);

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

    /// <summary>
    /// Disposes Libraries, IlgpuCompiler, NvrtcCache, Pool, ModuleCache, Device.
    /// </summary>
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
/// <summary>
/// Tracks dirty state for the CUDA pipeline. Subscribes to StructureChanged
/// events from BlockRegistry and ConnectionGraph. CudaEngine reads these
/// flags each frame to decide: Cold Rebuild vs Code Rebuild vs Recapture vs Hot/Warm Update vs no-op.
/// Priority: Structure > Code > Recapture > Parameters.
/// </summary>
public sealed class DirtyTracker
{
    public bool IsStructureDirty { get; }       // starts true → first build
    public bool IsCodeDirty { get; }            // true if any code entries dirty
    public bool AreParametersDirty { get; }     // true if any parameters dirty
    public bool AreCapturedNodesDirty { get; }  // true if any captured nodes need recapture

    public void Subscribe(BlockRegistry registry, ConnectionGraph connectionGraph);

    /// <summary>
    /// Mark a specific parameter as dirty (called by CudaContext.OnParameterChanged).
    /// </summary>
    public void MarkParameterDirty(DirtyParameter param);

    /// <summary>
    /// Mark a captured node dirty for Recapture Update.
    /// </summary>
    public void MarkCapturedNodeDirty(DirtyCapturedNode node);

    /// <summary>
    /// Mark a kernel source as dirty (code changed, needs recompilation).
    /// Called by CudaContext.OnCodeChanged.
    /// </summary>
    public void MarkCodeDirty(DirtyCodeEntry entry);

    public IReadOnlySet<DirtyParameter> GetDirtyParameters();
    public IReadOnlySet<DirtyCapturedNode> GetDirtyCapturedNodes();
    public IReadOnlySet<DirtyCodeEntry> GetDirtyCodeEntries();

    /// <summary>
    /// Clear structure dirty. Also clears code, parameters, and captured nodes
    /// since rebuild applies all current values.
    /// </summary>
    public void ClearStructureDirty();

    /// <summary>
    /// Clear code dirty flags after Code Rebuild.
    /// </summary>
    public void ClearCodeDirty();

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

### DirtyCodeEntry

```csharp
/// <summary>
/// Identifies a kernel whose source code changed: which block, which kernel handle, and the new source.
/// Used by CudaEngine.CodeRebuild() to invalidate the correct compiler cache entry.
/// </summary>
public readonly record struct DirtyCodeEntry(Guid BlockId, Guid KernelHandleId, KernelSource NewSource);
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
    CUdeviceptr DataPointer { get; }
    CUdeviceptr CounterPointer { get; }
    int MaxCapacity { get; }
    int ReadCount();
    int LastReadCount { get; }
    int LastRawCount { get; }
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
    public GpuBuffer<T> Data { get; }
    public GpuBuffer<uint> Counter { get; }
    public int MaxCapacity { get; }
    public Type ElementType { get; }
    public BufferLifetime Lifetime { get; }

    public CUdeviceptr DataPointer { get; }
    public CUdeviceptr CounterPointer { get; }
    public int ReadCount();
    public int LastReadCount { get; }
    public int LastRawCount { get; }
    public bool DidOverflow { get; }
    public T[] DownloadValid();
    internal void SetReadCount(int count);
    public void Dispose();
}
```

### BufferLifetime

```csharp
public enum BufferLifetime { PerBuild, Persistent }
```

### DType

```csharp
public enum DType { F32, F64, S32, U32, S64, U64, S16, U16, S8, U8 }

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
    NodeContext NodeContext { get; }
    IReadOnlyList<IBlockPort> Inputs { get; }
    IReadOnlyList<IBlockPort> Outputs { get; }
    IReadOnlyList<IBlockParameter> Parameters { get; }
    IBlockDebugInfo DebugInfo { get; set; }
}
```

### IBlockPort / BlockPort

```csharp
public interface IBlockPort
{
    Guid BlockId { get; }
    string Name { get; }
    PortDirection Direction { get; }
    PinType Type { get; }
}

public sealed class BlockPort : IBlockPort
{
    public Guid BlockId { get; }
    public string Name { get; }
    public PortDirection Direction { get; }
    public PinType Type { get; }
    internal Guid KernelNodeId { get; set; }
    internal int KernelParamIndex { get; set; }
    public BlockPort(Guid blockId, string name, PortDirection direction, PinType type);
}
```

### IBlockParameter / BlockParameter\<T\>

```csharp
public interface IBlockParameter
{
    string Name { get; }
    Type ValueType { get; }
    object Value { get; set; }
    bool IsDirty { get; }
    void ClearDirty();
}

public sealed class BlockParameter<T> : IBlockParameter where T : unmanaged
{
    public string Name { get; }
    public Type ValueType => typeof(T);
    public T TypedValue { get; set; }
    public event Action<BlockParameter<T>>? ValueChanged;
    public bool IsDirty { get; }
    public void ClearDirty();
    object IBlockParameter.Value { get; set; }
    internal Guid KernelNodeId { get; set; }
    internal int KernelParamIndex { get; set; }
    public BlockParameter(string name, T defaultValue = default);
}
```

### IBlockDebugInfo / BlockDebugInfo

```csharp
public interface IBlockDebugInfo
{
    BlockState State { get; }
    string? StateMessage { get; }
    TimeSpan LastExecutionTime { get; }
    IReadOnlyDictionary<string, int>? AppendCounts { get; }
}

public sealed class BlockDebugInfo : IBlockDebugInfo
{
    public BlockState State { get; set; } = BlockState.NotCompiled;
    public string? StateMessage { get; set; }
    public TimeSpan LastExecutionTime { get; set; }
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
    /// Source: KernelSource.FilesystemPtx.
    /// </summary>
    public KernelHandle AddKernel(string ptxPath);

    // === Add Kernel (ILGPU Method → PTX) ===

    /// <summary>
    /// Add a kernel from an ILGPU-compiled C# method. Returns a handle for binding ports.
    /// The method must follow ILGPU kernel conventions (Index1D first parameter, etc.).
    /// The user-provided descriptor is stored for recompilation; the In()/Out() indices
    /// are auto-remapped to account for ILGPU's (pointer, length) parameter expansion.
    /// Source: KernelSource.IlgpuMethod.
    /// </summary>
    public KernelHandle AddKernel(MethodInfo kernelMethod, KernelDescriptor descriptor);

    // === Add Kernel (NVRTC CUDA C++ → PTX) ===

    /// <summary>
    /// Add a kernel from CUDA C++ source compiled via NVRTC. Returns a handle for binding ports.
    /// Source: KernelSource.NvrtcSource.
    /// </summary>
    public KernelHandle AddKernelFromCuda(string cudaSource, string entryPoint,
        KernelDescriptor descriptor);

    // === Add Captured (Library Call via Stream Capture) ===

    /// <summary>
    /// Add a captured library operation. Returns a handle for binding ports.
    /// </summary>
    public CapturedHandle AddCaptured(Action<CUstream, CUdeviceptr[]> captureAction,
        CapturedNodeDescriptor descriptor);

    // === Buffer Ports ===

    public BlockPort Input<T>(string name, KernelPin pin) where T : unmanaged;
    public BlockPort Output<T>(string name, KernelPin pin) where T : unmanaged;
    public BlockPort Input<T>(string name, CapturedPin pin) where T : unmanaged;
    public BlockPort Output<T>(string name, CapturedPin pin) where T : unmanaged;

    // === Scalar Parameters ===

    public BlockParameter<T> InputScalar<T>(string name, KernelPin pin, T defaultValue = default)
        where T : unmanaged;

    // === Append Buffer Outputs ===

    public AppendOutputPort AppendOutput<T>(string name, KernelPin dataPin, KernelPin counterPin,
        int maxCapacity) where T : unmanaged;

    // === Internal Connections ===

    public void Connect(KernelPin source, KernelPin target);

    // === Finalize ===

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
/// <summary>
/// Represents a kernel added to a block via BlockBuilder.AddKernel().
/// Provides In()/Out() to reference specific parameters for port binding.
/// For ILGPU kernels, an index remap translates user-facing parameter indices
/// to expanded PTX indices (accounting for implicit kernel_length and ArrayView length params).
/// </summary>
public sealed class KernelHandle
{
    public Guid Id { get; }

    /// <summary>
    /// The kernel source: FilesystemPtx, IlgpuMethod, or NvrtcSource.
    /// </summary>
    public KernelSource Source { get; }

    /// <summary>
    /// The kernel descriptor. For ILGPU/NVRTC: user-provided (before expansion).
    /// For filesystem PTX: from JSON metadata.
    /// </summary>
    public KernelDescriptor Descriptor { get; }

    public uint GridDimX { get; set; } = 1;
    public uint GridDimY { get; set; } = 1;
    public uint GridDimZ { get; set; } = 1;

    /// <summary>
    /// Create from filesystem PTX path (backward-compatible). Wraps in FilesystemPtx.
    /// </summary>
    public KernelHandle(string ptxPath, KernelDescriptor descriptor);

    /// <summary>
    /// Create from any kernel source, with optional index remap for ILGPU.
    /// </summary>
    public KernelHandle(KernelSource source, KernelDescriptor descriptor, int[]? indexRemap = null);

    /// <summary>
    /// Reference to an input parameter by index.
    /// For ILGPU: remapped to expanded PTX index (shifted by +1 for kernel_length).
    /// </summary>
    public KernelPin In(int index);

    /// <summary>
    /// Reference to an output parameter by index.
    /// For ILGPU: remapped to expanded PTX index (shifted by +1 for kernel_length).
    /// </summary>
    public KernelPin Out(int index);
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

    /// <summary>
    /// The kernel source: FilesystemPtx, IlgpuMethod, or NvrtcSource.
    /// </summary>
    public KernelSource Source { get; }

    public string EntryPoint { get; }
    public uint GridDimX { get; }
    public uint GridDimY { get; }
    public uint GridDimZ { get; }

    /// <summary>
    /// The user-provided kernel descriptor. Null for filesystem PTX (comes from JSON sidecar).
    /// For ILGPU/NVRTC: stores the original descriptor so CudaEngine can recompile on cache miss.
    /// </summary>
    public KernelDescriptor? Descriptor { get; }

    public KernelEntry(Guid handleId, KernelSource source, string entryPoint,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        KernelDescriptor? descriptor = null);

    /// <summary>
    /// Structural equality ignores HandleId. Compares source cache key, entry point, grid dims.
    /// </summary>
    public bool StructuralEquals(KernelEntry? other);
}
```

### BlockDescription

```csharp
public sealed class BlockDescription
{
    public IReadOnlyList<KernelEntry> KernelEntries { get; }
    public IReadOnlyList<CapturedEntry> CapturedEntries { get; }
    public IReadOnlyList<(string Name, PortDirection Direction, PinType Type)> Ports { get; }
    public IReadOnlyList<(int SrcKernelIndex, int SrcParam, int TgtKernelIndex, int TgtParam)> InternalConnections { get; }
    public IReadOnlyList<AppendBufferInfo> AppendBuffers { get; }

    public BlockDescription(
        IReadOnlyList<KernelEntry> kernelEntries,
        IReadOnlyList<(string Name, PortDirection Direction, PinType Type)> ports,
        IReadOnlyList<(int SrcKernelIndex, int SrcParam, int TgtKernelIndex, int TgtParam)> internalConnections,
        IReadOnlyList<AppendBufferInfo>? appendBuffers = null,
        IReadOnlyList<CapturedEntry>? capturedEntries = null);

    public bool StructuralEquals(BlockDescription? other);
}
```

### AppendBufferInfo

```csharp
public sealed class AppendBufferInfo
{
    public Guid BlockId { get; }
    public string PortName { get; }
    public string CountPortName { get; }  // → "{PortName} Count"
    public Guid DataKernelHandleId { get; }
    public int DataParamIndex { get; }
    public Guid CounterKernelHandleId { get; }
    public int CounterParamIndex { get; }
    public int MaxCapacity { get; }
    public int ElementSize { get; }
    public bool StructuralEquals(AppendBufferInfo? other);
}
```

### AppendOutputPort

```csharp
public sealed class AppendOutputPort
{
    public BlockPort DataPort { get; }
    public AppendBufferInfo Info { get; }
    public string CountPortName { get; }
}
```

### CapturedHandle

```csharp
public sealed class CapturedHandle
{
    public Guid Id { get; }
    public CapturedNodeDescriptor Descriptor { get; }
    public Action<CUstream, CUdeviceptr[]> CaptureAction { get; }
    public CapturedHandle(CapturedNodeDescriptor descriptor, Action<CUstream, CUdeviceptr[]> captureAction);
    public CapturedPin In(int index);    // flat index: [inputs...]
    public CapturedPin Out(int index);   // flat index: [inputs.Length + outputs...]
    public CapturedPin Scalar(int index); // flat index: [inputs.Length + outputs.Length + scalars...]
}
```

### CapturedPin / CapturedPinCategory / CapturedEntry

```csharp
public readonly struct CapturedPin
{
    public Guid CapturedHandleId { get; }
    public int ParamIndex { get; }
    public CapturedPinCategory Category { get; }
    public CapturedPin(Guid capturedHandleId, int paramIndex, CapturedPinCategory category);
}

public enum CapturedPinCategory { Input, Output, Scalar }

public sealed class CapturedEntry
{
    public Guid HandleId { get; }
    public CapturedNodeDescriptor Descriptor { get; }
    public Action<CUstream, CUdeviceptr[]> CaptureAction { get; }
    public CapturedEntry(Guid handleId, CapturedNodeDescriptor descriptor,
        Action<CUstream, CUdeviceptr[]> captureAction);
    public bool StructuralEquals(CapturedEntry? other);
}
```

---

## Graph

### IGraphNode

```csharp
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
    public string DebugName { get; set; }
    public LoadedKernel LoadedKernel { get; }

    public uint GridDimX { get; set; }
    public uint GridDimY { get; set; }
    public uint GridDimZ { get; set; }
    public uint BlockDimX { get; set; }
    public uint BlockDimY { get; set; }
    public uint BlockDimZ { get; set; }
    public uint SharedMemoryBytes { get; set; }
    public int ParameterCount { get; }

    public KernelNode(LoadedKernel loadedKernel, string? debugName = null);

    /// <summary>
    /// Set a pointer parameter (buffer device pointer).
    /// </summary>
    public void SetPointer(int index, CUdeviceptr ptr);

    /// <summary>
    /// Set an ILGPU ArrayView struct parameter: writes the device pointer at offset 0
    /// and the length at offset 8 in the 16-byte struct slot.
    /// Used for ILGPU-compiled kernels where ArrayView&lt;T&gt; compiles to
    /// `.param .align 8 .b8 name[16]` — a struct containing {void* ptr, long length}.
    /// </summary>
    public void SetArrayView(int index, CUdeviceptr ptr, long length);

    /// <summary>
    /// Set a scalar parameter value.
    /// </summary>
    public void SetScalar<T>(int index, T value) where T : unmanaged;

    /// <summary>
    /// Get the IntPtr to the parameter array for CudaKernelNodeParams.kernelParams.
    /// </summary>
    public IntPtr GetParamsPtr();

    internal CudaKernelNodeParams BuildNodeParams();

    public void Dispose(); // frees pinned native memory
}
```

**ILGPU parameter layout in KernelNode:**
- Each param slot is sized by `KernelParamDescriptor.SizeBytes`. When `SizeBytes=0`: pointers use `IntPtr.Size`, scalars use type-based size. When `SizeBytes=16`: allocates a 16-byte slot for ILGPU ArrayView structs.
- `SetArrayView()` writes `{void* ptr (8 bytes at offset 0), long length (8 bytes at offset 8)}`.
- `SetPointer()` writes only the pointer (for standard non-ILGPU pointer params).

### CapturedNode

```csharp
public sealed class CapturedNode : IGraphNode, IDisposable
{
    public Guid Id { get; }
    public string DebugName { get; set; }
    public CapturedNodeDescriptor Descriptor { get; }
    internal Action<CUstream, CUdeviceptr[]> CaptureAction { get; }
    internal CUdeviceptr[] BufferBindings { get; set; }
    internal CUgraph CapturedGraph { get; }
    internal bool HasCapturedGraph { get; }

    public CapturedNode(CapturedNodeDescriptor descriptor,
        Action<CUstream, CUdeviceptr[]> captureAction, string? debugName = null);
    internal CUgraph Capture(CUstream stream);
    public void Dispose();
}
```

### CapturedNodeDescriptor / CapturedParam

```csharp
public sealed class CapturedNodeDescriptor
{
    public string DebugName { get; }
    public IReadOnlyList<CapturedParam> Inputs { get; }
    public IReadOnlyList<CapturedParam> Outputs { get; }
    public IReadOnlyList<CapturedParam> Scalars { get; }
    public int TotalParamCount { get; }
    public CapturedNodeDescriptor(string debugName,
        IReadOnlyList<CapturedParam>? inputs = null,
        IReadOnlyList<CapturedParam>? outputs = null,
        IReadOnlyList<CapturedParam>? scalars = null);
}

public sealed class CapturedParam
{
    public string Name { get; }
    public string Type { get; }
    public bool IsPointer { get; }
    public CapturedParam(string name, string type, bool isPointer = true);
    public static CapturedParam Pointer(string name, string type);
    public static CapturedParam Scalar(string name, string type);
}
```

### StreamCaptureHelper

```csharp
internal static class StreamCaptureHelper
{
    public static CUgraph CaptureToGraph(CUstream stream, Action<CUstream> work);
    public static void DestroyGraph(CUgraph graph);
    public static CUgraphNode AddChildGraphNode(CUgraph parentGraph,
        CUgraphNode[]? dependencies, CUgraph childGraph);
    public static void UpdateChildGraphNode(CUgraphExec exec,
        CUgraphNode node, CUgraph newChildGraph);
}
```

### CapturedDependency / Edge

```csharp
internal readonly record struct CapturedDependency(Guid SourceNodeId, Guid TargetNodeId);

public sealed record Edge(
    Guid SourceNodeId, int SourceParamIndex,
    Guid TargetNodeId, int TargetParamIndex);
```

### GraphBuilder

```csharp
public sealed class GraphBuilder
{
    public GraphBuilder(DeviceContext device, ModuleCache moduleCache);

    public KernelNode AddKernel(LoadedKernel loaded, string? debugName = null);
    public void AddEdge(KernelNode source, int sourceParam, KernelNode target, int targetParam);
    public void SetExternalBuffer(KernelNode node, int paramIndex, CUdeviceptr pointer);

    public CapturedNode AddCaptured(CapturedNodeDescriptor descriptor,
        Action<CUstream, CUdeviceptr[]> captureAction, string? debugName = null);
    public void AddCapturedDependency(CapturedNode captured, KernelNode kernel);
    public void AddCapturedDependency(KernelNode kernel, CapturedNode captured);
    public void SetExternalBuffer(CapturedNode node, int paramIndex, CUdeviceptr ptr);

    internal MemsetDescriptor AddMemset(CUdeviceptr dst, uint value, uint elemSize,
        ulong width, string? debugName = null);
    internal void AddMemsetDependency(MemsetDescriptor memset, KernelNode kernel);

    public IReadOnlyList<KernelNode> Nodes { get; }
    public IReadOnlyList<Edge> Edges { get; }
    public IReadOnlyList<CapturedNode> CapturedNodes { get; }
    internal IReadOnlyList<CapturedDependency> CapturedDependencies { get; }
    internal IReadOnlyList<MemsetDescriptor> MemsetDescriptors { get; }

    public ValidationResult Validate();
}
```

### MemsetDescriptor

```csharp
internal sealed class MemsetDescriptor
{
    public Guid Id { get; }
    public CUdeviceptr Destination { get; }
    public uint Value { get; }
    public uint ElementSize { get; }
    public ulong Width { get; }
    public string? DebugName { get; }
    public IReadOnlyList<Guid> DependentKernelNodeIds { get; }
}
```

### GraphCompiler

```csharp
public sealed class GraphCompiler
{
    public GraphCompiler(DeviceContext device, BufferPool pool);

    /// <summary>
    /// Validates, topologically sorts, allocates intermediate buffers, compiles to CUDA Graph.
    /// For CapturedNodes: executes stream capture, inserts as ChildGraphNodes.
    /// For ILGPU ArrayView struct params (SizeBytes=16): uses SetBufferOnNode helper
    /// which calls node.SetArrayView() to write both pointer and length.
    /// </summary>
    public CompiledGraph Compile(GraphBuilder builder);
}
```

**SetBufferOnNode** (internal, static helper in GraphCompiler):
Dispatches based on `KernelParamDescriptor.SizeBytes`:
- `SizeBytes == 16` → `node.SetArrayView(paramIndex, ptr, long.MaxValue)` (ILGPU ArrayView struct)
- Otherwise → `node.SetPointer(paramIndex, ptr)` (standard pointer)

### CompiledGraph

```csharp
public sealed class CompiledGraph : IDisposable
{
    public Guid Id { get; }
    public void Launch(ManagedCuda.CudaStream stream);
    public void UpdateScalar<T>(Guid nodeId, int paramIndex, T value) where T : unmanaged;
    public void UpdatePointer(Guid nodeId, int paramIndex, CUdeviceptr newPointer);
    public void UpdateGrid(Guid nodeId, uint gridX, uint gridY, uint gridZ);
    internal void RecaptureNode(Guid nodeId, CUgraph newChildGraph);
    internal IReadOnlyDictionary<Guid, CUgraphNode> MemsetNodeHandles { get; }
    internal IReadOnlyDictionary<Guid, CUgraphNode> CapturedNodeHandles { get; }
    public void Dispose();
}
```

---

## PTX

### KernelSource

```csharp
/// <summary>
/// Discriminated union describing how a kernel's PTX was produced.
/// Three variants: FilesystemPtx, IlgpuMethod, NvrtcSource.
/// </summary>
public abstract class KernelSource  // VL.Cuda.Core.PTX
{
    /// <summary>
    /// Returns a stable cache key for deduplication.
    /// </summary>
    public abstract string GetCacheKey();

    /// <summary>
    /// Human-readable name for diagnostics.
    /// </summary>
    public abstract string GetDebugName();

    /// <summary>
    /// Kernel loaded from a filesystem PTX + JSON pair (Triton, nvcc, hand-written).
    /// Cache key: "file:{fullPath}".
    /// </summary>
    public sealed class FilesystemPtx : KernelSource
    {
        public string PtxPath { get; }
        public FilesystemPtx(string ptxPath);
        public override string GetCacheKey();  // → "file:{Path.GetFullPath(PtxPath)}"
        public override string GetDebugName(); // → Path.GetFileNameWithoutExtension(PtxPath)
    }

    /// <summary>
    /// Kernel compiled from a C# method via ILGPU IR → PTX.
    /// Cache key: "ilgpu:{MethodHash}".
    /// </summary>
    public sealed class IlgpuMethod : KernelSource
    {
        public string MethodHash { get; }
        public MethodInfo KernelMethod { get; }
        public IlgpuMethod(string methodHash, MethodInfo kernelMethod);
        public override string GetCacheKey();  // → "ilgpu:{MethodHash}"
        public override string GetDebugName(); // → "ILGPU:{DeclaringType.Name}.{Method.Name}"
    }

    /// <summary>
    /// Kernel compiled from CUDA C++ source via NVRTC.
    /// Cache key: "nvrtc:{SourceHash}".
    /// </summary>
    public sealed class NvrtcSource : KernelSource
    {
        public string SourceHash { get; }
        public string CudaSource { get; }
        public string EntryPoint { get; }
        public NvrtcSource(string sourceHash, string cudaSource, string entryPoint);
        public override string GetCacheKey();  // → "nvrtc:{SourceHash}"
        public override string GetDebugName(); // → "NVRTC:{EntryPoint}"
    }
}
```

### PtxLoader

```csharp
public static class PtxLoader
{
    /// <summary>
    /// Load a PTX file + companion JSON metadata. Returns LoadedKernel.
    /// </summary>
    public static LoadedKernel Load(DeviceContext device, string ptxPath);

    /// <summary>
    /// Load PTX from bytes with a pre-parsed descriptor.
    /// Used by IlgpuCompiler and NvrtcCache after compilation.
    /// </summary>
    public static LoadedKernel LoadFromBytes(DeviceContext device, byte[] ptxBytes,
        KernelDescriptor descriptor);
}
```

### LoadedKernel

```csharp
public sealed class LoadedKernel : IDisposable
{
    public CUmodule Module { get; }
    public CudaKernel Kernel { get; }
    public KernelDescriptor Descriptor { get; }
    public void Dispose(); // unloads the CUmodule
}
```

### ModuleCache

```csharp
public sealed class ModuleCache : IDisposable
{
    public ModuleCache(DeviceContext device);
    public LoadedKernel GetOrLoad(string ptxPath);
    public bool Contains(string ptxPath);

    /// <summary>
    /// Evict a cached entry. Used by CudaEngine.CodeRebuild() for FilesystemPtx.
    /// </summary>
    public bool Evict(string ptxPath);
    public int Count { get; }
    public void Dispose();
}
```

### KernelDescriptor

```csharp
/// <summary>
/// Describes a CUDA kernel: entry point, parameters, and grid config hints.
/// Parsed from JSON metadata or constructed programmatically for ILGPU/NVRTC.
/// </summary>
public sealed class KernelDescriptor
{
    public required string EntryPoint { get; init; }
    public required IReadOnlyList<KernelParamDescriptor> Parameters { get; init; }

    /// <summary>Default block size hint (threads per block). 0 = not specified.</summary>
    public int BlockSize { get; init; }

    /// <summary>Shared memory size in bytes. 0 = not specified.</summary>
    public int SharedMemoryBytes { get; init; }
}
```

### KernelParamDescriptor

```csharp
/// <summary>
/// Direction of a kernel parameter (for documentation/validation).
/// </summary>
public enum ParamDirection { In, Out, InOut }

/// <summary>
/// Describes a single kernel parameter from JSON metadata or constructed programmatically.
/// </summary>
public sealed class KernelParamDescriptor
{
    public required string Name { get; init; }
    public required string Type { get; init; }
    public required int Index { get; init; }
    public ParamDirection Direction { get; init; } = ParamDirection.In;

    /// <summary>
    /// Whether this parameter is a pointer (buffer) vs a scalar value.
    /// </summary>
    public bool IsPointer { get; init; }

    /// <summary>
    /// Explicit size in bytes for this parameter slot. When 0, the size is
    /// computed automatically (IntPtr.Size for pointers, type-based for scalars).
    ///
    /// Set to 16 for ILGPU ArrayView struct params:
    /// ILGPU compiles ArrayView&lt;T&gt; as {void* ptr, long length}
    /// → `.param .align 8 .b8 name[16]` in PTX.
    /// </summary>
    public int SizeBytes { get; init; }
}
```

---

## PTX.Compilation (Phase 4b)

### IlgpuCompiler

```csharp
/// <summary>
/// Compiles C# methods to PTX via ILGPU's PTX backend.
/// Caches compiled kernels by method hash. No CUDA accelerator needed.
/// ~1-10ms per compilation.
///
/// ILGPU parameter layout for implicitly-grouped kernels:
///   param 0: _kernel_length (.b32 for Index1D) — implicit, total element count
///   param 1..N: user params where each ArrayView&lt;T&gt; is a 16-byte struct {ptr, length}
///
/// Example: ScaleKernel(Index1D, ArrayView&lt;float&gt;, ArrayView&lt;float&gt;)
///   → 3 PTX params: _kernel_length (.b32) + data (.b8[16]) + output (.b8[16])
/// </summary>
internal sealed class IlgpuCompiler : IDisposable  // VL.Cuda.Core.PTX.Compilation
{
    public IlgpuCompiler(DeviceContext device);

    /// <summary>
    /// Compile a C# method to PTX and load as a CUDA module. Cached by method hash.
    /// The method must follow ILGPU kernel conventions (Index1D first parameter, etc.).
    /// The descriptor is auto-expanded via ExpandDescriptorForIlgpu: each pointer
    /// (ArrayView) parameter becomes a 16-byte struct {ptr, length}.
    /// The returned LoadedKernel has the expanded descriptor.
    /// </summary>
    public LoadedKernel GetOrCompile(MethodInfo kernelMethod, KernelDescriptor descriptor);

    /// <summary>
    /// Compile a C# method to a PTX string without loading it.
    /// </summary>
    internal string CompileToString(MethodInfo kernelMethod);

    /// <summary>
    /// Remove a cached entry by method hash (SHA256).
    /// Called by CudaEngine.CodeRebuild() when IlgpuMethod source changes.
    /// </summary>
    public bool Invalidate(string methodHash);

    /// <summary>
    /// Number of cached compiled kernels.
    /// </summary>
    public int CacheCount { get; }

    /// <summary>
    /// Expand a user-provided KernelDescriptor to match ILGPU's actual PTX layout.
    /// Inserts _kernel_length at index 0 (SizeBytes=4, IsPointer=false).
    /// Sets SizeBytes=16 for all pointer (ArrayView) params.
    /// </summary>
    internal static KernelDescriptor ExpandDescriptorForIlgpu(
        KernelDescriptor original, string entryPointName);

    /// <summary>
    /// Compute index remap from original descriptor indices to expanded PTX indices.
    /// All user indices shift by +1 for the implicit _kernel_length at index 0.
    /// </summary>
    internal static int[] ComputeIndexRemap(KernelDescriptor original);

    /// <summary>
    /// Compute a SHA256 hash from a MethodInfo for cache keying.
    /// Input: "{DeclaringType.AssemblyQualifiedName}.{Name}.{MetadataToken}"
    /// </summary>
    internal static string ComputeMethodHash(MethodInfo method);

    public void Dispose();
}
```

### NvrtcCache

```csharp
/// <summary>
/// Compiles CUDA C++ source to PTX via NVRTC and loads as CUDA modules.
/// Caches compiled kernels by source hash. ~100ms-2s per compilation.
/// </summary>
internal sealed class NvrtcCache : IDisposable  // VL.Cuda.Core.PTX.Compilation
{
    public NvrtcCache(DeviceContext device);

    /// <summary>
    /// Compile CUDA C++ source to PTX and load as a CUDA module. Cached by source hash.
    /// Uses CudaRuntimeCompiler (ManagedCuda.NVRTC) with --gpu-architecture matching
    /// the device's compute capability.
    /// </summary>
    public LoadedKernel GetOrCompile(string cudaSource, string entryPoint,
        KernelDescriptor descriptor);

    /// <summary>
    /// Remove a cached entry by source hash.
    /// Called by CudaEngine.CodeRebuild() when NvrtcSource changes.
    /// </summary>
    public bool Invalidate(string sourceHash);

    /// <summary>
    /// Number of cached compiled kernels.
    /// </summary>
    public int CacheCount { get; }

    /// <summary>
    /// Compute a SHA256 hex string from CUDA source code for cache keying.
    /// </summary>
    internal static string ComputeSourceKey(string source);

    public void Dispose();
}
```

---

## Device

### DeviceContext

```csharp
public sealed class DeviceContext : IDisposable
{
    public DeviceContext(int deviceId = 0);

    public ManagedCuda.CudaContext Context { get; }
    public int DeviceId { get; }
    public string DeviceName { get; }
    public long TotalMemory { get; }
    public int ComputeCapabilityMajor { get; }
    public int ComputeCapabilityMinor { get; }

    public CUmodule LoadModule(string ptxPath);
    public CUfunction GetFunction(CUmodule module, string entryPoint);

    public void Dispose();
}
```

---

## Constructor Pattern (Filesystem PTX Block)

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

## Constructor Pattern (ILGPU Patchable Kernel Block)

Blocks using ILGPU-compiled C# methods. The `MethodInfo` references a static kernel with ILGPU conventions (Index1D first param). The user-facing `KernelDescriptor` describes params before expansion. `In()/Out()` indices are automatically remapped to ILGPU's expanded PTX layout.

```csharp
public class ScaleBlock : ICudaBlock, IDisposable
{
    private readonly CudaContext _ctx;

    public Guid Id { get; } = Guid.NewGuid();
    public string TypeName => "Scale";
    public NodeContext NodeContext { get; }
    // ... ICudaBlock members ...

    static void ScaleKernel(ILGPU.Index1D index,
        ILGPU.Runtime.ArrayView<float> data,
        ILGPU.Runtime.ArrayView<float> output)
    {
        output[index] = data[index] * 2.0f;
    }

    public ScaleBlock(NodeContext nodeContext, CudaContext ctx)
    {
        _ctx = ctx;
        NodeContext = nodeContext;

        var method = typeof(ScaleBlock).GetMethod(nameof(ScaleKernel),
            BindingFlags.Static | BindingFlags.NonPublic)!;

        var descriptor = new KernelDescriptor
        {
            EntryPoint = "ScaleKernel",
            Parameters = new[]
            {
                new KernelParamDescriptor { Name = "data", Type = "float*", Index = 0,
                    IsPointer = true, Direction = ParamDirection.In },
                new KernelParamDescriptor { Name = "output", Type = "float*", Index = 1,
                    IsPointer = true, Direction = ParamDirection.Out },
            },
        };

        var builder = new BlockBuilder(ctx, this);
        var kernel = builder.AddKernel(method, descriptor);
        kernel.GridDimX = 256;

        // In(0) → PTX param 1 (after implicit _kernel_length at 0)
        _inputs.Add(builder.Input<float>("Data", kernel.In(0)));
        _outputs.Add(builder.Output<float>("Output", kernel.Out(1)));

        builder.Commit();
        ctx.RegisterBlock(this);
    }

    public void Dispose() => _ctx.UnregisterBlock(Id);
}
```

---

## Constructor Pattern (NVRTC CUDA C++ Block)

Blocks using user-written CUDA C++ compiled at runtime via NVRTC.

```csharp
public class CustomKernelBlock : ICudaBlock, IDisposable
{
    private readonly CudaContext _ctx;
    // ... ICudaBlock members ...

    public CustomKernelBlock(NodeContext nodeContext, CudaContext ctx)
    {
        _ctx = ctx;
        NodeContext = nodeContext;

        const string cudaSource = @"
extern ""C"" __global__ void double_values(float* input, float* output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * 2.0f;
}";

        var descriptor = new KernelDescriptor
        {
            EntryPoint = "double_values",
            Parameters = new[]
            {
                new KernelParamDescriptor { Name = "input", Type = "float*", Index = 0,
                    IsPointer = true, Direction = ParamDirection.In },
                new KernelParamDescriptor { Name = "output", Type = "float*", Index = 1,
                    IsPointer = true, Direction = ParamDirection.Out },
                new KernelParamDescriptor { Name = "n", Type = "int", Index = 2,
                    IsPointer = false, Direction = ParamDirection.In },
            },
            BlockSize = 256,
        };

        var builder = new BlockBuilder(ctx, this);
        var kernel = builder.AddKernelFromCuda(cudaSource, "double_values", descriptor);
        kernel.GridDimX = 4;

        _inputs.Add(builder.Input<float>("Input", kernel.In(0)));
        _outputs.Add(builder.Output<float>("Output", kernel.Out(1)));

        builder.Commit();
        ctx.RegisterBlock(this);
    }

    public void Dispose() => _ctx.UnregisterBlock(Id);
}
```

---

## Constructor Pattern (Captured Block)

Blocks that use library calls via stream capture. The `CapturedHandle` is used the same way as `KernelHandle`.

```csharp
public class MatMulBlock : ICudaBlock, IDisposable
{
    private readonly CudaContext _ctx;
    // ... ICudaBlock members ...

    public MatMulBlock(NodeContext nodeContext, CudaContext ctx, int m, int n, int k)
    {
        _ctx = ctx;
        NodeContext = nodeContext;

        var builder = new BlockBuilder(ctx, this);
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

## Libraries (Phase 4a)

### LibraryHandleCache

```csharp
public sealed class LibraryHandleCache : IDisposable
{
    public CudaBlas GetOrCreateBlas();
    public CudaFFTPlan1D GetOrCreateFFT1D(int nx, cufftType type, int batch = 1);
    public CudaSparseContext GetOrCreateSparse();
    public CudaRandDevice GetOrCreateRand(GeneratorType type = GeneratorType.PseudoDefault);
    public CudaSolveDense GetOrCreateSolveDense();
    public void Dispose();
}
```

### BlasOperations

```csharp
public static class BlasOperations  // VL.Cuda.Core.Libraries.Blas
{
    public static CapturedHandle Sgemm(BlockBuilder builder, LibraryHandleCache libs,
        int m, int n, int k, float alpha = 1.0f, float beta = 0.0f,
        Operation transA = Operation.NonTranspose, Operation transB = Operation.NonTranspose);

    public static CapturedHandle Dgemm(BlockBuilder builder, LibraryHandleCache libs,
        int m, int n, int k, double alpha = 1.0, double beta = 0.0,
        Operation transA = Operation.NonTranspose, Operation transB = Operation.NonTranspose);

    public static CapturedHandle Sgemv(BlockBuilder builder, LibraryHandleCache libs,
        int m, int n, float alpha = 1.0f, float beta = 0.0f,
        Operation transA = Operation.NonTranspose);

    public static CapturedHandle Sscal(BlockBuilder builder, LibraryHandleCache libs,
        int n, float alpha = 1.0f);
}
```

### FftOperations

```csharp
public static class FftOperations  // VL.Cuda.Core.Libraries.FFT
{
    public static CapturedHandle Forward1D(BlockBuilder builder, LibraryHandleCache libs,
        int nx, cufftType type = cufftType.C2C, int batch = 1);
    public static CapturedHandle Inverse1D(BlockBuilder builder, LibraryHandleCache libs,
        int nx, cufftType type = cufftType.C2C, int batch = 1);
    public static CapturedHandle R2C1D(BlockBuilder builder, LibraryHandleCache libs,
        int nx, int batch = 1);
    public static CapturedHandle C2R1D(BlockBuilder builder, LibraryHandleCache libs,
        int nx, int batch = 1);
}
```

### SparseOperations

```csharp
public static class SparseOperations  // VL.Cuda.Core.Libraries.Sparse
{
    /// <summary>
    /// SpMV: y = alpha*A*x + beta*y. A is sparse CSR.
    /// In[0]=csrValues, In[1]=csrRowOffsets, In[2]=csrColInd, In[3]=x, Out[0]=y, Out[1]=workspace.
    /// </summary>
    public static CapturedHandle SpMV(BlockBuilder builder, LibraryHandleCache libs,
        int m, int n, int nnz, int workspaceSizeBytes);
}
```

### RandOperations

```csharp
public static class RandOperations  // VL.Cuda.Core.Libraries.Rand
{
    public static CapturedHandle GenerateUniform(BlockBuilder builder, LibraryHandleCache libs,
        int count);
    public static CapturedHandle GenerateNormal(BlockBuilder builder, LibraryHandleCache libs,
        int count, float mean = 0.0f, float stddev = 1.0f);
}
```

### SolveOperations

```csharp
public static class SolveOperations  // VL.Cuda.Core.Libraries.Solve
{
    public static CapturedHandle Sgetrf(BlockBuilder builder, LibraryHandleCache libs,
        int m, int n);
    public static CapturedHandle Sgetrs(BlockBuilder builder, LibraryHandleCache libs,
        int n, int nrhs, Operation trans = Operation.NonTranspose);
}
```

---

## Planned (Phase 3+ / Phase 5)

The following types are designed but not yet implemented:

- **InputHandle\<T\> / OutputHandle\<T\>** -- VL handle-flow for typed links (Phase 3)
- **GridConfig / GridSizeMode** -- Auto-grid from buffer size (Phase 3)
- **Regions** -- If/While/For conditional graph execution (Phase 3)
- **Composite blocks** -- AddChild / ConnectChildren / ExposeInput/Output (Phase 3)
- **ProfilingPipeline** -- Async GPU event readback (Phase 3+)
- **IDebugInfo** -- Full engine debug info (Phase 3+)
- **CudaDX11Interop** -- DX11/Stride graphics sharing (Phase 5)
- **ProfilingLevel** -- None/Summary/PerBlock/PerKernel/DeepAsync/DeepSync (Phase 3+)

See `PHASES.md` for the implementation roadmap.
