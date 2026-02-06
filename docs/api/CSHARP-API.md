# C# API Reference

## Module Structure

```
VL.Cuda.Core
  ├── Engine
  │     CudaEngine
  │     ProfilingPipeline
  │     FrameDebugData
  │     BlockDebugSnapshot
  │     
  ├── Context
  │     CudaContext
  │     CudaContextOptions
  │     CudaStream
  │     
  ├── Buffers
  │     GpuBuffer<T>
  │     AppendBuffer<T>
  │     BufferPool
  │     BufferShape
  │     DType
  │     BufferLifetime
  │     BufferState
  │     
  ├── Handles
  │     IHandle
  │     IInputHandle
  │     IOutputHandle
  │     InputHandle<T>
  │     OutputHandle<T>
  │     PinType
  │     PinTypeKind
  │     
  ├── Blocks
  │     ICudaBlock
  │     IBlockPort
  │     IBlockParameter
  │     BlockPort<T>
  │     BlockParameter<T>
  │     BlockBuilder
  │     
  ├── Graph
  │     GraphCompiler
  │     CompiledGraph
  │     GraphModel
  │     BlockModel
  │     ConnectionModel
  │     
  ├── Kernel
  │     KernelHandle
  │     KernelPin
  │     KernelDescriptor
  │     KernelParamDescriptor
  │     GridConfig
  │     GridSizeMode
  │     ParamDirection
  │     PTXLoader
  │     ModuleCache
  │     NvrtcCache
  │     
  ├── Captured
  │     CapturedHandle
  │     CapturedNodeDescriptor
  │     CapturedOpDescriptor
  │     StreamCaptureHelper
  │     LibraryHandleCache
  │     
  ├── NodeDescriptors
  │     INodeDescriptor
  │     KernelNodeDescriptor
  │     CapturedNodeDescriptor
  │     
  ├── Debug
  │     IDebugInfo
  │     IBlockDebugInfo
  │     RegionTiming
  │     KernelTiming
  │     KernelDebugInfo
  │     BufferInfo
  │     GraphStructure
  │     NodeInfo
  │     EdgeInfo
  │     RegionInfo
  │     DiagnosticMessage
  │     NodeKind
  │     RegionKind
  │     BlockState
  │     DiagnosticLevel
  │     ProfilingLevel
  │     
  └── Interop
        IDX11SharedResource
        SharedBuffer<T>
        SharedTexture2D<T>
        SharedTexture3D<T>
        CudaDX11Interop
```

---

## Engine

### CudaEngine

The single active ProcessNode. The ONLY component that does GPU work.

> See `EXECUTION-MODEL.md` for the full execution model.

```csharp
public class CudaEngine : IDisposable
{
    // === Creation ===
    
    /// <summary>
    /// VL calls: new CudaEngine(nodeContext, options)
    /// NodeContext provides: UniqueId (for VL diagnostics), ILogger, AppHost.
    /// </summary>
    public CudaEngine(NodeContext nodeContext, CudaContextOptions? options = null);
    
    // === Properties ===
    
    /// <summary>
    /// The CudaContext managed by this engine.
    /// Exposed as output pin in VL — flows to all blocks.
    /// </summary>
    public CudaContext Context { get; }
    
    /// <summary>
    /// Current profiling level. Controls debug data collection overhead.
    /// Can be changed at runtime via VL pin.
    /// </summary>
    public ProfilingLevel ProfilingLevel { get; set; }
    
    /// <summary>
    /// Debug info for the engine itself (total frame time, GPU memory, etc.)
    /// Displayed as tooltip on the CudaEngine node in VL.
    /// </summary>
    public IDebugInfo DebugInfo { get; }
    
    // === Execution (called every frame by VL) ===
    
    /// <summary>
    /// Main update method. Called once per frame.
    /// 1. Collects parameter values from all registered blocks
    /// 2. Checks dirty state (structure vs parameters)
    /// 3. Cold Rebuild or Hot/Warm Update as needed
    /// 4. Launches the CUDA Graph
    /// 5. Collects profiling data (async, based on ProfilingLevel)
    /// 6. Distributes debug info back to blocks
    /// 7. Reports diagnostics to VL (warnings/errors on nodes via IVLRuntime)
    /// 
    /// This is the ONLY place where GPU work happens.
    /// </summary>
    public void Update();
    
    /// <summary>
    /// Update with explicit stream.
    /// </summary>
    public void Update(CudaStream stream);
    
    // === State ===
    
    /// <summary>
    /// True if the last Update() performed a Cold Rebuild.
    /// Useful for VL to show rebuild indicator.
    /// </summary>
    public bool DidRebuildLastFrame { get; }
    
    /// <summary>
    /// Number of frames since last Cold Rebuild.
    /// </summary>
    public int FramesSinceLastRebuild { get; }
    
    // === ToString (VL Tooltip) ===
    
    /// <summary>
    /// VL calls ToString() on hover. Shows engine summary.
    /// Example: "CudaEngine: 4 blocks, 12 kernels\n  Launch: 2.1ms\n  Pool: 12 MB / 64 MB"
    /// </summary>
    public override string ToString();
    
    // === Dispose ===
    public void Dispose();
}
```

---

## Context

### CudaContext

Shared state between Engine and Blocks. Manages block registry, connections, dirty-tracking, and buffer pool.

```csharp
public class CudaContext : IDisposable
{
    // === Creation (internal — created by CudaEngine) ===
    internal static CudaContext Create(CudaContextOptions options);
    
    // === Properties ===
    public int DeviceId { get; }
    public CudaStream DefaultStream { get; }
    public IDebugInfo Debug { get; }
    public ProfilingLevel Profiling { get; set; }
    public BufferPool Pool { get; }
    
    // === Block Registry ===
    
    /// <summary>
    /// Register a block. Called by block constructors.
    /// Sets structureDirty = true.
    /// </summary>
    public void RegisterBlock(ICudaBlock block);
    
    /// <summary>
    /// Unregister a block. Called by block Dispose().
    /// Sets structureDirty = true.
    /// </summary>
    public void UnregisterBlock(Guid blockId);
    
    /// <summary>
    /// All currently registered blocks.
    /// </summary>
    public IReadOnlyDictionary<Guid, ICudaBlock> Blocks { get; }
    
    // === Connections (called by VL when user draws/removes links) ===
    
    /// <summary>
    /// Connect two block ports. Sets structureDirty = true.
    /// </summary>
    public void Connect(Guid sourceBlockId, string sourcePort,
                        Guid targetBlockId, string targetPort);
    
    /// <summary>
    /// Disconnect two block ports. Sets structureDirty = true.
    /// </summary>
    public void Disconnect(Guid sourceBlockId, string sourcePort,
                           Guid targetBlockId, string targetPort);
    
    /// <summary>
    /// All current connections.
    /// </summary>
    public IReadOnlyList<ConnectionModel> Connections { get; }
    
    // === Dirty Tracking ===
    
    /// <summary>
    /// True if graph structure changed (blocks added/removed, connections changed).
    /// Requires Cold Rebuild.
    /// </summary>
    public bool IsStructureDirty { get; }
    
    /// <summary>
    /// True if parameter values changed (scalars, buffer pointers).
    /// Requires Hot/Warm Update.
    /// </summary>
    public bool AreParametersDirty { get; }
    
    /// <summary>
    /// Called by CudaEngine after rebuild.
    /// </summary>
    internal void ClearStructureDirty();
    
    /// <summary>
    /// Called by CudaEngine after parameter update.
    /// </summary>
    internal void ClearParametersDirty();
    
    /// <summary>
    /// Called by blocks when a parameter value changes.
    /// Sets parametersDirty = true.
    /// </summary>
    internal void OnParameterChanged(Guid blockId, string paramName);
    
    // === Compilation (used by CudaEngine) ===
    internal CompiledGraph Compile();
    
    // === Serialization ===
    public GraphModel GetModel();
    public void LoadModel(GraphModel model);
    
    // === Dispose ===
    public void Dispose();
}
```

### CudaContextOptions

```csharp
public class CudaContextOptions
{
    public int DeviceId { get; set; } = 0;
    public ProfilingLevel Profiling { get; set; } = ProfilingLevel.None;
    public int BufferPoolInitialSizeMB { get; set; } = 256;
    public bool EnableDebug { get; set; } = true;
}
```

### CudaStream

```csharp
public class CudaStream : IDisposable
{
    public static CudaStream Create(CudaContext context);
    
    public IntPtr Handle { get; }
    public CudaContext Context { get; }
    
    public void Synchronize();
    public Task SynchronizeAsync();
    
    public void Dispose();
}
```

### ProfilingLevel

```csharp
public enum ProfilingLevel
{
    None,       // Zero overhead. Only state (OK/Warning/Error).
    Summary,    // 1 event pair around entire graph. Async, 1 frame latency.
    PerBlock,   // Event pair per block. Buffer sizes on link tooltips. Async.
    PerKernel,  // Event pair per kernel within blocks. Async.
    DeepAsync,  // + AppendBuffer count readbacks, occupancy. Async, 1-2 frame latency.
    DeepSync    // Same as DeepAsync but synchronous GPU readback. Exact values, causes GPU stall.
}
```

---

## Buffers

### GpuBuffer<T>

```csharp
public class GpuBuffer<T> : IDisposable where T : unmanaged
{
    // === Identity ===
    public Guid Id { get; }
    
    // === Memory ===
    public CUdeviceptr Pointer { get; }
    public long SizeInBytes { get; }
    public int ElementCount { get; }
    
    // === Type ===
    public DType ElementType { get; }
    public BufferShape Shape { get; }
    
    // === Lifecycle ===
    public BufferLifetime Lifetime { get; }
    public BufferState State { get; }
    
    // === Data Transfer ===
    public void Upload(ReadOnlySpan<T> data, CudaStream? stream = null);
    public void Download(Span<T> destination, CudaStream? stream = null);
    public Task<T[]> DownloadAsync(CudaStream? stream = null);
    
    // === Views ===
    public GpuBuffer<T> Slice(int offset, int count);
    public GpuBuffer<TNew> Reinterpret<TNew>() where TNew : unmanaged;
    
    // === Internal ===
    internal int RefCount { get; }
    internal void AddRef();
    internal void Release();
    
    // === Dispose ===
    public void Dispose();
}
```

### AppendBuffer<T>

```csharp
public class AppendBuffer<T> : GpuBuffer<T> where T : unmanaged
{
    public int MaxCapacity { get; }
    public CUdeviceptr CountPointer { get; }
    
    public int ReadCount(CudaStream? stream = null);
    public Task<int> ReadCountAsync(CudaStream? stream = null);
    public void ResetCount(CudaStream? stream = null);
}
```

### BufferShape

```csharp
public readonly struct BufferShape
{
    public int Rank { get; }
    public int Dim0 { get; }
    public int Dim1 { get; }
    public int Dim2 { get; }
    public int Stride0 { get; }
    public int Stride1 { get; }
    public int Stride2 { get; }
    public long TotalElements { get; }
    
    public BufferShape(int dim0);
    public BufferShape(int dim0, int dim1);
    public BufferShape(int dim0, int dim1, int dim2);
    public BufferShape(int[] dims, int[]? strides = null);
    
    public BufferShape WithStride(int[] strides);
    public BufferShape Transpose();
    public BufferShape Flatten();
}
```

### BufferPool

```csharp
public class BufferPool : IDisposable
{
    public BufferPool(CudaContext context, int initialSizeMB = 256);
    
    public GpuBuffer<T> Acquire<T>(int elementCount, BufferLifetime lifetime) 
        where T : unmanaged;
    public GpuBuffer<T> Acquire<T>(BufferShape shape, BufferLifetime lifetime)
        where T : unmanaged;
    public AppendBuffer<T> AcquireAppend<T>(int maxCapacity, BufferLifetime lifetime)
        where T : unmanaged;
    
    public void Release(GpuBuffer buffer);
    
    public long TotalAllocatedBytes { get; }
    public long CurrentlyUsedBytes { get; }
    public int BufferCount { get; }
    public IReadOnlyDictionary<int, int> BucketStats { get; }
    
    public void Dispose();
}
```

### Enums

```csharp
public enum DType
{
    U8, U16, U32, U64,
    S8, S16, S32, S64,
    F16, F32, F64,
    BF16,
    Complex64, Complex128
}

public enum BufferLifetime
{
    External,   // User-managed
    Graph,      // Lives with graph
    Region      // Temporary
}

public enum BufferState
{
    Valid,
    Uninitialized,
    Released
}
```

---

## Handles

### Interfaces

```csharp
public interface IHandle
{
    Guid SourceBlockId { get; }
    string PinName { get; }
    PinType Type { get; }
}

public interface IInputHandle : IHandle
{
    IOutputHandle? Source { get; set; }
}

public interface IOutputHandle : IHandle
{
}
```

### InputHandle<T>

```csharp
public class InputHandle<T> : IInputHandle
{
    public Guid SourceBlockId { get; }
    public string PinName { get; }
    public PinType Type { get; }
    
    public OutputHandle<T>? Source { get; set; }
    
    IOutputHandle? IInputHandle.Source
    {
        get => Source;
        set => Source = (OutputHandle<T>?)value;
    }
}
```

### OutputHandle<T>

```csharp
public class OutputHandle<T> : IOutputHandle
{
    public Guid SourceBlockId { get; }
    public string PinName { get; }
    public PinType Type { get; }
}
```

### PinType

```csharp
public class PinType
{
    public PinTypeKind Kind { get; }
    public DType ElementType { get; }
    public int? Channels { get; }
    
    public static PinType Buffer<T>() where T : unmanaged;
    public static PinType AppendBuffer<T>() where T : unmanaged;
    public static PinType Scalar<T>() where T : unmanaged;
    
    public bool IsCompatible(PinType other);
}

public enum PinTypeKind
{
    Buffer,
    AppendBuffer,
    Scalar
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
    
    IReadOnlyList<IBlockPort> Inputs { get; }
    IReadOnlyList<IBlockPort> Outputs { get; }
    IReadOnlyList<IBlockParameter> Parameters { get; }
    
    /// <summary>
    /// Debug info written by CudaEngine after each frame.
    /// </summary>
    IBlockDebugInfo DebugInfo { get; set; }
    
    /// <summary>
    /// VL NodeContext for this block instance.
    /// Used by CudaEngine to report warnings/errors on the correct VL node.
    /// </summary>
    NodeContext NodeContext { get; }
}
```

#### Constructor Pattern

All blocks follow this constructor pattern. VL provides `NodeContext` as the first parameter:

```csharp
public class MyBlock : ICudaBlock
{
    public MyBlock(NodeContext nodeContext, CudaContext cudaContext)
    {
        NodeContext = nodeContext;
        _logger = nodeContext.GetLogger();
        
        // Declarative setup (no GPU work)
        var builder = new BlockBuilder(cudaContext, this);
        // ... define kernels, pins ...
        builder.Commit();
        
        cudaContext.RegisterBlock(this, nodeContext);
    }
    
    public NodeContext NodeContext { get; }
    
    /// <summary>
    /// VL calls ToString() on hover. Reads from profiling cache only — no GPU access.
    /// </summary>
    public override string ToString()
    {
        var info = _cudaContext.Profiling.GetCached(this.Id);
        if (info == null) return $"{TypeName}: {DebugInfo?.State ?? BlockState.OK}";
        return $"{TypeName}: {info.State}, {info.Timing?.TotalMilliseconds:F2}ms";
    }
    
    public void Dispose() => _cudaContext.UnregisterBlock(this);
}
```

### IBlockPort

```csharp
public interface IBlockPort
{
    string Name { get; }
    PinType Type { get; }
    string? Description { get; }
}
```

### IBlockParameter

```csharp
public interface IBlockParameter
{
    string Name { get; }
    Type ValueType { get; }
    object Value { get; set; }
    object DefaultValue { get; }
    string? Description { get; }
}
```

### BlockPort<T>

```csharp
public class BlockPort<T> : IBlockPort
{
    public string Name { get; }
    public PinType Type { get; }
    public string? Description { get; }
    
    internal Guid NodeId { get; }
    internal int PinIndex { get; }
}
```

### BlockParameter<T>

```csharp
public class BlockParameter<T> : IBlockParameter
{
    public string Name { get; }
    public Type ValueType => typeof(T);
    public T Value { get; set; }
    public T DefaultValue { get; }
    public string? Description { get; }
    
    object IBlockParameter.Value
    {
        get => Value!;
        set => Value = (T)value;
    }
    object IBlockParameter.DefaultValue => DefaultValue!;
}
```

### BlockBuilder

```csharp
public class BlockBuilder
{
    // === Context ===
    public CudaContext Context { get; }
    public Guid BlockId { get; }
    
    // === Construction (takes block reference for registration) ===
    public BlockBuilder(CudaContext context, ICudaBlock block);
    
    // === Kernels (Source 1: Filesystem PTX) ===
    public KernelHandle AddKernel(string ptxPath, string entryPoint,
                                   string? debugName = null);
    public KernelHandle AddKernel(string ptxPath, string entryPoint,
                                   GridConfig grid, string? debugName = null);
    
    // === Kernels (Source 2: NVRTC Patchable Kernel) ===
    public KernelHandle AddKernel(CUmodule nvrtcModule, string entryPoint,
                                   string? debugName = null);
    
    // === Captured Operations (Source 3: Library Calls) ===
    public CapturedHandle AddCaptured(string name, Action<CUstream> captureAction,
                                       CapturedOpDescriptor descriptor);
    
    // === Buffer Inputs ===
    public InputHandle<GpuBuffer<T>> Input<T>(string name, KernelPin pin,
                                               string? description = null)
        where T : unmanaged;
    public InputHandle<AppendBuffer<T>> InputAppend<T>(string name, KernelPin pin,
                                                        string? description = null)
        where T : unmanaged;
    
    // === Scalar Inputs ===
    public void InputScalar<T>(string name, KernelPin pin,
                                T defaultValue = default,
                                string? description = null)
        where T : unmanaged;
    
    // === Outputs ===
    public OutputHandle<GpuBuffer<T>> Output<T>(string name, KernelPin pin,
                                                 string? description = null)
        where T : unmanaged;
    public OutputHandle<AppendBuffer<T>> OutputAppend<T>(string name, KernelPin pin,
                                                          string? description = null)
        where T : unmanaged;
    
    // === Internal Connections ===
    public void Connect<T>(OutputHandle<T> source, InputHandle<T> target);
    
    // === Child Blocks ===
    public T AddChild<T>() where T : ICudaBlock;
    public ICudaBlock AddChild(Type blockType);
    
    // === Child Connections ===
    public void ConnectChildren<T>(OutputHandle<T> source, InputHandle<T> target);
    public void ConnectChildren(ICudaBlock source, string sourcePort,
                                 ICudaBlock target, string targetPort);
    
    // === Expose Ports ===
    public InputHandle<T> ExposeInput<T>(string name, InputHandle<T> childInput);
    public InputHandle<T> ExposeInput<T>(InputHandle<T> childInput);
    
    public OutputHandle<T> ExposeOutput<T>(string name, OutputHandle<T> childOutput);
    public OutputHandle<T> ExposeOutput<T>(OutputHandle<T> childOutput);
    
    // === Expose Parameters ===
    public BlockParameter<T> ExposeParameter<T>(string name, 
                                                 ICudaBlock child, 
                                                 string childParamName);
    
    // === Finalize ===
    public void Commit();
}
```

---

## Kernel

### KernelHandle

```csharp
public class KernelHandle
{
    public Guid Id { get; }
    public string PTXPath { get; }
    public string EntryPoint { get; }
    public string DebugName { get; }
    public GridConfig Grid { get; set; }
    
    public KernelPin In(int index);
    public KernelPin Out(int index);
    
    public KernelDescriptor Descriptor { get; }
}
```

### KernelPin

```csharp
public readonly struct KernelPin
{
    public Guid KernelId { get; }
    public int Index { get; }
    public bool IsOutput { get; }
    public KernelParamDescriptor Descriptor { get; }
}
```

### GridConfig

```csharp
public class GridConfig
{
    public int[] BlockDim { get; set; } = new[] { 256 };
    public GridSizeMode Mode { get; set; } = GridSizeMode.Auto;
    public int[]? FixedGridDim { get; set; }
    public int? AutoSizeFromParam { get; set; }
}

public enum GridSizeMode
{
    Fixed,
    Auto
}
```

### KernelDescriptor

```csharp
public class KernelDescriptor
{
    public string PTXPath { get; }
    public string EntryPoint { get; }
    public string? Name { get; }
    public string? Version { get; }
    public string? Category { get; }
    public string? Summary { get; }
    public string[]? Tags { get; }
    
    public IReadOnlyList<KernelParamDescriptor> Parameters { get; }
    
    public int[]? MaxThreads { get; }
    public int[]? ReqThreads { get; }
    public int SharedMemoryStatic { get; }
    public bool UsesDynamicShared { get; }
    
    public bool Generated { get; }
    public bool NeedsReview { get; }
    public string? PtxVersion { get; }
    public string? TargetSM { get; }
}
```

### KernelParamDescriptor

```csharp
public class KernelParamDescriptor
{
    public int Index { get; }
    public string PtxType { get; }
    public bool IsPointer { get; }
    public int? Alignment { get; }
    public string? ElementType { get; }
    public ParamDirection Direction { get; }
    public string? Name { get; }
    public string? Semantic { get; }
    public string? ShapeRule { get; }
}

public enum ParamDirection
{
    In,
    Out,
    InOut,
    Unknown
}
```

---

## Graph

### CompiledGraph

```csharp
public class CompiledGraph : IDisposable
{
    public Guid Id { get; }
    public bool IsValid { get; }
    
    public void Launch(CudaStream? stream = null);
    
    /// <summary>
    /// Hot Update: change scalar value without rebuild.
    /// </summary>
    public void SetScalar<T>(Guid nodeId, int paramIndex, T value) where T : unmanaged;
    
    /// <summary>
    /// Warm Update: change buffer pointer without rebuild.
    /// </summary>
    public void UpdatePointer(Guid nodeId, int paramIndex, CUdeviceptr newPointer);
    
    /// <summary>
    /// Recapture Update: re-capture a CapturedNode's library call and update the child graph.
    /// Used when CapturedNode parameters change.
    /// </summary>
    public void UpdateChildGraph(CapturedNodeDescriptor node, CUgraph newChildGraph);
    
    /// <summary>
    /// Dispatches parameter updates by node type (Hot/Warm for KernelNodes, Recapture for CapturedNodes).
    /// </summary>
    public void UpdateKernelParams(KernelNodeDescriptor node);
    
    public IReadOnlyList<Guid> KernelNodes { get; }
    public IReadOnlyList<Guid> CapturedNodes { get; }
    public IReadOnlyDictionary<Guid, GpuBuffer> AllocatedBuffers { get; }
    
    public void Dispose();
}
```

### GraphModel

```csharp
public class GraphModel
{
    public string Version { get; set; } = "1.0";
    public List<BlockModel> Blocks { get; set; } = new();
    public List<ConnectionModel> Connections { get; set; } = new();
    
    public string ToJson();
    public static GraphModel FromJson(string json);
    
    public void SaveToFile(string path);
    public static GraphModel LoadFromFile(string path);
}

public class BlockModel
{
    public Guid Id { get; set; }
    public string TypeName { get; set; } = "";
    public Dictionary<string, object> Parameters { get; set; } = new();
}

public class ConnectionModel
{
    public Guid SourceBlockId { get; set; }
    public string SourcePort { get; set; } = "";
    public Guid TargetBlockId { get; set; }
    public string TargetPort { get; set; } = "";
}
```

---

## Debug

### IDebugInfo

```csharp
public interface IDebugInfo
{
    IReadOnlyDictionary<Guid, TimeSpan> LastFrameTiming { get; }
    RegionTiming GetHierarchicalTiming();
    
    IReadOnlyList<BufferInfo> AllocatedBuffers { get; }
    Task<T[]> ReadbackAsync<T>(GpuBuffer<T> buffer) where T : unmanaged;
    Task<T[]> ReadbackRangeAsync<T>(GpuBuffer<T> buffer, int offset, int count)
        where T : unmanaged;
    int ReadAppendCount<T>(AppendBuffer<T> buffer) where T : unmanaged;
    
    GraphStructure GetGraphStructure();
    
    IReadOnlyList<DiagnosticMessage> Messages { get; }
    event Action<DiagnosticMessage>? OnMessage;
    void ClearMessages();
    
    IReadOnlyDictionary<Guid, object> CurrentScalars { get; }
    IReadOnlyDictionary<Guid, bool> ConditionalStates { get; }
}
```

### IBlockDebugInfo

```csharp
public interface IBlockDebugInfo
{
    TimeSpan LastExecutionTime { get; }
    TimeSpan AverageExecutionTime { get; }
    
    IReadOnlyList<BufferInfo> Buffers { get; }
    IReadOnlyList<KernelDebugInfo> Kernels { get; }
    
    BlockState State { get; }
    string? StateMessage { get; }
    
    IReadOnlyList<IBlockDebugInfo> Children { get; }
}

public enum BlockState
{
    OK,
    Warning,
    Error,
    NotCompiled
}
```

### Timing Classes

```csharp
public class RegionTiming
{
    public Guid BlockId { get; }
    public string Name { get; }
    public TimeSpan TotalTime { get; }
    public TimeSpan SelfTime { get; }
    
    public IReadOnlyList<RegionTiming> Children { get; }
    public IReadOnlyList<KernelTiming> Kernels { get; }
}

public class KernelTiming
{
    public Guid KernelId { get; }
    public string Name { get; }
    public TimeSpan Time { get; }
}

public class KernelDebugInfo
{
    public string Name { get; }
    public string PTXPath { get; }
    public string EntryPoint { get; }
    
    public TimeSpan LastExecutionTime { get; }
    public TimeSpan MinTime { get; }
    public TimeSpan MaxTime { get; }
    public TimeSpan AvgTime { get; }
    
    public int[] GridDim { get; }
    public int[] BlockDim { get; }
    public int TotalThreads { get; }
    
    public float TheoreticalOccupancy { get; }
    public float AchievedOccupancy { get; }
    
    public int SharedMemoryBytes { get; }
    public int RegistersPerThread { get; }
}
```

### Structure Classes

```csharp
public class BufferInfo
{
    public Guid Id { get; }
    public string? Name { get; }
    public DType ElementType { get; }
    public long SizeInBytes { get; }
    public int[] Shape { get; }
    public BufferLifetime Lifetime { get; }
    public BufferState State { get; }
    
    public bool IsAppendBuffer { get; }
    public int? CurrentCount { get; }
    public int? MaxCapacity { get; }
    
    public int PoolBucket { get; }
    public bool IsReused { get; }
}

public class GraphStructure
{
    public IReadOnlyList<NodeInfo> Nodes { get; }
    public IReadOnlyList<EdgeInfo> Edges { get; }
    public IReadOnlyList<RegionInfo> Regions { get; }
}

public class NodeInfo
{
    public Guid Id { get; }
    public string Name { get; }
    public NodeKind Kind { get; }
    public Guid? ParentRegionId { get; }
    public int TopoLevel { get; }
}

public class EdgeInfo
{
    public Guid SourceNodeId { get; }
    public int SourcePinIndex { get; }
    public Guid TargetNodeId { get; }
    public int TargetPinIndex { get; }
    public string ElementType { get; }
    public bool IsBufferReused { get; }
}

public class RegionInfo
{
    public Guid Id { get; }
    public string Name { get; }
    public RegionKind Kind { get; }
    public Guid? ParentId { get; }
    public IReadOnlyList<Guid> ChildNodes { get; }
}
```

### Enums

```csharp
public enum NodeKind
{
    Kernel,          // KernelNode (filesystem PTX or NVRTC patchable)
    CapturedOp,      // CapturedNode (library operation via Stream Capture)
    BufferCreate,
    ResetCount,
    Memset,
    Cast
}

public enum RegionKind
{
    Root,
    If,
    IfElse,
    While,
    For,
    Delegate
}

public class DiagnosticMessage
{
    public DiagnosticLevel Level { get; }
    public Guid? BlockId { get; }
    public Guid? KernelId { get; }
    public string Message { get; }
    public string? Details { get; }
    public DateTime Timestamp { get; }
}

public enum DiagnosticLevel
{
    Info,
    Warning,
    Error
}
```

---

## Interop

### IDX11SharedResource

```csharp
public interface IDX11SharedResource : IDisposable
{
    IntPtr DX11Resource { get; }
    CUdeviceptr CudaPointer { get; }
    
    void MapForCuda(CudaStream? stream = null);
    void UnmapFromCuda(CudaStream? stream = null);
    
    bool IsMappedForCuda { get; }
}
```

### CudaDX11Interop

```csharp
public static class CudaDX11Interop
{
    public static SharedBuffer<T> RegisterBuffer<T>(
        IntPtr dx11Buffer, CudaContext ctx,
        CUgraphicsRegisterFlags flags = CUgraphicsRegisterFlags.None)
        where T : unmanaged;
    
    public static SharedTexture2D<T> RegisterTexture2D<T>(
        IntPtr dx11Texture, CudaContext ctx,
        CUgraphicsRegisterFlags flags = CUgraphicsRegisterFlags.None)
        where T : unmanaged;
    
    public static SharedTexture3D<T> RegisterTexture3D<T>(
        IntPtr dx11Texture, CudaContext ctx,
        CUgraphicsRegisterFlags flags = CUgraphicsRegisterFlags.None)
        where T : unmanaged;
    
    public static SharedBuffer<T> CreateSharedBuffer<T>(
        int elementCount, CudaContext ctx, ID3D11Device dx11Device)
        where T : unmanaged;
}
```
