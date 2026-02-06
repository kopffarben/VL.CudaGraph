# Execution Model

## Overview

The execution model defines how CUDA compute graphs are built, updated, and executed within the VL runtime. The core principle is **one central engine, passive blocks**:

```
┌──────────────────────────────────────────────────────────────────┐
│ VL Patch                                                          │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                     │
│  │ Emitter  │──▶│ Forces   │──▶│Integrate │   ← Passive Blocks  │
│  │ (Block)  │   │ (Block)  │   │ (Block)  │     (data containers│
│  └──────────┘   └──────────┘   └──────────┘      + debug info)  │
│                                                                   │
│  ┌────────────────────────────────────────────┐                  │
│  │ CudaEngine (ProcessNode)                    │  ← Active        │
│  │  → owns CudaContext                         │    (compiles,    │
│  │  → compiles CUDA Graph                      │     executes,    │
│  │  → executes every frame                     │     reports)     │
│  │  → distributes debug info                   │                  │
│  └────────────────────────────────────────────┘                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Responsibilities

| Component | Type | Responsibility |
|-----------|------|----------------|
| **Block** | ProcessNode (passive) | Graph description, parameter storage, debug display via `ToString()` |
| **CudaEngine** | ProcessNode (active) | Graph compile, execute, dirty-tracking, debug collection, VL diagnostics |
| **CudaContext** | Internal object | Graph state, BufferPool, module cache, block registry |

---

## VL Lifecycle Integration

### NodeContext

Every VL-created instance receives a `NodeContext` as its first constructor parameter. This provides:
- **Identity**: `UniqueId` via `nodeContext.Path.Stack.Peek()` for addressing VL warnings/errors to the correct node
- **Logging**: `nodeContext.GetLogger()` returns a standard `ILogger`
- **App access**: `nodeContext.AppHost` for accessing services like `IVLRuntime`

### Block Lifecycle

```
VL creates Block:
    new EmitterBlock(NodeContext nodeContext, CudaContext cudaContext)
        │
        ├── nodeContext stored (for identity, logging)
        ├── Setup() called (defines kernels, pins, connections)
        ├── cudaContext.RegisterBlock(this, nodeContext)
        └── _structureDirty = true on CudaContext

VL calls every frame:
    block.Update()
        │
        └── Only updates debug display (ToString)
            No GPU work happens here

VL Hot-Swap (code change):
    block.Dispose()
        │
        ├── cudaContext.UnregisterBlock(this)
        └── _structureDirty = true on CudaContext
    
    new EmitterBlock(nodeContext, cudaContext)   ← new instance, new code
    VL restores connections via Connect() calls
```

### CudaEngine Lifecycle

```
VL creates Engine:
    new CudaEngine(NodeContext nodeContext, CudaEngineOptions options)
        │
        ├── _runtime = IVLRuntime.Current
        ├── _logger = nodeContext.GetLogger()
        └── _cudaContext = new CudaContext(options)

VL calls every frame:
    cudaEngine.Update()
        │
        ├── 1. Collect parameters from all blocks
        ├── 2. Dirty-check (structure / parameters)
        ├── 3. Cold Rebuild or Warm Update
        ├── 4. Graph Launch
        ├── 5. Collect profiling data (async)
        ├── 6. Update debug cache
        └── 7. Report diagnostics to VL (warnings/errors on nodes)
```

### Constructor Signatures

```csharp
public class CudaEngine
{
    public CudaEngine(NodeContext nodeContext, CudaEngineOptions? options = null)
    {
        _nodeContext = nodeContext;
        _runtime = IVLRuntime.Current;
        _logger = nodeContext.GetLogger();
        _cudaContext = new CudaContext(options ?? new CudaEngineOptions());
    }
}

public class EmitterBlock : ICudaBlock
{
    public EmitterBlock(NodeContext nodeContext, CudaContext cudaContext)
    {
        _nodeContext = nodeContext;
        _logger = nodeContext.GetLogger();
        
        Setup(new BlockBuilder(cudaContext, this));
        cudaContext.RegisterBlock(this, nodeContext);
    }
    
    public void Dispose()
    {
        _cudaContext.UnregisterBlock(this);
    }
}
```

---

## Dirty-Tracking

### Three Update Tiers

| Tier | Cost | Trigger | Action |
|------|------|---------|--------|
| **Hot Update** | Near-zero, no GPU stall | Scalar value changed (gravity, deltaTime) | `cuGraphExecKernelNodeSetParams` for scalar only |
| **Warm Update** | Cheap, exec-level update | Buffer rebind (different pointer, same size), grid size change | `cuGraphExecKernelNodeSetParams` for pointer/grid |
| **Cold Rebuild** | Expensive, full graph rebuild | Node added/removed, edge added/removed, PTX hot-reload, block structure changed | Destroy old graph, build new, instantiate |

### Design Rationale

VL compiles .vl files on-the-fly to C# and supports Hot-Swap: when the user edits a block's code, VL disposes the old instance and creates a new one with the updated code. VL then restores connections by calling `Connect()` again.

Structural changes (Cold Rebuild) happen during development only. During runtime of an exported application, only Hot/Warm updates occur. The graph rebuild may take several milliseconds, which is acceptable during interactive development.

### Dirty Flags

```csharp
class CudaContext
{
    private bool _structureDirty = true;    // Cold Rebuild needed
    private bool _parametersDirty = false;  // Warm/Hot Update needed
    
    // --- These set _structureDirty = true ---
    
    public void RegisterBlock(ICudaBlock block, NodeContext nodeContext);
    public void UnregisterBlock(ICudaBlock block);
    public void Connect(Guid sourceBlockId, string sourcePort,
                        Guid targetBlockId, string targetPort);
    public void Disconnect(Guid sourceBlockId, string sourcePort,
                           Guid targetBlockId, string targetPort);
    internal void OnBlockStructureChanged(Guid blockId);
    
    // --- These set _parametersDirty = true ---
    
    public void SetParameter<T>(Guid blockId, string name, T value);
    public void RebindBuffer(Guid blockId, string pinName, CUdeviceptr newPointer);
}
```

### Block Structure Change Detection

When VL hot-swaps a block, the new instance calls `Setup()` in its constructor. The `BlockBuilder` produces a `BlockDescription` (list of kernels, pins, internal connections). This is compared against the previous description for the same block ID:

```csharp
class BlockBuilder
{
    public void Commit()
    {
        var newDescription = BuildDescription();
        var oldDescription = _context.GetBlockDescription(_blockId);
        
        if (!newDescription.StructuralEquals(oldDescription))
        {
            _context.OnBlockStructureChanged(_blockId);
        }
        
        _context.SetBlockDescription(_blockId, newDescription);
    }
}
```

`BlockDescription` structural equality checks:
- Same kernel set (ptxPath + entryPoint, in order)
- Same pin set (name + type + direction)
- Same internal connections

### Execution Flow Per Frame

```csharp
class CudaEngine
{
    public void Update()
    {
        // 1. Collect current parameter values from all blocks
        CollectParameters();
        
        // 2. Rebuild or update
        if (_cudaContext.StructureDirty)
        {
            ColdRebuild();
            // Rebuild includes all current parameters,
            // so parameter dirty flag is also cleared
            _cudaContext.ClearAllDirty();
        }
        else if (_cudaContext.ParametersDirty)
        {
            WarmUpdate();
            _cudaContext.ClearParameterDirty();
        }
        
        // 3. Launch
        _compiledGraph.Launch(_cudaContext.DefaultStream);
        
        // 4. Profiling (async collection)
        _profilingPipeline.OnPostLaunch();
        
        // 5. Report diagnostics to VL
        ReportDiagnostics();
    }
}
```

---

## Profiling

### Profiling Levels

```csharp
public enum ProfilingLevel
{
    None,        // Zero overhead. Only state (OK/Warning/Error).
    Summary,     // 1 event pair around entire graph. Async readback.
    PerBlock,    // Event pair per block. Buffer sizes on links. Async.
    PerKernel,   // Event pair per kernel. Async.
    DeepAsync,   // + AppendBuffer count readbacks, occupancy. Async, 1-2 frame latency.
    DeepSync     // Same as DeepAsync but synchronous GPU readback. Exact values, causes GPU stall.
}
```

### What Each Level Provides

| Level | GPU Events | Readback | Cost | Display | Latency |
|-------|-----------|----------|------|---------|---------|
| None | None | None | Zero | State only | — |
| Summary | 1 pair (whole graph) | Async event query | Minimal | Total launch time on Engine | 1 frame |
| PerBlock | 2 × N blocks | Async event query | Medium | Timing per block, buffer sizes on links | 1 frame |
| PerKernel | 2 × M kernels | Async event query | Higher | Timing per kernel within blocks | 1 frame |
| DeepAsync | 2 × M kernels | Async events + async buffer readbacks | High | Everything: timing, counts, occupancy | 1-2 frames |
| DeepSync | 2 × M kernels | Synchronous (GPU stall!) | Very high | Everything, exact current-frame values | 0 (blocks CPU) |

### Async Readback Pipeline

Profiling data is collected asynchronously to avoid GPU stalls. A ring-buffer of 3 in-flight frames ensures results arrive without blocking:

```
Frame 0:              Frame 1:              Frame 2:

Launch()              Launch()              Launch()
Record Events(F0)     Record Events(F1)     Record Events(F2)
Request Readbacks(F0) Request Readbacks(F1) Request Readbacks(F2)
                      │                     │
                      Read Results(F0)      Read Results(F1)
                      Cache Update(F0)      Cache Update(F1)
                                            │
                                            ▼
                                         ToString() shows F1 data
```

```csharp
class ProfilingPipeline
{
    private const int InFlightFrames = 3;
    private FrameDebugData[] _inFlight = new FrameDebugData[InFlightFrames];
    private int _writeIndex;
    private FrameDebugData? _current;  // Latest completed data
    
    public void OnPreLaunch(CompiledGraph graph, ProfilingLevel level)
    {
        // Check if oldest in-flight frame is ready
        var oldest = _inFlight[_writeIndex % InFlightFrames];
        if (oldest != null && oldest.IsReady())
            _current = oldest;
        
        // Prepare new frame
        var frame = new FrameDebugData();
        
        if (level >= ProfilingLevel.Summary)
            frame.RecordGraphEvents(graph);
        if (level >= ProfilingLevel.PerBlock)
            frame.RecordBlockEvents(graph);
        if (level >= ProfilingLevel.PerKernel)
            frame.RecordKernelEvents(graph);
        if (level >= ProfilingLevel.DeepAsync)
            frame.RequestAsyncReadbacks(graph);
        if (level == ProfilingLevel.DeepSync)
            frame.RequestSyncReadbacks(graph);  // GPU stall here
        
        _inFlight[_writeIndex++ % InFlightFrames] = frame;
    }
    
    /// <summary>
    /// Called by blocks via ToString(). Returns null if no data available yet.
    /// </summary>
    public BlockDebugSnapshot? GetCached(Guid blockId)
    {
        return _current?.GetBlock(blockId);
    }
}
```

---

## Debug Display

### Principle: Zero Overhead When Not Observed

VL calls `ToString()` only when the user hovers a node or pin. No profiling data is collected in `ToString()` — it reads from the async profiling cache only. The profiling cache is populated by the `ProfilingPipeline` based on the current `ProfilingLevel` set on the `CudaEngine`.

### Node Tooltips (Block.ToString)

```csharp
class EmitterBlock : ICudaBlock
{
    public override string ToString()
    {
        var info = _cudaContext.Profiling.GetCached(this.Id);
        
        if (info == null)
            return $"Emitter: {State}";
        
        var sb = new StringBuilder();
        sb.Append($"Emitter: {State}");
        
        // PerBlock and above
        if (info.Timing.HasValue)
            sb.Append($", {info.Timing.Value.TotalMilliseconds:F2}ms");
        
        // DeepAsync/DeepSync
        if (info.BufferStats != null)
        {
            foreach (var buf in info.BufferStats)
                sb.Append($"\n  {buf.Name}: {buf.Count}/{buf.MaxCapacity}");
        }
        
        return sb.ToString();
    }
}
```

Per profiling level:
```
None:       "Emitter: OK"
PerBlock:   "Emitter: OK, 0.42ms"
DeepAsync:  "Emitter: OK, 0.42ms
              Particles: 45231/100000"
```

### Link Tooltips (OutputHandle.ToString)

Links in VL carry `OutputHandle<GpuBuffer<T>>`. When the user hovers a link or attaches an IOBox, `ToString()` shows buffer information:

```csharp
public class OutputHandle<T> : IOutputHandle
{
    public override string ToString()
    {
        if (_buffer == null)
            return $"OutputHandle<{typeof(T).Name}> (unbound)";
        
        var sb = new StringBuilder();
        sb.Append($"GpuBuffer<{typeof(T).Name}>");
        sb.Append($"\n  {_buffer.ElementCount} elements");
        sb.Append($"\n  {FormatBytes(_buffer.SizeInBytes)}");
        
        // PerBlock+: show pool allocation overhead
        if (_profilingLevel >= ProfilingLevel.PerBlock)
            sb.Append($" ({FormatBytes(_buffer.PoolBucketSize)} allocated)");
        
        // DeepAsync/DeepSync: AppendBuffer live count
        if (_buffer is AppendBuffer append)
        {
            var cached = _profiling?.GetCachedAppendCount(append.Id);
            if (cached.HasValue)
                sb.Append($"\n  Count: {cached.Value}/{append.MaxCapacity}");
        }
        
        return sb.ToString();
    }
}
```

Per profiling level:
```
None:       "GpuBuffer<Particle>
              45231 elements
              708 KB"

PerBlock:   "GpuBuffer<Particle>
              45231 elements
              708 KB (1024 KB allocated)"

DeepAsync:  "GpuBuffer<Particle>
              45231 elements
              708 KB (1024 KB allocated)
              Count: 45231/100000"
```

### Engine Tooltip (CudaEngine.ToString)

```csharp
class CudaEngine
{
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append($"CudaEngine: {_blocks.Count} blocks");
        
        if (_compiledGraph != null)
            sb.Append($", {_compiledGraph.KernelNodes.Count} kernels");
        
        var timing = _profilingPipeline.GetGraphTiming();
        if (timing.HasValue)
            sb.Append($"\n  Launch: {timing.Value.TotalMilliseconds:F2}ms");
        
        sb.Append($"\n  Profiling: {_profilingLevel}");
        sb.Append($"\n  Pool: {FormatBytes(_cudaContext.Pool.CurrentlyUsedBytes)}" +
                  $" / {FormatBytes(_cudaContext.Pool.TotalAllocatedBytes)}");
        
        return sb.ToString();
    }
}
```

```
"CudaEngine: 4 blocks, 12 kernels
  Launch: 2.1ms
  Profiling: PerBlock
  Pool: 12 MB / 64 MB"
```

---

## VL Diagnostics Integration

### Warnings and Errors on Nodes

VL displays warnings (node turns orange) and errors (node turns red) directly on nodes. The CudaEngine uses `IVLRuntime` to report diagnostics after each frame:

```csharp
class CudaEngine
{
    private IVLRuntime _runtime;
    private Dictionary<Guid, IDisposable?> _persistentMessages = new();
    
    private void ReportDiagnostics()
    {
        foreach (var reg in _blockRegistry.Values)
        {
            var diag = GetDiagnostics(reg.Block.Id);
            var elementId = reg.NodeContext.Path.Stack.Peek();
            
            if (diag.HasError)
            {
                // Persistent: stays until error is resolved
                if (!_persistentMessages.ContainsKey(reg.Block.Id) 
                    || _persistentMessages[reg.Block.Id] == null)
                {
                    _persistentMessages[reg.Block.Id] = 
                        _runtime.AddPersistentMessage(
                            new Message(elementId, MessageSeverity.Error, diag.ErrorText));
                }
            }
            else
            {
                // Clear persistent error if resolved
                if (_persistentMessages.TryGetValue(reg.Block.Id, out var msg) && msg != null)
                {
                    msg.Dispose();
                    _persistentMessages[reg.Block.Id] = null;
                }
                
                if (diag.HasWarning)
                {
                    // Per-frame: disappears when warning condition clears
                    _runtime.AddMessage(elementId, diag.WarningText, MessageSeverity.Warning);
                }
            }
        }
    }
}
```

### Diagnostic Sources

| Source | Severity | Example |
|--------|----------|---------|
| PTX load failure | Error | "Failed to load kernel.ptx: file not found" |
| CUDA launch error | Error | "CUDA_ERROR_LAUNCH_FAILED in particle_emit" |
| Graph validation | Error | "Type mismatch: GpuBuffer\<float4\> → GpuBuffer\<float3\>" |
| Missing connection | Warning | "Required input 'Particles' not connected" |
| Buffer near capacity | Warning | "AppendBuffer at 95% capacity (95000/100000)" |
| Low occupancy | Warning | "Kernel 'emit' occupancy: 25% (consider reducing registers)" |

### Logging

Standard `ILogger` via `NodeContext.GetLogger()`. Logs appear in VL's log viewer (`AppHost.CurrentDefaultLogger`):

```csharp
// Block-level logging
_logger.LogInformation("Kernel loaded: {EntryPoint} from {Path}", entryPoint, ptxPath);
_logger.LogWarning("Buffer at {Percent}% capacity", percentUsed);

// Engine-level logging  
_logger.LogInformation("Cold rebuild: {NodeCount} nodes, {KernelCount} kernels, {Time}ms",
    nodeCount, kernelCount, rebuildTime);
_logger.LogError(ex, "CUDA launch failed");
```

---

## Visual Summary

```
┌──────────────────────────────────────────────────────────────────┐
│ VL Patch                                                          │
│                                                                   │
│  ┌──────────────┐                              ┌──────────────┐  │
│  │ Emitter  ⚠   │  "GpuBuffer<Particle>        │ Forces       │  │
│  │              │   45231 elem, 708KB"         │              │  │
│  │         Out ─┼──────────────────────────────┼─ In          │  │
│  │  "0.42ms"    │      (Link Tooltip)          │  "0.15ms"    │  │
│  └──────────────┘                              └──────────────┘  │
│    (orange: "Buffer at 95%")                    (Node Tooltip)    │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ CudaEngine                                                  │  │
│  │ ProfilingLevel: [PerBlock ▼]                                │  │
│  │                                                             │  │
│  │ "CudaEngine: 4 blocks, 12 kernels                          │  │
│  │   Launch: 2.1ms                                             │  │
│  │   Pool: 12 MB / 64 MB"                                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  VL Log:                                                          │
│    [INF] Cold rebuild: 12 nodes, 8 kernels, 3.2ms                │
│    [WRN] EmitterBlock: Buffer at 95% capacity                    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```
