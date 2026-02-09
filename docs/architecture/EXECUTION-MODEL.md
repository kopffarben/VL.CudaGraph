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
Block created:
    new EmitterBlock(NodeContext nodeContext, CudaContext cudaContext)
        │
        ├── nodeContext stored (for identity/logging/app access)
        ├── BlockBuilder(cudaContext, this) created
        ├── Define kernels, ports, params (declarative)
        ├── builder.Commit()   → wires param events, stores BlockDescription
        ├── cudaContext.RegisterBlock(this)   // facade: delegates to Registry
        └── Registry fires StructureChanged → DirtyTracker marks structure dirty

VL calls every frame (optional):
    block.Update()
        │
        └── Only updates debug display (reads DebugInfo)
            Push parameter changes (TypedValue = ...) → ValueChanged fires
            No GPU work happens here

Hot-Swap (code change):
    block.Dispose()
        │
        ├── cudaContext.UnregisterBlock(this.Id)
        │     → removes from Registry + ConnectionGraph + BlockDescriptions
        └── Registry fires StructureChanged → DirtyTracker marks structure dirty

    new EmitterBlock(nodeContext, cudaContext)   ← new instance, new code
    VL restores connections via ctx.Connect() calls
```

### CudaEngine Lifecycle

```
Engine created:
    new CudaEngine(CudaEngineOptions? options)
        │
        └── Context = new CudaContext(options)
            _stream = new ManagedCuda.CudaStream()

Called every frame:
    cudaEngine.Update()
        │
        ├── 1. IsStructureDirty → ColdRebuild → ClearStructureDirty
        ├── 2. Else AreParametersDirty → UpdateParameters → ClearParametersDirty
        ├── 3. If compiled graph exists: Launch + Synchronize
        └── 4. Distribute DebugInfo to all blocks (BlockState + LastExecutionTime)

Dispose:
    → Disposes CompiledGraph, owned KernelNodes, stream, CudaContext
```

### Constructor Signatures

```csharp
public sealed class CudaEngine : IDisposable
{
    // Production constructor. NodeContext is injected by VL as first parameter.
    public CudaEngine(NodeContext nodeContext, CudaEngineOptions? options = null)
    {
        NodeContext = nodeContext;
        Context = new CudaContext(options ?? new CudaEngineOptions());
        _stream = new ManagedCuda.CudaStream();
    }

    // Test-friendly constructor with injected CudaContext
    internal CudaEngine(CudaContext context);
}

public class EmitterBlock : ICudaBlock
{
    public EmitterBlock(NodeContext nodeContext, CudaContext cudaContext)
    {
        NodeContext = nodeContext;

        var builder = new BlockBuilder(cudaContext, this);
        // ... define kernels, pins ...
        builder.Commit();
        cudaContext.RegisterBlock(this);
    }

    public void Dispose()
    {
        _cudaContext.UnregisterBlock(this.Id);
    }
}
```

**Future VL integration** will add:
- `IVLRuntime` diagnostics (AddMessage, AddPersistentMessage)
- `AppHost.TakeOwnership(this)` for cleanup on app shutdown
- `nodeContext.GetLogger()` for structured logging

---

## Dirty-Tracking

### Update Tiers

The system has two node types with different update capabilities. See `KERNEL-SOURCES.md` for the three kernel sources and two node types.

**KernelNode** (filesystem PTX & patchable kernels):

| Tier | Cost | Trigger | Action |
|------|------|---------|--------|
| **Hot Update** | Near-zero, no GPU stall | Scalar value changed (gravity, deltaTime) | `cuGraphExecKernelNodeSetParams` for scalar only |
| **Warm Update** | Cheap, exec-level update | Buffer rebind (different pointer, same size), grid size change | `cuGraphExecKernelNodeSetParams` for pointer/grid |
| **Code Rebuild** | Medium | Patchable kernel logic changed | ILGPU IR recompile (~1-10ms) → new CUmodule → Cold Rebuild of affected block |
| **Cold Rebuild** | Expensive, full graph rebuild | Node added/removed, edge added/removed, PTX hot-reload, block structure changed | Destroy old graph, build new, instantiate |

**CapturedNode** (Library Calls — cuBLAS, cuFFT, cuDNN):

| Tier | Cost | Trigger | Action |
|------|------|---------|--------|
| **Recapture** | Medium | Any parameter changed (scalar or pointer) | Recapture Library Call + `cuGraphExecChildGraphNodeSetParams` |
| **Cold Rebuild** | Expensive, full graph rebuild | Node added/removed, structure changed | Destroy old graph, build new, instantiate |

**Key difference:** KernelNodes support Hot Update/Warm Update parameter patching at near-zero cost. CapturedNodes require Recapture for any parameter change because library kernels are opaque. Exception: with `CUBLAS_POINTER_MODE_DEVICE`, scalar-only changes can avoid Recapture.

### Design Rationale

VL compiles .vl files on-the-fly to C# and supports Hot-Swap: when the user edits a block's code, VL disposes the old instance and creates a new one with the updated code. VL then restores connections by calling `Connect()` again.

Structural changes (Cold Rebuild) happen during development only. During runtime of an exported application, only Hot Updates and Warm Updates occur. The graph rebuild may take several milliseconds, which is acceptable during interactive development.

### Dirty Flags

**Current implementation (Phase 2):**

```csharp
// CudaContext is a facade — dirty state lives in DirtyTracker
class CudaContext
{
    // Service properties
    public DirtyTracker Dirty { get; }
    public BlockRegistry Registry { get; }
    public ConnectionGraph Connections { get; }

    // --- Facade methods that trigger structure dirty ---
    // Delegate to Registry/ConnectionGraph, which fire StructureChanged → DirtyTracker subscribes
    public void RegisterBlock(ICudaBlock block);
    public void UnregisterBlock(Guid blockId);
    public void Connect(Guid srcId, string srcPort, Guid tgtId, string tgtPort);
    public void Disconnect(Guid srcId, string srcPort, Guid tgtId, string tgtPort);

    // --- Parameter dirty (called by BlockParameter.ValueChanged event wiring) ---
    public void OnParameterChanged(Guid blockId, string paramName);
}

// DirtyTracker (subscribes to Registry + ConnectionGraph events)
class DirtyTracker
{
    public bool IsStructureDirty { get; }    // starts true → first build
    public bool AreParametersDirty { get; }  // true if any dirty params

    public void Subscribe(BlockRegistry registry, ConnectionGraph connectionGraph);
    public void MarkParameterDirty(DirtyParameter param);
    public IReadOnlySet<DirtyParameter> GetDirtyParameters();
    public void ClearStructureDirty();   // also clears params (rebuild applies all)
    public void ClearParametersDirty();
}
```

**Planned additions (Phase 4b):**
- `IsCodeDirty` / `MarkCodeDirty(Guid blockId)` — patchable kernel recompile tracking

### Block Structure Change Detection

When VL hot-swaps a block, the new instance calls `BlockBuilder.Commit()` in its constructor. The `BlockBuilder` produces a `BlockDescription` (ordered list of `KernelEntry` objects, ports, internal connections). This is compared against the previous description for the same block ID:

```csharp
class BlockBuilder
{
    public void Commit()
    {
        var description = BuildDescription();
        var oldDescription = _context.GetBlockDescription(_block.Id);

        // If structure changed, the Register/Unregister cycle already
        // fires StructureChanged via BlockRegistry. This catches in-place changes.

        _context.SetBlockDescription(_block.Id, description);
    }
}
```

`BlockDescription.StructuralEquals()` checks:
- Same `KernelEntry` list (PtxPath + EntryPoint + GridDimX/Y/Z, in order — HandleId ignored)
- Same port set (Name + Direction + PinType)
- Same internal connections (using kernel indices, not GUIDs — stable across reconstruction)

### Execution Flow Per Frame

**Current implementation (Phase 2):**

```csharp
class CudaEngine
{
    public void Update()
    {
        // Priority: Structure > Parameters > Launch
        if (Context.Dirty.IsStructureDirty)
        {
            ColdRebuild();
            Context.Dirty.ClearStructureDirty();
        }
        else if (Context.Dirty.AreParametersDirty)
        {
            UpdateParameters();  // Hot Update via CompiledGraph.UpdateScalar
            Context.Dirty.ClearParametersDirty();
        }

        if (_compiledGraph != null)
        {
            _compiledGraph.Launch(_stream);
            _stream.Synchronize();
            DistributeDebugInfo(BlockState.OK);
        }
    }

    private void UpdateParameters()
    {
        foreach (var dirty in Context.Dirty.GetDirtyParameters())
        {
            var block = Context.Registry.Get(dirty.BlockId);
            var param = block?.Parameters.FirstOrDefault(p => p.Name == dirty.ParamName);
            if (param == null) continue;

            if (_paramMapping.TryGetValue((dirty.BlockId, dirty.ParamName), out var mapping))
            {
                // Hot Update via CompiledGraph.UpdateScalar — near-zero cost
                _compiledGraph.UpdateScalar(mapping.KernelNodeId, mapping.ParamIndex, param.Value);
            }
        }
    }
}
```

**Planned additions (Phase 4a+):**
- `CodeRebuild()` — ILGPU IR recompile → targeted Cold rebuild (Phase 4b)
- `UpdateDirtyNodes()` — dispatch by node type: KernelNode → Hot/Warm, CapturedNode → Recapture (Phase 4a)
- `ProfilingPipeline.OnPostLaunch()` — async GPU event readback (Phase 3+)
- `ReportDiagnostics()` — IVLRuntime integration (Phase 3+)

---

## Profiling *(Phase 3+ — not yet implemented)*

> **Note:** This entire section describes **planned** profiling infrastructure. The current Phase 2 implementation distributes only `BlockDebugInfo` (State, StateMessage, LastExecutionTime) via `CudaEngine.DistributeDebugInfo()`. The types below (`ProfilingPipeline`, `ProfilingLevel`, `FrameDebugData`) do not exist yet.

### Profiling Levels

```csharp
// Planned (Phase 3+)
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
// Planned (Phase 3+) — types do not exist yet
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
    /// Returns IBlockDebugInfo (or a richer subtype when profiling is active).
    /// </summary>
    public IBlockDebugInfo? GetCached(Guid blockId)
    {
        return _current?.GetBlock(blockId);
    }
}
```

---

## Debug Display *(Phase 3+ — design preview)*

> **Note:** The examples below show the **planned** tooltip design. In the current Phase 2 implementation, blocks only have `BlockDebugInfo` (State, StateMessage, LastExecutionTime) written by `CudaEngine.DistributeDebugInfo()`. The `ProfilingPipeline` and `OutputHandle` types referenced below do not exist yet.

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

Two separate mechanisms for two separate purposes:

### Errors/Warnings → IVLRuntime

VL displays warnings (node turns orange) and errors (node turns red) directly on nodes. CudaEngine uses `IVLRuntime` to report diagnostics after each frame. This is VL's native mechanism — no custom error display system needed.

```csharp
class CudaEngine
{
    private IVLRuntime _runtime;
    private Dictionary<Guid, IDisposable?> _persistentMessages = new();

    private void ReportDiagnostics()
    {
        foreach (var reg in _cudaContext.Registry.All)
        {
            var diag = GetDiagnostics(reg.Block.Id);
            var elementId = reg.NodeContext.Path.Stack.Peek();

            if (diag.HasError)
            {
                // Persistent: stays until error is resolved
                _persistentMessages[reg.Block.Id] ??=
                    _runtime.AddPersistentMessage(
                        new Message(elementId, MessageSeverity.Error, diag.ErrorText));
            }
            else
            {
                // Clear persistent error if resolved
                _persistentMessages[reg.Block.Id]?.Dispose();
                _persistentMessages[reg.Block.Id] = null;

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

### Timing/Stats → ToString / DebugInfo

Profiling data (execution times, buffer statistics) is displayed via VL's standard tooltip mechanism. `ToString()` is called only when the user hovers a node — zero overhead when not observed. Data comes from the async profiling pipeline, not from live GPU queries.

### Diagnostic Sources

| Source | Severity | Example |
|--------|----------|---------|
| PTX load failure | Error | "Failed to load kernel.ptx: file not found" |
| CUDA launch error | Error | "CUDA_ERROR_LAUNCH_FAILED in particle_emit" |
| Graph validation | Error | "Type mismatch: GpuBuffer\<float4\> → GpuBuffer\<float3\>" |
| Cross-engine connection | Error | "Block uses different CudaContext than engine" |
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
