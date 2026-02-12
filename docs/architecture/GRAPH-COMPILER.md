# Graph Compiler

> **Note:** The Graph Compiler is invoked by `CudaEngine` during Cold Rebuilds. Blocks never call it directly. See `EXECUTION-MODEL.md` for when and why rebuilds happen.

## Overview

The Graph Compiler transforms the high-level graph description into an executable CUDA Graph.

```
CudaEngine.Update() detects structureDirty
         │
         ▼
    ┌─────────────────────────────────────┐
    │         Graph Compiler               │
    │                                      │
    │  1. Validation                       │  ← Phase 1 (implemented)
    │  2. Topological Sort                 │  ← Phase 1 (implemented)
    │  3. Buffer Allocation                │  ← Phase 1 (implemented)
    │  4. Wire Parameters                  │  ← Phase 1 (implemented)
    │  4.5 Memset Nodes (AppendBuffers)    │  ← Phase 3.1 (implemented)
    │  5b. Stream Capture (CapturedNodes)  │  ← Phase 4a (implemented)
    │  5c. Kernel node deps on captured    │  ← Phase 4a (implemented)
    │  6. CUDA Graph Build                 │  ← Phase 1 (implemented)
    │  7. Instantiate                      │  ← Phase 1 (implemented)
    │  ─── Planned ───────────────────── │
    │  Shape Propagation                   │  ← Phase 3 (planned)
    │  Liveness Analysis                   │  ← Phase 3 (planned)
    │                                      │
    └─────────────────────────────────────┘
         │
         ▼
    CompiledGraph (CUgraphExec)
         │
         ▼
    CudaEngine stores for subsequent Launch() calls
```

## Compilation Phases

> **Current implementation (Phase 1-4a):** The compiler performs Validation → Topological Sort → Buffer Allocation → Wire Parameters → Memset Nodes (AppendBuffer counters) → Stream Capture (CapturedNodes) → Kernel dependency wiring → CUDA Graph Build → Instantiate. Shape Propagation and Liveness Analysis are Phase 3 features.

### Phase 1: Validation

```csharp
class ValidationResult
{
    bool IsValid;
    List<DiagnosticMessage> Errors;
    List<DiagnosticMessage> Warnings;
}
```

Checks performed:
- **Type compatibility**: Source/target pin types match
- **Cycle detection**: No circular dependencies (DAG requirement)
- **Completeness**: All required inputs connected
- **Shape compatibility**: Dimension rules satisfied
- **Resource availability**: Kernels loadable, memory sufficient

Validation errors are distributed to blocks as DebugInfo by CudaEngine — they appear as tooltips in VL.

### Phase 2: Buffer Allocation

```csharp
void AllocateBuffers(GraphDescription desc, BufferPool pool)
{
    var buffers = new Dictionary<Guid, GpuBuffer>();
    
    foreach (var b in desc.Buffers.Where(b => b.Lifetime == External))
        buffers[b.Id] = externalBuffers[b.Id];
    
    foreach (var b in desc.Buffers.Where(b => b.Lifetime == Graph))
        buffers[b.Id] = pool.Acquire(b.ElementType, b.Shape);
    
    foreach (var b in desc.Buffers.Where(b => b.Lifetime == Region))
        buffers[b.Id] = pool.Acquire(b.ElementType, b.Shape);
    
    foreach (var region in desc.Children.Where(c => c.Type == If))
        foreach (var output in GetRegionOutputs(region))
            EnsureZeroInitialized(output, buffers);
}
```

### Phase 3: Topological Sort

```
Topological levels:
  Level 0: A           (no dependencies)
  Level 1: B, C        (depend on A, can run parallel)
  Level 2: D           (depends on B and C)

CUDA Graph automatically parallelizes nodes at same level.
```

### Phase 4.5: Memset Nodes (AppendBuffer Counter Reset)

After Wire Parameters and before kernel node insertion, the compiler generates memset nodes to zero-initialize AppendBuffer counters. This ensures counters are reset at the start of every graph launch without any manual intervention.

#### MemsetDescriptor

Each `AppendBufferInfo` in the `BlockDescription` produces a `MemsetDescriptor`:

```csharp
public sealed class MemsetDescriptor
{
    public CUdeviceptr Pointer { get; }   // Counter buffer pointer
    public uint Value { get; }            // Always 0 (reset)
    public int SizeInBytes { get; }       // sizeof(uint) = 4
}
```

#### Graph Construction

```csharp
// For each AppendBuffer, insert a memset node:
foreach (var memsetDesc in memsetDescriptors)
{
    var memsetParams = new CudaMemsetNodeParams
    {
        dst = memsetDesc.Pointer,
        value = memsetDesc.Value,       // 0
        elementSize = sizeof(uint),
        width = 1,
        height = 1
    };
    var memsetNode = graph.AddMemsetNode(memsetParams, dependencies: Array.Empty<CUgraphNode>());
    memsetNodes.Add(memsetNode);
}
```

#### Dependency Wiring

Memset nodes execute **before** any kernel nodes that use the corresponding append buffer. The compiler adds each memset node as an additional dependency for kernel nodes that write to that append output:

```
Memset(Counter=0)  ──dependency──▶  KernelNode (appends to buffer)
```

This guarantees the counter is zeroed before the kernel's `atomicAdd` operations begin. Multiple kernel nodes writing to the same append buffer all depend on the same memset node.

### Phase 4: Shape Propagation *(Phase 3 — not yet implemented)*

```
Shape Rules:
  "same"           → output.shape = input.shape
  "same(0)"        → output.shape = inputs[0].shape
  "broadcast(0,1)" → output.shape = broadcast(inputs[0], inputs[1])
  "[N, 16]"        → output.shape = [inputs[0].dim0, 16]
```

### Phase 5: Liveness Analysis *(Phase 3 — not yet implemented)*

Determines when buffers can be reused:

```
Timeline:
  t0: Kernel1 produces Buffer_A
  t1: Kernel2 reads Buffer_A, produces Buffer_B
  t2: Kernel3 reads Buffer_A (last use!)
  t3: Kernel4 reads Buffer_B

Buffer_A live range: [t0, t2] → can be released after t2
```

### Phase 5b: Stream Capture (CapturedNodes)

For each `CapturedNode` in the graph description, the compiler performs stream capture on a **dedicated capture stream** (separate from the main execution stream) and inserts the result as a child graph node:

```csharp
// GraphCompiler — Phase 5b
using var captureStream = new CudaStream();
foreach (var capturedNode in desc.CapturedNodes)
{
    // 1. Perform stream capture: calls CaptureAction with buffer bindings
    var childGraph = capturedNode.Capture(captureStream.Stream);

    // 2. Collect dependencies from CapturedDependency records
    var deps = new List<CUgraphNode>();
    foreach (var dep in desc.CapturedDependencies.Where(d => d.TargetNodeId == capturedNode.Id))
    {
        // Source can be a kernel node or another captured node
        if (kernelNodeHandles.TryGetValue(dep.SourceNodeId, out var kernelHandle))
            deps.Add(kernelHandle);
        if (capturedNodeHandles.TryGetValue(dep.SourceNodeId, out var capHandle))
            deps.Add(capHandle);
    }

    // 3. Insert as child graph node
    var childNode = StreamCaptureHelper.AddChildGraphNode(graph.Graph, deps.ToArray(), childGraph);
    capturedNodeHandles[capturedNode.Id] = childNode;
}
```

This phase only runs for CapturedNodes. KernelNodes skip this phase entirely. The `CapturedNode` owns its captured `CUgraph` handle and destroys the previous one on recapture or dispose.

### Phase 5c: Kernel Node Dependencies on Captured Nodes

After captured nodes are inserted, the compiler builds the dependency map for kernel nodes. If a kernel node depends on a captured node (via `CapturedDependency`), the captured node's `CUgraphNode` handle is included in the kernel node's dependency array. This allows mixed graphs where kernel nodes consume outputs from captured library operations.

### Phase 6: CUDA Graph Build

Constructs the actual CUDA Graph structure from topological order and dependencies:
- **KernelNodes**: `cuGraphAddKernelNode` (standard path for both filesystem PTX and NVRTC-compiled kernels)
- **CapturedNodes**: Already inserted in Phase 5b as `cuGraphAddChildGraphNode`

### Phase 7: Instantiate

```csharp
CompiledGraph Instantiate(CudaGraph graph)
{
    var exec = graph.Instantiate();
    return new CompiledGraph
    {
        Executable = exec,
        // ...
        capturedNodeHandles   // Dict<Guid, CUgraphNode> for Recapture updates
    };
}
```

`CompiledGraph` stores `capturedNodeHandles` alongside the existing `nodeHandles` and `memsetNodeHandles`. The `RecaptureNode()` method uses `StreamCaptureHelper.UpdateChildGraphNode()` to perform in-place child graph updates without a full graph rebuild:

```csharp
internal void RecaptureNode(Guid nodeId, CUgraph newChildGraph)
{
    var graphNode = _capturedNodeHandles[nodeId];
    StreamCaptureHelper.UpdateChildGraphNode(_exec.Graph, graphNode, newChildGraph);
}
```

---

## Regions

Regions enable control flow within the graph.

### If Region (Conditional)

```
┌──────────────────────────────────────┐
│ IF Region                            │
│  Condition: ConditionBuffer[0] != 0  │
│  Zero-fill outputs before region     │
│  ┌────────────────────────────────┐  │
│  │ Body Graph                     │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

### While Region (Loop)

```
┌──────────────────────────────────────┐
│ WHILE Region                         │
│  while (ConditionBuffer[0] != 0)     │
│  { Body Graph }                      │
└──────────────────────────────────────┘
```

### For Region (Bounded Loop)

```
┌──────────────────────────────────────┐
│ FOR Region                           │
│  for (i = 0; i < N; i++)             │
│  { Body Graph }                      │
└──────────────────────────────────────┘
```

---

## Update Levels

> See `EXECUTION-MODEL.md` for the full dirty-tracking design.
> See `KERNEL-SOURCES.md` for the three kernel sources and two node types.

### KernelNode Updates (Filesystem PTX & Patchable Kernels)

| Change | Update Level |
|--------|--------------|
| Scalar value changed | **Hot Update** — `cuGraphExecKernelNodeSetParams` |
| Buffer content changed | None needed (data already on GPU) |
| Buffer rebind (same size) | **Warm Update** — `cuGraphExecKernelNodeSetParams` |
| Buffer resize | **Warm Update** + Pool realloc |
| Grid size changed | **Warm Update** — `cuGraphExecKernelNodeSetParams` |
| Patchable kernel logic changed | **Code Rebuild** — Recompile (ILGPU IR or NVRTC) → new CUmodule → Cold Rebuild of affected block |
| Node added/removed | **Cold Rebuild** — Full Rebuild |
| Edge added/removed | **Cold Rebuild** — Full Rebuild |
| Region added/removed | **Cold Rebuild** — Full Rebuild |
| PTX changed (hot-reload) | **Cold Rebuild** — Full Rebuild |

### CapturedNode Updates (Library Calls)

| Change | Update Level |
|--------|--------------|
| Scalar or pointer parameter changed | **Recapture** — Recapture + `cuGraphExecChildGraphNodeSetParams` |
| Scalar with DEVICE pointer mode* | **Warm Update** — Buffer write, avoids Recapture |
| Node added/removed | **Cold Rebuild** — Full Rebuild |

*With `CUBLAS_POINTER_MODE_DEVICE`, scalar values live in GPU memory and can be updated via buffer writes.

---

## Profiling

When profiling is enabled, event nodes are inserted:

```
With ProfilingLevel.PerKernel:
  KernelNode:   [Event: Start] → [Kernel1] → [Event: Stop]     ← per-kernel timing
  CapturedNode: [Event: Start] → [ChildGraph] → [Event: Stop]  ← per-block only (opaque)
```

Timing data is collected by CudaEngine and distributed to blocks as DebugInfo.

**CapturedNode profiling limitation:** Library operations in CapturedNodes run opaque internal kernels. Profiling can only measure the total time of the captured child graph, not individual internal kernels. PerKernel profiling degrades to PerBlock for CapturedNodes. See `KERNEL-SOURCES.md` for details.

---

## Error Handling

Errors during compilation are caught by CudaEngine and distributed to affected blocks. The engine does not crash — it shows errors as tooltips and keeps the old compiled graph if available.
