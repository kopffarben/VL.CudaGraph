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
    │  1. Validation                       │
    │  2. Buffer Allocation                │
    │  3. Topological Sort                 │
    │  4. Shape Propagation                │
    │  5. Liveness Analysis                │
    │  5.5 Stream Capture (CapturedNodes)  │
    │  6. CUDA Graph Build                 │
    │  7. Instantiate                      │
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

### Phase 4: Shape Propagation

```
Shape Rules:
  "same"           → output.shape = input.shape
  "same(0)"        → output.shape = inputs[0].shape
  "broadcast(0,1)" → output.shape = broadcast(inputs[0], inputs[1])
  "[N, 16]"        → output.shape = [inputs[0].dim0, 16]
```

### Phase 5: Liveness Analysis

Determines when buffers can be reused:

```
Timeline:
  t0: Kernel1 produces Buffer_A
  t1: Kernel2 reads Buffer_A, produces Buffer_B
  t2: Kernel3 reads Buffer_A (last use!)
  t3: Kernel4 reads Buffer_B

Buffer_A live range: [t0, t2] → can be released after t2
```

### Phase 5.5: Stream Capture (CapturedNodes only)

For CapturedNode descriptors (library operations like cuBLAS, cuFFT), the compiler executes Stream Capture to produce child graphs:

```
For each CapturedNodeDescriptor:
  cuStreamBeginCapture(stream, RELAXED)
  descriptor.CaptureAction(stream)     // e.g. cublasSgemm(...)
  cuStreamEndCapture(stream) → childGraph
  Store childGraph for Phase 6
```

This phase only runs for CapturedNodes. KernelNodes (from filesystem PTX or NVRTC) skip this phase entirely. See `KERNEL-SOURCES.md` for the full CapturedNode design.

### Phase 6: CUDA Graph Build

Constructs the actual CUDA Graph structure from topological order and dependencies:
- **KernelNodes**: `cuGraphAddKernelNode` (standard path for both filesystem PTX and NVRTC-compiled kernels)
- **CapturedNodes**: `cuGraphAddChildGraphNode` (using child graph from Phase 5.5)

### Phase 7: Instantiate

```csharp
CompiledGraph Instantiate(CudaGraph graph)
{
    var exec = graph.Instantiate();
    return new CompiledGraph { Executable = exec, ... };
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
| Scalar value changed | **Hot** — `cuGraphExecKernelNodeSetParams` |
| Buffer content changed | None needed (data already on GPU) |
| Buffer rebind (same size) | **Warm** — `cuGraphExecKernelNodeSetParams` |
| Buffer resize | **Warm** + Pool realloc |
| Grid size changed | **Warm** — `cuGraphExecKernelNodeSetParams` |
| Patchable kernel logic changed | **Code** — NVRTC recompile → new CUmodule → Cold rebuild of affected block |
| Node added/removed | **Cold** — Full Rebuild |
| Edge added/removed | **Cold** — Full Rebuild |
| Region added/removed | **Cold** — Full Rebuild |
| PTX changed (hot-reload) | **Cold** — Full Rebuild |

### CapturedNode Updates (Library Operations)

| Change | Update Level |
|--------|--------------|
| Scalar or pointer parameter changed | **Recapture** — Re-capture + `cuGraphExecChildGraphNodeSetParams` |
| Scalar with DEVICE pointer mode* | **Warm** — Buffer write, avoids re-capture |
| Node added/removed | **Cold** — Full Rebuild |

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
