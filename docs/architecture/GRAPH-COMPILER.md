# Graph Compiler

## Overview

The Graph Compiler transforms the high-level graph description into an executable CUDA Graph.

```
GraphDescription (declarative)
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
    │  6. CUDA Graph Build                 │
    │  7. Instantiate                      │
    │                                      │
    └─────────────────────────────────────┘
         │
         ▼
    CompiledGraph (CUgraphExec)
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

```csharp
// Example validation error
error: Type mismatch on edge
  Source: Emitter.Particles (GpuBuffer<float4>)
  Target: Forces.Positions (GpuBuffer<float3>)
  Expected: GpuBuffer<float3>
```

### Phase 2: Buffer Allocation

```csharp
void AllocateBuffers(GraphDescription desc, BufferPool pool)
{
    var buffers = new Dictionary<Guid, GpuBuffer>();
    
    // External: reference existing buffers
    foreach (var b in desc.Buffers.Where(b => b.Lifetime == External))
        buffers[b.Id] = externalBuffers[b.Id];
    
    // Graph-lifetime: allocate from pool
    foreach (var b in desc.Buffers.Where(b => b.Lifetime == Graph))
        buffers[b.Id] = pool.Acquire(b.ElementType, b.Shape);
    
    // Region-lifetime: allocate, track for later release
    foreach (var b in desc.Buffers.Where(b => b.Lifetime == Region))
        buffers[b.Id] = pool.Acquire(b.ElementType, b.Shape);
    
    // Conditional outputs: pre-allocate with zero-fill
    foreach (var region in desc.Children.Where(c => c.Type == If))
        foreach (var output in GetRegionOutputs(region))
            EnsureZeroInitialized(output, buffers);
}
```

### Phase 3: Topological Sort

Determines execution order based on data dependencies:

```
Input edges:
  A.out → B.in
  A.out → C.in
  B.out → D.in
  C.out → D.in

Topological levels:
  Level 0: A           (no dependencies)
  Level 1: B, C        (depend on A, can run parallel)
  Level 2: D           (depends on B and C)

CUDA Graph automatically parallelizes nodes at same level.
```

### Phase 4: Shape Propagation

Infers output shapes from input shapes using rules:

```
Shape Rules:
  "same"           → output.shape = input.shape
  "same(0)"        → output.shape = inputs[0].shape
  "broadcast(0,1)" → output.shape = broadcast(inputs[0], inputs[1])
  "[N, 16]"        → output.shape = [inputs[0].dim0, 16]
```

Example:
```
Kernel: MatMul
  Inputs:
    A: shape [M, K]
    B: shape [K, N]
  Output:
    C: shape_rule = "[A.dim0, B.dim1]" → [M, N]
```

### Phase 5: Liveness Analysis

Determines when buffers can be reused:

```
Timeline:
  t0: Kernel1 produces Buffer_A
  t1: Kernel2 reads Buffer_A, produces Buffer_B
  t2: Kernel3 reads Buffer_A (last use!)
  t3: Kernel4 reads Buffer_B

Buffer_A live range: [t0, t2]
Buffer_B live range: [t1, t3]

After t2, Buffer_A can be released/reused.
```

Implementation:
```csharp
Dictionary<Guid, (int firstUse, int lastUse)> ComputeLiveRanges(
    List<NodeDesc> topoOrder,
    Dictionary<Guid, List<EdgeDesc>> bufferUsages)
{
    var ranges = new Dictionary<Guid, (int, int)>();
    
    for (int t = 0; t < topoOrder.Count; t++)
    {
        var node = topoOrder[t];
        
        foreach (var buffer in GetBuffersUsedBy(node))
        {
            if (!ranges.ContainsKey(buffer))
                ranges[buffer] = (t, t);
            else
                ranges[buffer] = (ranges[buffer].firstUse, t);
        }
    }
    
    return ranges;
}
```

### Phase 6: CUDA Graph Build

Constructs the actual CUDA Graph structure:

```csharp
void BuildGraph(GraphDescription desc, Dictionary<Guid, GpuBuffer> buffers)
{
    var graph = new CudaGraph();
    var handles = new Dictionary<Guid, CUgraphNode>();
    
    foreach (var node in topoOrder)
    {
        var deps = GetDependencyNodes(node, handles);
        
        switch (node.Type)
        {
            case NodeType.Kernel:
                handles[node.Id] = AddKernelNode(graph, node, buffers, deps);
                break;
                
            case NodeType.Memset:
                handles[node.Id] = AddMemsetNode(graph, node, buffers, deps);
                break;
                
            case NodeType.Library:
                handles[node.Id] = AddLibraryNode(graph, node, buffers, deps);
                break;
        }
    }
    
    // Handle regions (conditional/loops)
    foreach (var region in desc.Children)
    {
        BuildRegion(graph, region, buffers, handles);
    }
}
```

#### Kernel Node

```csharp
CUgraphNode AddKernelNode(
    CudaGraph graph,
    NodeDesc node,
    Dictionary<Guid, GpuBuffer> buffers,
    CUgraphNode[] dependencies)
{
    // Load kernel from cache
    var module = ModuleCache.Get(node.Kernel.PTXPath);
    var function = module.GetFunction(node.Kernel.EntryPoint);
    
    // Build parameter list
    var args = new List<object>();
    foreach (var binding in node.Bindings)
    {
        if (binding.IsBuffer)
            args.Add(buffers[binding.BufferId].Pointer);
        else
            args.Add(binding.ScalarValue);
    }
    
    // Calculate grid size
    var (gridDim, blockDim) = CalculateGrid(node.Grid, buffers);
    
    // Create node
    var nodeParams = new CUDA_KERNEL_NODE_PARAMS
    {
        func = function,
        gridDimX = gridDim.x,
        gridDimY = gridDim.y,
        gridDimZ = gridDim.z,
        blockDimX = blockDim.x,
        blockDimY = blockDim.y,
        blockDimZ = blockDim.z,
        kernelParams = args.ToArray(),
        sharedMemBytes = node.Kernel.SharedMemoryStatic
    };
    
    return graph.AddKernelNode(nodeParams, dependencies);
}
```

### Phase 7: Instantiate

```csharp
CompiledGraph Instantiate(CudaGraph graph)
{
    // Instantiate creates an executable graph
    var exec = graph.Instantiate();
    
    return new CompiledGraph
    {
        Executable = exec,
        NodeHandles = handles,
        Buffers = buffers,
        StructureVersion = ctx.StructureVersion
    };
}
```

---

## Regions

Regions enable control flow within the graph.

### If Region (Conditional)

```
┌──────────────────────────────────────┐
│ IF Region                            │
│                                      │
│  Condition: ConditionBuffer[0] != 0  │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ Body Graph                     │  │
│  │   [Kernel1] → [Kernel2] → ...  │  │
│  └────────────────────────────────┘  │
│                                      │
│  Zero-fill outputs before region     │
│  (ensures valid data if skipped)     │
│                                      │
└──────────────────────────────────────┘
```

Implementation:
```csharp
void AddIfRegion(
    CudaGraph graph,
    GraphDescription region,
    Dictionary<Guid, GpuBuffer> buffers,
    Dictionary<Guid, CUgraphNode> handles)
{
    // 1. Zero-fill outputs BEFORE the condition
    foreach (var output in GetRegionOutputs(region))
    {
        var buf = buffers[output];
        graph.AddMemsetNode(buf.Pointer, 0, buf.SizeInBytes);
    }
    
    // 2. Create condition handle
    var condHandle = graph.CreateConditionalHandle(
        defaultValue: 0,  // Default: don't execute
        flags: CU_GRAPH_COND_ASSIGN_DEFAULT
    );
    
    // 3. Add kernel that sets condition
    AddConditionKernel(graph, region.ConditionBuffer, condHandle);
    
    // 4. Create conditional node with body
    var bodyGraph = new CudaGraph();
    BuildGraph(region, buffers, bodyGraph);  // Recursive!
    
    graph.AddConditionalNode(
        type: CU_GRAPH_COND_TYPE_IF,
        handle: condHandle,
        bodyGraph: bodyGraph
    );
}
```

### While Region (Loop)

```
┌──────────────────────────────────────┐
│ WHILE Region                         │
│                                      │
│  while (ConditionBuffer[0] != 0)     │
│  {                                   │
│    ┌────────────────────────────┐    │
│    │ Body Graph                 │    │
│    │   [Kernel1] → [Kernel2]    │    │
│    │   (may update condition)   │    │
│    └────────────────────────────┘    │
│  }                                   │
│                                      │
└──────────────────────────────────────┘
```

### For Region (Bounded Loop)

```
┌──────────────────────────────────────┐
│ FOR Region                           │
│                                      │
│  for (i = 0; i < N; i++)             │
│  {                                   │
│    ┌────────────────────────────┐    │
│    │ Body Graph                 │    │
│    │   [Kernel1] → [Kernel2]    │    │
│    │   (can read loop index)    │    │
│    └────────────────────────────┘    │
│  }                                   │
│                                      │
└──────────────────────────────────────┘
```

---

## Rebuild vs Update

| Change | Action |
|--------|--------|
| Scalar value changed | Parameter Update |
| Buffer content changed | No update needed (data already on GPU) |
| Buffer rebind (same size) | Parameter Update |
| Buffer resize | Parameter Update + Pool realloc |
| Node added/removed | **Full Rebuild** |
| Edge added/removed | **Full Rebuild** |
| Region added/removed | **Full Rebuild** |
| Grid size changed | Parameter Update |
| PTX changed (hot-reload) | **Full Rebuild** |

```csharp
// Parameter update (fast)
compiledGraph.SetScalar(nodeId, paramIndex, newValue);
compiledGraph.UpdatePointer(nodeId, paramIndex, newPointer);

// Full rebuild (when structure changes)
if (ctx.NeedsRecompile)
{
    ctx.Compile();  // Creates new CompiledGraph
}
```

---

## Execution

```csharp
// Launch on default stream
compiledGraph.Launch();

// Launch on specific stream
compiledGraph.Launch(stream);

// The graph executes:
// 1. All nodes in topological order
// 2. Parallel nodes launched concurrently
// 3. Conditionals evaluated on-GPU
// 4. Loops iterate on-GPU (no CPU roundtrip)
```

---

## Profiling

When profiling is enabled, event nodes are inserted:

```
With ProfilingLevel.PerKernel:

  [Event: Start] → [Kernel1] → [Event: Stop]
        ↓
  [Event: Start] → [Kernel2] → [Event: Stop]
        ↓
  ...
```

```csharp
// Access timing
var timings = ctx.Debug.LastFrameTiming;
foreach (var (nodeId, time) in timings)
{
    Console.WriteLine($"Node {nodeId}: {time.TotalMilliseconds}ms");
}

// Hierarchical timing
var hierarchy = ctx.Debug.GetHierarchicalTiming();
PrintTiming(hierarchy, indent: 0);
```

---

## Error Handling

```csharp
try
{
    ctx.Compile();
}
catch (GraphValidationException ex)
{
    foreach (var error in ex.Errors)
    {
        Console.WriteLine($"{error.Level}: {error.Message}");
        if (error.BlockId.HasValue)
            Console.WriteLine($"  Block: {error.BlockId}");
    }
}

try
{
    ctx.Execute();
}
catch (CudaException ex)
{
    // GPU runtime error
    Console.WriteLine($"CUDA Error: {ex.CudaError}");
}
```
