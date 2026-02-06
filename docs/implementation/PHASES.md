# Implementation Phases

## Overview

The implementation is divided into phases, each building on the previous. Each phase results in a working, testable system.

```
Phase 0: Foundation
    ↓
Phase 1: Graph Basics
    ↓
Phase 2: VL Integration
    ↓
Phase 3: Advanced Features
    ↓
Phase 4: Libraries & Interop
```

---

## Phase 0: Foundation

**Goal**: Basic CUDA infrastructure working in .NET

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 0.1 | Create solution structure | - |
| 0.2 | Add ManagedCuda dependency | 0.1 |
| 0.3 | Implement CudaContext wrapper | 0.2 |
| 0.4 | Implement CudaStream wrapper | 0.3 |
| 0.5 | Implement GpuBuffer<T> | 0.3 |
| 0.6 | Implement BufferPool | 0.5 |
| 0.7 | Implement PTX loading | 0.3 |
| 0.8 | Implement ModuleCache | 0.7 |
| 0.9 | Basic unit tests | 0.8 |

### Deliverables

```csharp
// Should work:
using var ctx = CudaContext.Create();
var buffer = ctx.Pool.Acquire<float>(1024, BufferLifetime.External);
buffer.Upload(data);
// Load and run a PTX kernel manually
```

### Test Cases

1. Create context on device 0
2. Allocate buffer, upload, download, verify
3. Pool acquire/release cycle
4. Load PTX module
5. Execute simple kernel (vector add)

---

## Phase 1: Graph Basics

**Goal**: Build and execute CUDA Graphs

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 1.1 | Define NodeDesc, EdgeDesc | - |
| 1.2 | Define GraphDescription | 1.1 |
| 1.3 | Implement validation | 1.2 |
| 1.4 | Implement topological sort | 1.2 |
| 1.5 | Implement CUDA Graph building | 1.4 |
| 1.6 | Implement CompiledGraph | 1.5 |
| 1.7 | Implement parameter update | 1.6 |
| 1.8 | Graph execution tests | 1.7 |

### Deliverables

```csharp
// Should work:
var desc = new GraphDescription();
desc.AddNode(kernel1);
desc.AddNode(kernel2);
desc.AddEdge(kernel1.Out(0), kernel2.In(0));

var compiled = GraphCompiler.Compile(desc, ctx);
compiled.Launch();
```

### Test Cases

1. Single kernel graph
2. Sequential kernel chain
3. Parallel kernels (diamond pattern)
4. Parameter update without rebuild
5. Cycle detection (should fail)
6. Type mismatch detection (should fail)

---

## Phase 2: VL Integration

**Goal**: Blocks and handles work with VL patterns

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 2.1 | Define ICudaBlock interface | - |
| 2.2 | Implement BlockBuilder | 2.1 |
| 2.3 | Implement InputHandle/OutputHandle | 2.1 |
| 2.4 | Implement BlockPort, BlockParameter | 2.1 |
| 2.5 | Implement simple block (kernels only) | 2.2 |
| 2.6 | Integrate with CudaContext | 2.5 |
| 2.7 | Implement PinGroups support | 2.5 |
| 2.8 | VL integration tests | 2.7 |

### Deliverables

```csharp
// Should work:
public class MyBlock : ICudaBlock
{
    public void Setup(BlockBuilder builder)
    {
        var k = builder.AddKernel("kernel.ptx", "entry");
        builder.Input<float>("In", k.In(0));
        builder.Output<float>("Out", k.Out(1));
        builder.Commit();
    }
}

var block = ctx.CreateBlock<MyBlock>();
```

### Test Cases

1. Simple block with one kernel
2. Block with multiple kernels
3. Scalar parameters
4. PinGroups generation
5. Handle flow through mock VL links

---

## Phase 3: Advanced Features

**Goal**: AppendBuffer, regions, liveness analysis

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 3.1 | Implement AppendBuffer<T> | - |
| 3.2 | Implement If region | 3.1 |
| 3.3 | Implement While region | 3.2 |
| 3.4 | Implement For region | 3.2 |
| 3.5 | Implement liveness analysis | 3.4 |
| 3.6 | Implement shape propagation | 3.4 |
| 3.7 | Implement composite blocks | 3.4 |
| 3.8 | Implement serialization | 3.7 |
| 3.9 | Advanced tests | 3.8 |

### Deliverables

```csharp
// Should work:
var append = ctx.Pool.AcquireAppend<Particle>(100000, BufferLifetime.Graph);

// Composite block
public class ParticleSystem : ICudaBlock
{
    public void Setup(BlockBuilder builder)
    {
        var emitter = builder.AddChild<EmitterBlock>();
        var forces = builder.AddChild<ForcesBlock>();
        builder.ConnectChildren(emitter, "Out", forces, "In");
        builder.ExposeOutput("Particles", forces, "Out");
    }
}

// Serialization
var model = ctx.GetModel();
model.SaveToFile("system.json");
```

### Test Cases

1. AppendBuffer with GPU counter
2. If region with condition kernel
3. While loop termination
4. Buffer reuse across regions
5. Composite block composition
6. Save/load graph model

---

## Phase 4: Libraries & Interop

**Goal**: cuBLAS, cuFFT, DX11 sharing

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 4.1 | Implement library handle management | - |
| 4.2 | Implement cuFFT wrapper | 4.1 |
| 4.3 | Implement cuBLAS wrapper | 4.1 |
| 4.4 | Implement CaptureToGraph for libraries | 4.3 |
| 4.5 | Implement DX11 interop | - |
| 4.6 | Implement SharedBuffer<T> | 4.5 |
| 4.7 | Implement SharedTexture | 4.5 |
| 4.8 | VL.Stride integration tests | 4.7 |

### Deliverables

```csharp
// Should work:
// cuFFT
var fft = ctx.CreateBlock<FFTBlock>();
fft.Parameters["Size"] = 1024;

// DX11 interop
var shared = CudaDX11Interop.RegisterBuffer<Particle>(
    strideBuffer.NativePointer, ctx);
shared.MapForCuda();
// Use in kernel
shared.UnmapFromCuda();
```

### Test Cases

1. cuFFT forward/inverse
2. cuBLAS GEMM
3. Library in CUDA Graph (CaptureToGraph)
4. DX11 buffer sharing
5. DX11 texture sharing
6. Full pipeline: CUDA compute → Stride render

---

## Dependencies

```
VL.Cuda.Core
    └── ManagedCuda (NuGet)
    └── VL.Core (later, for PinGroups)

VL.Cuda.Libraries
    └── VL.Cuda.Core
    └── ManagedCuda.CUFFT
    └── ManagedCuda.CUBLAS

VL.Cuda.Stride
    └── VL.Cuda.Core
    └── VL.Stride
```

---

## Risk Areas

| Risk | Mitigation |
|------|------------|
| ManagedCuda CUDA Graph support | Verify API coverage early, may need to extend |
| Conditional nodes (CUDA 12.4+) | Require minimum CUDA version |
| DX11 interop complexity | Prototype early with simple buffer |
| VL PinGroups integration | Test with VL team early |
| Performance (graph rebuild) | Profile rebuild vs update paths |

---

## Testing Strategy

### Unit Tests

```
VL.Cuda.Core.Tests/
    ├── Context/
    │   ├── CudaContextTests.cs
    │   └── CudaStreamTests.cs
    ├── Buffers/
    │   ├── GpuBufferTests.cs
    │   ├── AppendBufferTests.cs
    │   └── BufferPoolTests.cs
    ├── Graph/
    │   ├── GraphCompilerTests.cs
    │   ├── ValidationTests.cs
    │   └── TopologicalSortTests.cs
    ├── Blocks/
    │   ├── BlockBuilderTests.cs
    │   └── CompositeBlockTests.cs
    └── PTX/
        ├── PTXParserTests.cs
        └── ModuleCacheTests.cs
```

### Integration Tests

```
VL.Cuda.Integration.Tests/
    ├── EndToEnd/
    │   ├── SimpleKernelChainTests.cs
    │   ├── ConditionalGraphTests.cs
    │   └── ParticleSystemTests.cs
    └── VL/
        ├── PinGroupsTests.cs
        └── HandleFlowTests.cs
```

### Performance Tests

```
VL.Cuda.Benchmarks/
    ├── GraphRebuildBenchmark.cs
    ├── BufferPoolBenchmark.cs
    └── KernelLaunchBenchmark.cs
```

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 0 | 2 weeks | - |
| Phase 1 | 3 weeks | Phase 0 |
| Phase 2 | 3 weeks | Phase 1 |
| Phase 3 | 4 weeks | Phase 2 |
| Phase 4 | 4 weeks | Phase 3 |

**Total**: ~16 weeks (4 months) for full implementation

Parallel work possible:
- PTX tooling (Python/Triton) can be developed alongside
- Documentation can be refined throughout
- DX11 interop can start during Phase 2

---

## Success Criteria

### Phase 0 Complete
- [ ] Can create CudaContext
- [ ] Can allocate/free GPU buffers
- [ ] Can load PTX and execute kernels
- [ ] Basic tests passing

### Phase 1 Complete
- [ ] Can build CUDA Graph from description
- [ ] Graph executes correctly
- [ ] Parameter updates work
- [ ] Validation catches errors

### Phase 2 Complete
- [ ] ICudaBlock implementation works
- [ ] BlockBuilder generates correct graph
- [ ] Handle flow through VL links works
- [ ] PinGroups display correctly

### Phase 3 Complete
- [ ] AppendBuffer works with GPU counter
- [ ] Conditional regions execute correctly
- [ ] Composite blocks compose properly
- [ ] Serialization round-trips

### Phase 4 Complete
- [ ] cuFFT/cuBLAS work in graph
- [ ] DX11 buffer sharing works
- [ ] Can render CUDA output in Stride
- [ ] Performance acceptable
