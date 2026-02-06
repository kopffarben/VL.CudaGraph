# Implementation Phases

## Overview

The implementation is divided into phases, each building on the previous. Each phase results in a working, testable system.

```
Phase 0: Foundation
    ↓
Phase 1: Graph Basics
    ↓
Phase 2: Execution Model & VL Integration
    ↓
Phase 3: Advanced Features
    ↓
Phase 4: Patchable Kernels & Libraries
    ↓
Phase 5: Graphics Interop
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
| 1.7 | Implement parameter update (Hot/Warm) | 1.6 |
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

// Hot Update (no rebuild):
compiled.SetScalar(nodeId, paramIndex, newValue);
compiled.Launch();

// Warm Update (no rebuild):
compiled.UpdatePointer(nodeId, paramIndex, newPointer);
compiled.Launch();
```

### Test Cases

1. Single kernel graph
2. Sequential kernel chain
3. Parallel kernels (diamond pattern)
4. Hot Update: scalar value change without rebuild
5. Warm Update: buffer pointer change without rebuild
6. Cycle detection (should fail)
7. Type mismatch detection (should fail)

---

## Phase 2: Execution Model & VL Integration

**Goal**: CudaEngine, passive blocks, dirty-tracking, VL integration

> Read `EXECUTION-MODEL.md` before implementing this phase.

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 2.1 | Define ICudaBlock interface (with IDisposable) | - |
| 2.2 | Implement BlockBuilder (used in constructor) | 2.1 |
| 2.3 | Implement InputHandle/OutputHandle | 2.1 |
| 2.4 | Implement BlockPort, BlockParameter | 2.1 |
| 2.5 | Implement CudaContext block registry | 2.1 |
| 2.6 | Implement CudaContext connection tracking | 2.5 |
| 2.7 | Implement dirty-tracking (structure + parameters) | 2.6 |
| 2.8 | Implement CudaEngine (active ProcessNode) | 2.7 |
| 2.9 | Implement debug info distribution (Engine → Blocks) | 2.8 |
| 2.10 | Implement simple block (kernels only) | 2.2 |
| 2.11 | Implement PinGroups support | 2.10 |
| 2.12 | Implement error handling (no crash on GPU error) | 2.8 |
| 2.13 | VL.Core: NodeContext in all constructors | 2.1 |
| 2.14 | VL.Core: IVLRuntime diagnostics (AddMessage, AddPersistentMessage) | 2.8 |
| 2.15 | VL.Core: AppHost.TakeOwnership for CudaEngine | 2.8 |
| 2.16 | VL.Core: ServiceRegistry for global singletons (DeviceInfo, DriverVersion) | 0.3 |
| 2.17 | VL.Core: Event-based coupling (Registry/Topology → DirtyTracker) | 2.7 |
| 2.18 | VL integration tests | 2.17 |

### Deliverables

```csharp
// Should work:

// Engine creates context, exposes it as output pin
var engine = new CudaEngine();
var ctx = engine.Context;

// Blocks register in constructor (passive)
var emitter = new EmitterBlock(ctx);
var forces = new ForcesBlock(ctx);

// VL draws links → Connect calls
ctx.Connect(emitter.Id, "Particles", forces.Id, "Particles");

// Engine update (called every frame by VL)
engine.Update();
// → Detects structureDirty (new blocks + connections)
// → Cold Rebuild
// → Launch
// → Distributes debug info

// Next frame: parameter change (no rebuild)
forces.Parameters["Gravity"].Value = 5.0f;
engine.Update();
// → Detects parametersDirty
// → Hot Update only
// → Launch

// Block shows debug tooltip in VL
var info = emitter.DebugInfo;
// info.LastExecutionTime = 0.42ms
// info.State = BlockState.OK
```

### Test Cases

1. Block registers on construction, unregisters on Dispose
2. Structure dirty flag set on RegisterBlock/UnregisterBlock/Connect/Disconnect
3. Parameter dirty flag set on parameter value change
4. CudaEngine Cold Rebuild on structure change
5. CudaEngine Hot Update on scalar change
6. CudaEngine Warm Update on buffer pointer change
7. Debug info distributed to blocks after launch
8. GPU error doesn't crash — sets BlockState.Error
9. PinGroups display correctly
10. Handle flow through mock VL links

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

// Composite block (children managed via BlockBuilder)
var particleSystem = new ParticleSystemBlock(ctx);
// ParticleSystemBlock internally creates Emitter, Forces, Integrate as children

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

## Phase 4: Patchable Kernels & Libraries

**Goal**: NVRTC patchable kernels, cuBLAS/cuFFT via CapturedNode, INodeDescriptor abstraction

> Read `KERNEL-SOURCES.md` before implementing this phase.

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 4.1 | Define INodeDescriptor, KernelNodeDescriptor, CapturedNodeDescriptor | - |
| 4.2 | Implement NvrtcCache (CUDA C++ → PTX/cubin compilation caching) | - |
| 4.3 | Implement AddKernel(CUmodule) overload in BlockBuilder | 4.1, 4.2 |
| 4.4 | Implement patchable kernel codegen (node-set → CUDA C++) | 4.3 |
| 4.5 | Implement StreamCaptureHelper | - |
| 4.6 | Implement LibraryHandleCache | - |
| 4.7 | Implement AddCaptured() in BlockBuilder | 4.1, 4.5, 4.6 |
| 4.8 | Implement cuBLAS wrapper (Sgemm via Stream Capture) | 4.7 |
| 4.9 | Implement cuFFT wrapper (via Stream Capture) | 4.7 |
| 4.10 | Graph Compiler Phase 5.5 (Stream Capture for CapturedNodes) | 4.7 |
| 4.11 | Dirty-Tracking: Add Recapture level for CapturedNodes | 4.10 |
| 4.12 | Dirty-Tracking: Add Code level for NVRTC recompile | 4.4 |
| 4.13 | CudaEngine.UpdateDirtyNodes() dispatch by node type | 4.11, 4.12 |
| 4.14 | Patchable kernel + CapturedNode tests | 4.13 |

### Deliverables

```csharp
// Should work:

// Patchable kernel (NVRTC)
string cudaSource = PatchableCodegen.Generate(nodeSet);
var module = nvrtcCache.GetOrCompile(cudaSource, "user_kernel", sm75);
var kernel = builder.AddKernel(module, "user_kernel");
// Hot/Warm parameter updates work like filesystem PTX

// Library operation (CapturedNode)
var op = builder.AddCaptured("MatMul", stream =>
{
    cublasSetStream(handle, stream);
    cublasSgemm(handle, ..., stream);
}, descriptor);
// Parameter changes trigger Recapture

// cuFFT
var fft = new FFTBlock(ctx);
fft.Parameters["Size"].Value = 1024;
```

### Test Cases

1. NVRTC compile CUDA C++ string → CUmodule
2. NvrtcCache deduplication (same source → cached module)
3. Patchable kernel Hot/Warm update (same as filesystem PTX)
4. Patchable kernel Code update (node-set change → recompile → Cold rebuild)
5. Stream Capture → ChildGraphNode
6. CapturedNode Recapture on parameter change
7. cuBLAS Sgemm in CUDA Graph via CapturedNode
8. cuFFT forward/inverse via CapturedNode
9. Mixed graph: KernelNodes + CapturedNodes in same graph
10. CUBLAS_POINTER_MODE_DEVICE scalar update avoids re-capture

---

## Phase 5: Graphics Interop

**Goal**: DX11/Stride sharing

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 5.1 | Implement DX11 interop | - |
| 5.2 | Implement SharedBuffer<T> | 5.1 |
| 5.3 | Implement SharedTexture | 5.1 |
| 5.4 | VL.Stride integration tests | 5.3 |

### Deliverables

```csharp
// Should work:
var shared = CudaDX11Interop.RegisterBuffer<Particle>(
    strideBuffer.NativePointer, ctx);
shared.MapForCuda();
// Use in kernel
shared.UnmapFromCuda();
```

### Test Cases

1. DX11 buffer sharing
2. DX11 texture sharing
3. Full pipeline: CUDA compute → Stride render

---

## Dependencies

```
VL.Cuda.Core
    └── ManagedCuda (NuGet)
    └── VL.Core (NodeContext, IVLRuntime, AppHost, ResourceProvider, PinGroups)

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
| Conditional nodes (CUDA 12.8) | Hard minimum — no fallback paths, verify ManagedCuda support |
| DX11 interop complexity | Prototype early with simple buffer |
| VL PinGroups integration | Test with VL team early |
| Performance (graph rebuild) | Profile Cold Rebuild cost at realistic graph sizes |
| Hot-Swap correctness | Test Dispose → new Constructor → reconnect cycle |

---

## Testing Strategy

### Unit Tests

```
VL.Cuda.Core.Tests/
    ├── Context/
    │   ├── CudaContextTests.cs
    │   ├── CudaStreamTests.cs
    │   └── DirtyTrackingTests.cs
    ├── Buffers/
    │   ├── GpuBufferTests.cs
    │   ├── AppendBufferTests.cs
    │   └── BufferPoolTests.cs
    ├── Graph/
    │   ├── GraphCompilerTests.cs
    │   ├── ValidationTests.cs
    │   └── TopologicalSortTests.cs
    ├── Engine/
    │   ├── CudaEngineTests.cs
    │   ├── ColdRebuildTests.cs
    │   ├── HotUpdateTests.cs
    │   └── WarmUpdateTests.cs
    ├── Blocks/
    │   ├── BlockBuilderTests.cs
    │   ├── BlockLifecycleTests.cs
    │   ├── CompositeBlockTests.cs
    │   └── DebugInfoTests.cs
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
    │   ├── ParticleSystemTests.cs
    │   └── HotSwapSimulationTests.cs
    └── VL/
        ├── PinGroupsTests.cs
        └── HandleFlowTests.cs
```

### Performance Tests

```
VL.Cuda.Benchmarks/
    ├── ColdRebuildBenchmark.cs     ← Critical: measure rebuild at various graph sizes
    ├── HotUpdateBenchmark.cs
    ├── BufferPoolBenchmark.cs
    └── KernelLaunchBenchmark.cs
```

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 0 | 2 weeks | - |
| Phase 1 | 3 weeks | Phase 0 |
| Phase 2 | 4 weeks | Phase 1 |
| Phase 3 | 4 weeks | Phase 2 |
| Phase 4 | 4 weeks | Phase 3 |

**Total**: ~17 weeks for full implementation

Parallel work possible:
- PTX tooling and example kernels (Triton, nvcc, etc.) can be developed alongside
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
- [ ] Hot/Warm parameter updates work without rebuild
- [ ] Validation catches errors

### Phase 2 Complete
- [ ] CudaEngine compiles and launches graph each frame
- [ ] Blocks register/unregister via constructor/Dispose
- [ ] Dirty-tracking correctly triggers Cold/Warm/Hot updates
- [ ] Debug info flows from Engine to Blocks (tooltips)
- [ ] GPU errors don't crash — graceful degradation
- [ ] PinGroups display correctly
- [ ] Hot-Swap simulation works (Dispose old → create new → reconnect)
- [ ] NodeContext flows through all constructors
- [ ] IVLRuntime routes errors/warnings to correct VL nodes
- [ ] AppHost.TakeOwnership ensures cleanup on app shutdown
- [ ] Event-based coupling between Registry/Topology and DirtyTracker

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
