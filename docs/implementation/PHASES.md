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
Phase 3: Advanced Features (Regions, Liveness)
    ↓
Phase 4a: Library Calls (Stream Capture)
    ↓
Phase 4b: Patchable Kernels (ILGPU IR + NVRTC escape-hatch)
    ↓
Phase 5: Graphics Interop
```

---

## Phase 0: Foundation — COMPLETE

**Goal**: Basic CUDA infrastructure working in .NET

**Status**: Complete. 25 tests passing. DeviceContext, GpuBuffer, BufferPool, PTX loading, ModuleCache all working.

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 0.1 | Create solution structure | - |
| 0.2 | Add ManagedCuda NuGet dependency, verify CUDA 13.0 compat | 0.1 |
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

## Phase 1: Graph Basics — COMPLETE

**Goal**: Build and execute CUDA Graphs

**Status**: Complete. 29 tests passing. KernelNode, GraphBuilder, GraphCompiler, CompiledGraph, Hot/Warm/Grid updates all working.

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

## Phase 2: Execution Model — COMPLETE

**Goal**: CudaEngine, passive blocks, dirty-tracking

**Status**: Complete. 90 tests passing (148 total with Phase 0+1). All core orchestration working.

> Read `EXECUTION-MODEL.md` for the execution model.

### Implemented

| Task | Description | Status |
|------|-------------|--------|
| 2.1 | ICudaBlock interface (with VL.Core NodeContext) | Done |
| 2.2 | BlockBuilder DSL (AddKernel, Input, Output, InputScalar, Connect, Commit) | Done |
| 2.4 | BlockPort (non-generic), BlockParameter\<T\> with change tracking | Done |
| 2.5 | BlockRegistry with StructureChanged event | Done |
| 2.6 | ConnectionGraph with StructureChanged event | Done |
| 2.7 | DirtyTracker (subscribes to events, IsStructureDirty/AreParametersDirty) | Done |
| 2.8 | CudaEngine (ColdRebuild, Hot Update, Launch, DebugInfo distribution) | Done |
| 2.9 | Debug info distribution (BlockState, StateMessage, LastExecutionTime) | Done |
| 2.12 | Error handling (GPU error → BlockState.Error, no crash) | Done |
| 2.17 | Event-based coupling (Registry/ConnectionGraph → DirtyTracker) | Done |

### Implementation Notes

- Uses VL.Core `NodeContext` (auto-injected by VL). Tests use `RuntimeHelpers.GetUninitializedObject`
- BlockPort is non-generic (not `BlockPort<T>`) — maps to kernel param via internal KernelNodeId + KernelParamIndex
- BlockDescription uses KernelEntry list (ordered, with grid dims) + kernel-index-based internal connections
- CudaEngine.ColdRebuild iterates KernelEntries deterministically (no HashSet ordering issues)
- Parameter change wiring via reflection delegates (contravariant: `OnChanged(object)` matches `Action<BlockParameter<T>>`)
- CompiledGraph.Dispose() does NOT dispose KernelNodes — CudaEngine owns their lifetime

### Deferred (VL Runtime Integration)

These items require a running VL instance and are not blockers for Phase 3+.
They will be addressed when the first real VL patch is built.

| Task | Description | Needs |
|------|-------------|-------|
| 2.3 | InputHandle\<T\>/OutputHandle\<T\> (handle-flow) | VL runtime |
| 2.10 | Simple block example with full VL integration | VL runtime |
| 2.11 | PinGroups support | VL.Core PinGroupKind |
| 2.14 | IVLRuntime diagnostics (node colors) | VL runtime |
| 2.15 | AppHost.TakeOwnership (cleanup on shutdown) | VL runtime |
| 2.16 | ServiceRegistry | VL runtime |
| 2.18 | VL integration tests | VL runtime |

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

## Phase 4a: Library Calls (Stream Capture)

**Goal**: cuBLAS/cuFFT via CapturedNode (static + patchable chained), INodeDescriptor abstraction

> Read `KERNEL-SOURCES.md` before implementing this phase.

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 4a.1 | Define INodeDescriptor, KernelNodeDescriptor, CapturedNodeDescriptor | - |
| 4a.2 | Implement StreamCaptureHelper | - |
| 4a.3 | Implement LibraryHandleCache | - |
| 4a.4 | Implement AddCaptured() in BlockBuilder (static + chained) | 4a.1, 4a.2, 4a.3 |
| 4a.5 | Implement cuBLAS wrapper (Sgemm via Stream Capture) | 4a.4 |
| 4a.6 | Implement cuFFT wrapper (via Stream Capture) | 4a.4 |
| 4a.7 | Graph Compiler Phase 5.5 (Stream Capture for CapturedNodes) | 4a.4 |
| 4a.8 | Dirty-Tracking: Add Recapture level for CapturedNodes | 4a.7 |
| 4a.9 | CudaEngine.UpdateDirtyNodes() dispatch by node type | 4a.8 |
| 4a.10 | Patchable CapturedNode: chained library call sequence | 4a.4 |
| 4a.11 | CapturedNode tests | 4a.10 |

### Deliverables

```csharp
// Should work:

// Static CapturedNode (single library call)
var op = builder.AddCaptured("MatMul", stream =>
{
    cublasSetStream(handle, stream);
    cublasSgemm(handle, ..., stream);
}, descriptor);
// Parameter changes trigger Recapture

// Patchable CapturedNode (chained library calls)
var chain = builder.AddCaptured("MatMulFFT", stream =>
{
    cublasSgemm(handle, ..., A, B, temp1);
    cufftExecC2C(plan, temp1, result, CUFFT_FORWARD);
}, chainDescriptor);
// From outside: one block. Recapture on any param change.

// cuFFT
var fft = new FFTBlock(ctx);
fft.Parameters["Size"].Value = 1024;
```

### Test Cases

1. Stream Capture → ChildGraphNode
2. CapturedNode Recapture on parameter change
3. cuBLAS Sgemm in CUDA Graph via CapturedNode
4. cuFFT forward/inverse via CapturedNode
5. Chained library calls as single CapturedNode
6. Mixed graph: KernelNodes + CapturedNodes in same graph
7. CUBLAS_POINTER_MODE_DEVICE scalar update avoids Recapture

---

## Phase 4b: Patchable Kernels (ILGPU IR + NVRTC)

**Goal**: ILGPU IR for VL-generated patchable kernels, NVRTC as escape-hatch for user CUDA C++

> Read `KERNEL-SOURCES.md` for the ILGPU IR vs NVRTC comparison.

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 4b.1 | Add ILGPU NuGet dependency | - |
| 4b.2 | Implement IlgpuCompiler (IR → PTX → CUmodule, with caching) | 4b.1 |
| 4b.3 | Implement IR construction layer (VL node-set → ILGPU IR) | 4b.2 |
| 4b.4 | Implement AddKernel(ilgpuModule) overload in BlockBuilder | 4b.3 |
| 4b.5 | Implement NvrtcCache (CUDA C++ → PTX, escape-hatch) | - |
| 4b.6 | Implement AddKernel(nvrtcModule) overload in BlockBuilder | 4b.5 |
| 4b.7 | Dirty-Tracking: Add Code level for ILGPU/NVRTC recompile | 4b.4 |
| 4b.8 | Define GPU primitive node-set (GPU.Add, GPU.Mul, GPU.Reduce, etc.) | 4b.3 |
| 4b.9 | Patchable kernel tests | 4b.8 |

### Deliverables

```csharp
// Should work:

// Patchable kernel via ILGPU IR (primary path)
var irDesc = IlgpuIRBuilder.FromNodeSet(nodeSet);  // VL node-set → IR description
var module = ilgpuCompiler.GetOrCompile(irDesc, sm_75);
var kernel = builder.AddKernel(module, "user_kernel");
// Hot/Warm parameter updates work like filesystem PTX
// Code update: ILGPU recompile ~1-10ms → Cold rebuild of affected block

// User CUDA C++ via NVRTC (escape-hatch)
string userCode = File.ReadAllText("my_kernel.cu");
var module = nvrtcCache.GetOrCompile(userCode, "my_kernel", sm_75);
var kernel = builder.AddKernel(module, "my_kernel");
```

### Test Cases

1. ILGPU IR construction from simple node-set (Add, Mul)
2. ILGPU compile → PTX string → CUmodule via ManagedCuda
3. IlgpuCompiler cache deduplication
4. Patchable kernel Hot/Warm update (same as filesystem PTX)
5. Patchable kernel Code update (node-set change → ILGPU recompile → Cold rebuild)
6. NVRTC compile CUDA C++ string → CUmodule (escape-hatch)
7. NvrtcCache deduplication (same source → cached module)
8. Mixed graph: Filesystem PTX + ILGPU patchable + CapturedNodes

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
    └── ManagedCuda (NuGet) — CUDA 13.0 minimum
    └── ILGPU (NuGet) — PTX compilation only, no accelerator
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
| ManagedCuda NuGet vs CUDA 13.0 | Verify latest NuGet compat in Phase 0, fall back to source-build |
| ILGPU IR as internal API | Pin ILGPU NuGet version, fork if breaking changes |
| Conditional nodes (CUDA 12.4+) | CUDA 13.0 minimum guarantees support |
| DX11 interop complexity | Prototype early with simple buffer |
| VL PinGroups integration | Test with VL team early |
| Performance (graph rebuild) | Profile Cold Rebuild cost, add stream fallback only if needed |
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
    ├── PTX/
    │   ├── PTXParserTests.cs
    │   └── ModuleCacheTests.cs
    ├── Patchable/
    │   ├── IlgpuCompilerTests.cs
    │   ├── IlgpuIRConstructionTests.cs
    │   ├── NvrtcCacheTests.cs
    │   └── NvrtcCompilationTests.cs
    └── Captured/
        ├── StreamCaptureHelperTests.cs
        ├── CapturedNodeTests.cs
        ├── LibraryHandleCacheTests.cs
        └── RecaptureTests.cs
```

### Integration Tests

```
VL.Cuda.Integration.Tests/
    ├── EndToEnd/
    │   ├── SimpleKernelChainTests.cs
    │   ├── ConditionalGraphTests.cs
    │   ├── ParticleSystemTests.cs
    │   └── HotSwapSimulationTests.cs
    ├── VL/
    │   ├── PinGroupsTests.cs
    │   └── HandleFlowTests.cs
    └── MixedGraph/
        ├── KernelAndCapturedNodeTests.cs
        ├── IlgpuPatchableKernelTests.cs
        ├── ChainedCapturedNodeTests.cs
        └── CuBlasCapturedNodeTests.cs
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

| Phase | Dependencies |
|-------|--------------|
| Phase 0 | - |
| Phase 1 | Phase 0 |
| Phase 2 | Phase 1 |
| Phase 3 | Phase 2 |
| Phase 4a | Phase 2 (can overlap Phase 3) |
| Phase 4b | Phase 2 (can overlap Phase 3/4a) |
| Phase 5 | Phase 2 (can start early) |

Parallel work possible:
- PTX tooling and example kernels (Triton, nvcc, etc.) can be developed alongside
- Documentation can be refined throughout
- DX11 interop can start during Phase 2

---

## Success Criteria

### Phase 0 Complete
- [x] Can create DeviceContext (wraps ManagedCuda.CudaContext)
- [x] Can allocate/free GPU buffers (GpuBuffer\<T\>)
- [x] Can load PTX and execute kernels
- [x] BufferPool with power-of-2 bucketing
- [x] ModuleCache for PTX/module caching
- [x] 25 tests passing

### Phase 1 Complete
- [x] Can build CUDA Graph from description (GraphBuilder)
- [x] Graph executes correctly (GraphCompiler → CompiledGraph)
- [x] Hot Update: scalar value change without rebuild
- [x] Warm Update: buffer pointer change without rebuild
- [x] Grid update without rebuild
- [x] Validation catches errors (cycles, param conflicts)
- [x] 29 tests passing

### Phase 2 Complete
- [x] CudaEngine compiles and launches graph each frame
- [x] Blocks register/unregister via CudaContext
- [x] Dirty-tracking correctly triggers Cold Rebuild / Hot Update
- [x] Debug info flows from Engine to Blocks (BlockState, LastExecutionTime)
- [x] GPU errors don't crash — BlockState.Error with message
- [x] Event-based coupling (BlockRegistry/ConnectionGraph → DirtyTracker)
- [x] BlockBuilder DSL with deterministic kernel ordering
- [x] BlockDescription with structural equality for hot-swap detection
- [x] 90 tests passing (148 total)
- [x] NodeContext in constructors (VL.Core NuGet added)

Phase 2 VL Runtime Integration (deferred — done when first VL patch is built):
- [ ] PinGroups display (needs VL.Core PinGroupKind)
- [ ] Hot-Swap simulation (needs VL runtime)
- [ ] IVLRuntime diagnostics (needs VL runtime)
- [ ] AppHost.TakeOwnership (needs VL runtime)

### Phase 3 Complete
- [ ] AppendBuffer works with GPU counter
- [ ] Conditional regions execute correctly
- [ ] Composite blocks compose properly
- [ ] Serialization round-trips

### Phase 4a Complete
- [ ] INodeDescriptor abstraction with KernelNodeDescriptor and CapturedNodeDescriptor
- [ ] StreamCaptureHelper captures library calls into child graphs
- [ ] LibraryHandleCache caches cuBLAS/cuFFT handles
- [ ] AddCaptured() works in BlockBuilder (static + chained)
- [ ] cuBLAS Sgemm works in CUDA Graph via CapturedNode
- [ ] cuFFT forward/inverse works via CapturedNode
- [ ] Patchable CapturedNode: chained library calls as single block
- [ ] Graph Compiler Phase 5.5 (Stream Capture) runs for CapturedNodes only
- [ ] Dirty-Tracking: Recapture level triggers Recapture for CapturedNodes
- [ ] CudaEngine.UpdateDirtyNodes() dispatches correctly by node type
- [ ] Mixed graph (KernelNodes + CapturedNodes) compiles and executes
- [ ] CUBLAS_POINTER_MODE_DEVICE scalar update avoids Recapture

### Phase 4b Complete
- [ ] ILGPU NuGet added, PTXBackend accessible
- [ ] IlgpuCompiler builds IR → PTX → CUmodule with caching
- [ ] IR construction layer maps VL node-set to ILGPU IR operations
- [ ] AddKernel(ilgpuModule) overload works in BlockBuilder
- [ ] GPU primitive node-set defined (GPU.Add, GPU.Mul, etc.)
- [ ] Dirty-Tracking: Code level triggers ILGPU recompile → Cold rebuild
- [ ] NvrtcCache compiles user CUDA C++ strings (escape-hatch)
- [ ] AddKernel(nvrtcModule) overload works (escape-hatch)
- [ ] Mixed graph: Filesystem PTX + ILGPU patchable + CapturedNodes all work together

### Phase 5 Complete
- [ ] DX11 buffer sharing works
- [ ] DX11 texture sharing works
- [ ] Can render CUDA output in Stride
- [ ] Full pipeline: CUDA compute → Stride render
