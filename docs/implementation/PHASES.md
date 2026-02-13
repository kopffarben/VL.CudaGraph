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
    ↓
Phase 6: Device Graph Launch & Dispatch Indirect
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

## VL Integration Gate — PASSED

**Status**: All 6 manual tests passed in vvvv gamma 7.1.

### Validated

| Test | Result |
|------|--------|
| Package Discovery | CudaEngine found in node browser |
| Node Creation | Node placed without error, CUDA context created on VL's thread |
| Update Loop | Update() called every frame, no crashes |
| Tooltip/DebugInfo | ToString() shows "CudaEngine: 0 blocks, not compiled" |
| Dispose | Node deletion cleans up without crash or GPU leaks |
| State Output | State output pin visible (HasStateOutput = true) |

### Source-Package Setup (validated)

| File | Purpose |
|------|---------|
| `deployment/VL.CudaGraph.nuspec` | Package metadata (for `nuget pack`) |
| `VL.CudaGraph.vl` | VL document with dependencies |
| `lib/net8.0/VL.Cuda.Core.dll` | Built DLL (OutputPath convention) |
| `lib/net8.0/VL.Cuda.Core.xml` | XML docs for VL tooltips |

Key learnings:
- `.nuspec` goes in `deployment/` folder (VL template convention)
- `.vl` uses `PlatformDependency` with `IsForward="true"` (NOT `AssemblyReference`)
- `.vl` needs explicit `NugetDependency` for ManagedCuda-13 (not just in nuspec)
- Don't hand-edit `.vl` files — VL regenerates them with new IDs on open
- `deps.json` alone is NOT enough for dependency resolution — VL resolves from `.vl` file

### Not Yet Tested (deferred to first block implementation)

- Hot-Swap (needs block code change)
- Handle-flow on links (needs block consuming CudaContext)
- IVLRuntime diagnostics (needs error scenario)
- PinGroups (needs block with dynamic pins)
- AppHost.TakeOwnership (needs app shutdown test)

---

## Phase 3: Advanced Features

**Goal**: AppendBuffer, regions, liveness analysis

### Phase 3.1: AppendBuffer — COMPLETE

**Status**: Complete. 19 tests passing (167 total with Phase 0+1+2).

#### Implemented

| Task | Description | Status |
|------|-------------|--------|
| 3.1a | AppendBuffer\<T\> (wraps GpuBuffer\<T\> Data + GpuBuffer\<uint\> Counter) | Done |
| 3.1b | IAppendBuffer interface (type-erased tracking for GraphCompiler/CudaEngine) | Done |
| 3.1c | BufferPool.AcquireAppend\<T\>(maxCapacity, lifetime) | Done |
| 3.1d | MemsetDescriptor + GraphCompiler memset node phase (counter reset before kernel execution) | Done |
| 3.1e | BlockBuilder.AppendOutput\<T\>() with auto-generated "{name} Count" port | Done |
| 3.1f | AppendBufferInfo in BlockDescription | Done |
| 3.1g | CudaEngine integration: memset dependency wiring in graph build | Done |
| 3.1h | CudaEngine auto-readback of append counts after launch | Done |
| 3.1i | BlockDebugInfo.AppendCounts (Dictionary\<string, uint\>) | Done |

### Remaining Tasks (Phase 3.2+)

| Task | Description | Depends On |
|------|-------------|------------|
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
// Should work (Phase 3.1 — implemented):
var append = ctx.Pool.AcquireAppend<Particle>(100000, BufferLifetime.Graph);
// Counter auto-reset via memset node before each graph launch
// Count auto-readback into BlockDebugInfo.AppendCounts after launch

// Phase 3.2+ (not yet implemented):
// Composite block (children managed via BlockBuilder)
var particleSystem = new ParticleSystemBlock(ctx);

// Serialization
var model = ctx.GetModel();
model.SaveToFile("system.json");
```

### Test Cases

1. ~~AppendBuffer with GPU counter~~ (Phase 3.1 — done)
2. If region with condition kernel
3. While loop termination
4. Buffer reuse across regions
5. Composite block composition
6. Save/load graph model

---

## Phase 4a: Library Calls (Stream Capture) — COMPLETE

**Goal**: cuBLAS/cuFFT/cuSPARSE/cuRAND/cuSOLVER via CapturedNode (static + patchable chained), IGraphNode abstraction

**Status**: Complete. 44 tests passing (211 total with Phase 0-3).

> See `KERNEL-SOURCES.md` for the full CapturedNode design.

### Implemented

| Task | Description | Status |
|------|-------------|--------|
| 4a.1 | IGraphNode interface (polymorphic: KernelNode + CapturedNode) | Done |
| 4a.2 | CapturedNode (stream capture → CUgraph, owns child graph lifetime) | Done |
| 4a.3 | CapturedNodeDescriptor (Inputs/Outputs/Scalars as CapturedParam lists) | Done |
| 4a.4 | StreamCaptureHelper (CaptureToGraph, DestroyGraph, AddChildGraphNode, UpdateChildGraphNode) | Done |
| 4a.5 | LibraryHandleCache (cuBLAS, cuFFT, cuSPARSE, cuRAND, cuSOLVER) | Done |
| 4a.6 | BlockBuilder.AddCaptured(Action\<CUstream, CUdeviceptr[]\>, descriptor) → CapturedHandle | Done |
| 4a.7 | CapturedHandle.In()/Out()/Scalar() → CapturedPin (flat buffer binding index) | Done |
| 4a.8 | BlockBuilder Input\<T\>/Output\<T\> overloads for CapturedPin | Done |
| 4a.9 | CapturedEntry in BlockDescription.CapturedEntries | Done |
| 4a.10 | GraphCompiler Phase 5b: capture on dedicated stream → AddChildGraphNode | Done |
| 4a.11 | GraphCompiler Phase 5c: kernel node deps on captured node handles (CapturedDependency) | Done |
| 4a.12 | CompiledGraph.capturedNodeHandles + RecaptureNode() | Done |
| 4a.13 | DirtyTracker: AreCapturedNodesDirty, MarkCapturedNodeDirty(), ClearCapturedNodesDirty() | Done |
| 4a.14 | DirtyCapturedNode record struct (BlockId, CapturedHandleId) | Done |
| 4a.15 | CudaContext.OnCapturedNodeChanged() → DirtyTracker | Done |
| 4a.16 | CudaEngine: Structure > Recapture > Parameters priority in Update() | Done |
| 4a.17 | CudaEngine: RecaptureNodes() — re-stream-capture + UpdateChildGraphNode | Done |
| 4a.18 | CudaEngine: CapturedNode creation in ColdRebuild via BuildBlockNodes() | Done |
| 4a.19 | BlasOperations (Sgemm, Dgemm, Sgemv, Sscal) | Done |
| 4a.20 | FftOperations (Forward1D, Inverse1D, R2C1D, C2R1D) | Done |
| 4a.21 | SparseOperations (SpMV) | Done |
| 4a.22 | RandOperations (GenerateUniform, GenerateNormal) | Done |
| 4a.23 | SolveOperations (Sgetrf, Sgetrs) | Done |

### Implementation Notes

- `IGraphNode` interface (Id, DebugName) provides polymorphic handling — both `KernelNode` and `CapturedNode` implement it
- `CapturedNode.Capture(stream)` destroys previous captured graph before re-capturing (prevents CUgraph leaks)
- Buffer bindings layout: `[inputs..., outputs..., scalars...]` — flat array matching descriptor order
- `CapturedHandle.In(i)` returns flat index `i`, `Out(i)` returns `Inputs.Count + i`, `Scalar(i)` returns `Inputs.Count + Outputs.Count + i`
- `CapturedDependency(SourceNodeId, TargetNodeId)` allows cross-type dependencies: kernel→captured and captured→kernel
- GraphCompiler creates a dedicated `CudaStream` for capture (separate from execution stream)
- `_capturedHandleToNodeId` dict in CudaEngine maps `(BlockId, HandleId)` → `CapturedNode.Id` for Recapture routing
- Library wrappers are static classes that take `BlockBuilder` + `LibraryHandleCache` and return `CapturedHandle`
- `BlockDescription.StructuralEquals()` includes `CapturedEntries` comparison (debug name + param counts)

### Deliverables

```csharp
// Should work (and does):

// Static CapturedNode via low-level API
var descriptor = new CapturedNodeDescriptor("cuBLAS.Sgemm",
    inputs: new[] { CapturedParam.Pointer("A", "float*"), CapturedParam.Pointer("B", "float*") },
    outputs: new[] { CapturedParam.Pointer("C", "float*") });

var op = builder.AddCaptured((stream, buffers) =>
{
    var blas = libs.GetOrCreateBlas();
    blas.Stream = stream;
    CudaBlasNativeMethods.cublasSgemm_v2(blas.CublasHandle, ...,
        buffers[0], ..., buffers[1], ..., buffers[2], ...);
}, descriptor);
builder.Input<float>("A", op.In(0));
builder.Output<float>("C", op.Out(0));

// Static CapturedNode via high-level wrapper
var sgemm = BlasOperations.Sgemm(builder, libs, m: 128, n: 128, k: 128);
builder.Input<float>("A", sgemm.In(0));
builder.Input<float>("B", sgemm.In(1));
builder.Output<float>("C", sgemm.Out(0));

// Patchable CapturedNode (chained library calls)
var chain = builder.AddCaptured((stream, buffers) =>
{
    var blas = libs.GetOrCreateBlas(); blas.Stream = stream;
    CudaBlasNativeMethods.cublasSgemm_v2(blas.CublasHandle, ..., buffers[0], buffers[1], temp);
    var plan = libs.GetOrCreateFFT1D(nx, cufftType.C2C);
    CudaFFTNativeMethods.cufftSetStream(plan.Handle, stream);
    plan.Exec(temp, buffers[2], TransformDirection.Forward);
}, chainDescriptor);

// CudaEngine dispatches: Structure > Recapture > Parameters
// RecaptureNodes() re-captures and updates child graph in-place
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

## Phase 4b: Patchable Kernels (ILGPU IR + NVRTC) — COMPLETE

**Goal**: ILGPU IR for VL-generated patchable kernels, NVRTC as escape-hatch for user CUDA C++

**Status**: Complete. 37 tests passing (248 total with Phase 0-4a). 9 tests skipped (require GPU). 257 total.

> See `KERNEL-SOURCES.md` for the ILGPU IR vs NVRTC comparison and full implementation details.

### Implemented

| Task | Description | Status |
|------|-------------|--------|
| 4b.1 | Add ILGPU NuGet dependency (+ ManagedCuda.NVRTC) | Done |
| 4b.2 | IlgpuCompiler (ILGPU Context + PTXBackend → PTX → PtxLoader.LoadFromBytes → LoadedKernel, cached by method hash SHA256) | Done |
| 4b.3 | KernelSource discriminated union (FilesystemPtx \| IlgpuMethod \| NvrtcSource) with GetCacheKey()/GetDebugName() | Done |
| 4b.4 | BlockBuilder.AddKernel(MethodInfo, KernelDescriptor) with ILGPU parameter expansion + index remap | Done |
| 4b.5 | NvrtcCache (CudaRuntimeCompiler → PTX → PtxLoader.LoadFromBytes, cached by source hash SHA256) | Done |
| 4b.6 | BlockBuilder.AddKernelFromCuda(cudaSource, entryPoint, descriptor) | Done |
| 4b.7 | DirtyTracker: IsCodeDirty / MarkCodeDirty(DirtyCodeEntry) / ClearCodeDirty() / GetDirtyCodeEntries() | Done |
| 4b.8 | DirtyCodeEntry record struct (BlockId, KernelHandleId, NewSource) | Done |
| 4b.9 | CudaContext.OnCodeChanged() → DirtyTracker.MarkCodeDirty() | Done |
| 4b.10 | CudaEngine.CodeRebuild(): invalidate compiler caches by KernelSource variant → ColdRebuild | Done |
| 4b.11 | CudaEngine.Update() priority: Structure > Code > CapturedNodes > Parameters | Done |
| 4b.12 | CudaEngine.LoadKernelFromSource(): dispatches FilesystemPtx/IlgpuMethod/NvrtcSource | Done |
| 4b.13 | CudaEngine.InitializeIlgpuParams(): sets _kernel_length and ArrayView struct length fields | Done |
| 4b.14 | KernelHandle: Source property (KernelSource), index remap for ILGPU kernels | Done |
| 4b.15 | IlgpuCompiler.ExpandDescriptorForIlgpu(): param 0 = _kernel_length, ArrayView → 16-byte struct | Done |
| 4b.16 | IlgpuCompiler.ComputeIndexRemap(): user indices shift +1 for _kernel_length | Done |
| 4b.17 | CudaContext owns IlgpuCompiler + NvrtcCache (created in constructor, disposed on Dispose) | Done |

### Implementation Notes

- `IlgpuCompiler` creates an ILGPU Context without accelerator (pure compiler mode) — no conflict with ManagedCuda's CUDA context
- PTXBackend targets the device's compute capability via `CudaArchitecture` + `CudaInstructionSet`
- ILGPU parameter layout: param 0 is implicit `_kernel_length` (int32), then user params where `ArrayView<T>` becomes a 16-byte struct `{void* ptr, long length}`
- `KernelHandle.In()/Out()` transparently remap indices via `_indexRemap` array (all user indices shift +1)
- `NvrtcCache` auto-derives `--gpu-architecture=compute_XY` from `DeviceContext`. Compilation errors include the NVRTC log
- Code dirty level currently triggers a full Cold Rebuild (not a partial rebuild of the affected block only)
- `DirtyTracker.ClearStructureDirty()` also clears code, captured, and parameter dirty flags (Cold Rebuild applies all current values)
- `KernelEntry.Source` stores the `KernelSource` variant; `KernelEntry.Descriptor` stores the original (pre-expansion) descriptor for ILGPU recompilation

### Deliverables

```csharp
// Should work (and does):

// Patchable kernel via ILGPU (primary path)
var descriptor = new KernelDescriptor { EntryPoint = "my_kernel", BlockSize = 256 };
descriptor.Parameters.Add(new KernelParamDescriptor { Name = "A", IsPointer = true, Direction = ParamDirection.In });
descriptor.Parameters.Add(new KernelParamDescriptor { Name = "B", IsPointer = true, Direction = ParamDirection.Out });

var kernel = builder.AddKernel(myKernelMethod, descriptor);
kernel.GridDimX = 1024;
builder.Input<float>("A", kernel.In(0));   // remapped to PTX param 1 (ArrayView struct)
builder.Output<float>("B", kernel.Out(1)); // remapped to PTX param 2
// Hot/Warm parameter updates work like filesystem PTX
// Code update: CudaContext.OnCodeChanged() → Code dirty → IlgpuCompiler cache invalidated → Cold Rebuild

// User CUDA C++ via NVRTC (escape-hatch)
string userCode = File.ReadAllText("my_kernel.cu");
var nvrtcDescriptor = new KernelDescriptor { EntryPoint = "my_kernel", BlockSize = 256, ... };
var kernel = builder.AddKernelFromCuda(userCode, "my_kernel", nvrtcDescriptor);
builder.Input<float>("A", kernel.In(0));

// Mixed graph: all three kernel sources + CapturedNodes in same graph
// CudaEngine.LoadKernelFromSource() dispatches based on KernelSource variant
```

### Test Cases

1. ILGPU compile C# method → PTX string → LoadedKernel via PtxLoader.LoadFromBytes
2. IlgpuCompiler cache deduplication (same method → same LoadedKernel)
3. IlgpuCompiler cache invalidation (Invalidate → recompile on next GetOrCompile)
4. ILGPU parameter expansion (ArrayView → 16-byte struct, implicit _kernel_length at index 0)
5. ILGPU index remap (user-facing indices shift +1)
6. KernelSource discriminated union (FilesystemPtx, IlgpuMethod, NvrtcSource)
7. KernelSource.GetCacheKey() uniqueness
8. NVRTC compile CUDA C++ string → PTX bytes → LoadedKernel
9. NvrtcCache deduplication (same source → cached LoadedKernel)
10. NvrtcCache invalidation
11. BlockBuilder.AddKernel(MethodInfo, descriptor) creates correct KernelHandle with index remap
12. BlockBuilder.AddKernelFromCuda() creates correct KernelHandle with NvrtcSource
13. DirtyTracker Code level (MarkCodeDirty, IsCodeDirty, GetDirtyCodeEntries, ClearCodeDirty)
14. DirtyTracker priority: Structure > Code > CapturedNodes > Parameters
15. CudaEngine.CodeRebuild() invalidates correct compiler cache per KernelSource variant
16. CudaEngine.Update() Code dirty → CodeRebuild → ColdRebuild
17. CudaEngine.InitializeIlgpuParams() sets _kernel_length and ArrayView lengths

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

## Phase 6: Device Graph Launch & Dispatch Indirect

**Goal**: Device-side graph control — GPU kernels dynamically set grid dimensions and control flow, enabling true dispatch indirect without CPU involvement.

**Prerequisites**: Phase 3.2 (Conditional Nodes), Phase 5 (Graphics Interop — optional but recommended)

> See `GRAPH-COMPILER.md` and `KERNEL-SOURCES.md` for the full architecture design.

### Motivation

Many GPU patterns produce variable-length output (particle emission, stream compaction, spatial queries). Without dispatch indirect, subsequent kernels must either (a) launch with max grid size and waste threads, or (b) read the count back to CPU causing a pipeline stall. Device Graph Launch eliminates both problems.

### Tasks

| Task | Description | Depends On |
|------|-------------|------------|
| 6.1 | AppendBuffer Counter Buffer as GPU pointer pin (expose `GpuBuffer<uint>` alongside CPU readback) | Phase 3.1 |
| 6.2 | Device-updatable kernel node launch attributes (`CUlaunchAttributeID.DeviceUpdatableKernelNode`) | 6.1 |
| 6.3 | DeviceLaunch graph instantiation (`CUgraphInstantiate_flags.DeviceLaunch`) | 6.2 |
| 6.4 | `cuGraphUpload()` integration in CudaEngine after device-side modifications | 6.3 |
| 6.5 | DispatcherNode — built-in PTX kernel that reads counter + calls `cudaGraphKernelNodeSetParam` | 6.3 |
| 6.6 | BlockBuilder.AddDispatcher(counterPin, targetKernel, blockSize) | 6.5 |
| 6.7 | GraphCompiler: detect DispatcherNode → mark target as device-updatable → DeviceLaunch instantiation | 6.6 |
| 6.8 | VL `DispatchIndirect` node (Consumer level — connects AppendBuffer counter to next block) | 6.7 |
| 6.9 | Integration tests: Emit → DispatchIndirect → Forces → Integrate chain | 6.8 |
| 6.10 | Performance benchmark: Max-Grid vs Conditional vs DispatchIndirect at various element counts | 6.9 |

### Deliverables

```csharp
// Should work:

// Level 1 (Phase 3, already possible): Max-Grid + Counter Pointer
var kernel = builder.AddKernel("forces.ptx");
builder.Input<float4>("Particles", kernel.In(0));
builder.Input<uint>("ActiveCount", kernel.In(1));    // GPU pointer to counter
kernel.GridDimX = maxParticles / 256;                // always max

// Level 3 (Phase 6): True Dispatch Indirect
var emit = builder.AddKernel("emit.ptx");
var forces = builder.AddKernel("forces.ptx");
var dispatcher = builder.AddDispatcher(
    counterPin: emit.AppendCounter(),   // GPU counter from AppendBuffer
    targetKernel: forces,               // kernel whose grid will be set dynamically
    blockSize: 256);
// GraphCompiler handles: DeviceLaunch flag, device-updatable attributes, Upload()
```

### Key Constraints

- All nodes must be on a **single CUcontext** (already our constraint)
- No CUDA Dynamic Parallelism inside device-updatable kernels
- Device-updatable nodes **cannot be removed** from the graph
- Graph with DeviceLaunch flag does **not support multiple instantiation**
- Must call `cuGraphUpload()` before launch after device-side modifications
- DispatcherNode PTX is a **system-provided kernel** (not user-written)

### ManagedCuda APIs (Verified Present)

| API | Status |
|-----|--------|
| `CUgraphInstantiate_flags.DeviceLaunch` | Present |
| `cuGraphInstantiateWithParams()` | Present |
| `cuGraphUpload()` | Present + managed wrapper |
| `CUlaunchAttributeID.DeviceUpdatableKernelNode` | Present |
| `CUgraphDeviceNode` | Present |
| `cuGraphConditionalHandleCreate()` | Present + managed wrapper |
| Device-side APIs (PTX intrinsics) | N/A (called from kernel code) |

### Success Criteria

- [ ] AppendBuffer Counter exposed as GPU pointer pin
- [ ] DispatcherNode PTX kernel reads counter + sets target grid
- [ ] DeviceLaunch instantiation works with mixed graphs (kernel + captured + dispatcher)
- [ ] `cuGraphUpload()` called automatically by CudaEngine
- [ ] VL DispatchIndirect node connects AppendBuffer → next processing block
- [ ] Zero CPU readback in Emit → Forces → Integrate chain
- [ ] Performance improvement measurable at > 100K elements vs Max-Grid approach

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

VL Integration Gate (validated in vvvv gamma 7.1):
- [x] Package discovery (source-package via `--package-repositories`)
- [x] NodeContext injection (auto-injected by VL)
- [x] ProcessNode lifecycle (Create → Update → Dispose)
- [x] GPU on VL's thread (CUDA calls work)
- [x] Tooltip display (ToString() shows in VL)
- [x] State output pin (HasStateOutput = true)

Phase 2 VL Runtime Integration (deferred — done when first block is built):
- [ ] PinGroups display (needs VL.Core PinGroupKind)
- [ ] Hot-Swap simulation (needs VL runtime)
- [ ] IVLRuntime diagnostics (needs VL runtime)
- [ ] AppHost.TakeOwnership (needs VL runtime)

### Phase 3.1 Complete (AppendBuffer)
- [x] AppendBuffer\<T\> wraps GpuBuffer\<T\> Data + GpuBuffer\<uint\> Counter (composition)
- [x] IAppendBuffer interface for type-erased tracking
- [x] BufferPool.AcquireAppend\<T\>() allocates both buffers
- [x] MemsetDescriptor → cuGraphAddMemsetNode for counter reset
- [x] GraphCompiler memset node phase (between WireParameters and kernel insertion)
- [x] Memset nodes wired as dependencies for kernel nodes using append outputs
- [x] BlockBuilder.AppendOutput\<T\>() with auto-generated "{name} Count" port
- [x] AppendBufferInfo in BlockDescription
- [x] CudaEngine auto-readback of append counts after launch
- [x] BlockDebugInfo.AppendCounts populated per frame
- [x] 19 tests passing (167 total)

### Phase 3 Remaining
- [ ] Conditional regions execute correctly
- [ ] Composite blocks compose properly
- [ ] Serialization round-trips

### Phase 4a Complete
- [x] IGraphNode interface (polymorphic: KernelNode + CapturedNode)
- [x] CapturedNode with stream capture → CUgraph, Capture()/Dispose() lifecycle
- [x] CapturedNodeDescriptor (Inputs/Outputs/Scalars as CapturedParam)
- [x] StreamCaptureHelper (CaptureToGraph, DestroyGraph, AddChildGraphNode, UpdateChildGraphNode)
- [x] LibraryHandleCache caches cuBLAS/cuFFT/cuSPARSE/cuRAND/cuSOLVER handles
- [x] AddCaptured() works in BlockBuilder (static + chained)
- [x] CapturedHandle.In()/Out()/Scalar() return CapturedPin with flat indices
- [x] Input\<T\>/Output\<T\> overloads accept CapturedPin
- [x] CapturedEntry in BlockDescription.CapturedEntries
- [x] GraphCompiler Phase 5b: capture on dedicated stream → AddChildGraphNode
- [x] GraphCompiler Phase 5c: kernel node deps on captured node handles
- [x] CompiledGraph stores capturedNodeHandles + RecaptureNode()
- [x] DirtyTracker: AreCapturedNodesDirty, MarkCapturedNodeDirty(), ClearCapturedNodesDirty()
- [x] DirtyCapturedNode record struct (BlockId, CapturedHandleId)
- [x] CudaContext.OnCapturedNodeChanged() routes to DirtyTracker
- [x] CudaEngine: Structure > Recapture > Parameters priority
- [x] CudaEngine.RecaptureNodes() dispatches correctly by node type
- [x] Mixed graph (KernelNodes + CapturedNodes) compiles and executes
- [x] BlasOperations (Sgemm, Dgemm, Sgemv, Sscal)
- [x] FftOperations (Forward1D, Inverse1D, R2C1D, C2R1D)
- [x] SparseOperations (SpMV)
- [x] RandOperations (GenerateUniform, GenerateNormal)
- [x] SolveOperations (Sgetrf, Sgetrs)
- [x] 44 tests passing (211 total)

### Phase 4b Complete
- [x] ILGPU NuGet added, PTXBackend accessible (pure compiler mode, no CUDA accelerator)
- [x] IlgpuCompiler: ILGPU Context + PTXBackend → PTX → PtxLoader.LoadFromBytes → LoadedKernel, cached by method hash (SHA256)
- [x] KernelSource discriminated union (FilesystemPtx | IlgpuMethod | NvrtcSource) with GetCacheKey()/GetDebugName()
- [x] BlockBuilder.AddKernel(MethodInfo, KernelDescriptor) with ILGPU parameter expansion + index remap
- [x] ILGPU parameter layout: param 0 = implicit _kernel_length, ArrayView\<T\> → 16-byte struct {ptr, length}
- [x] CudaEngine.InitializeIlgpuParams() sets _kernel_length and ArrayView struct length fields
- [x] Dirty-Tracking: Code level (IsCodeDirty, MarkCodeDirty, DirtyCodeEntry) triggers CodeRebuild → Cold Rebuild
- [x] CudaEngine.Update() priority: Structure > Code > CapturedNodes > Parameters
- [x] CudaEngine.CodeRebuild() invalidates compiler caches by KernelSource variant
- [x] NvrtcCache compiles CUDA C++ strings via CudaRuntimeCompiler, cached by source hash (SHA256)
- [x] BlockBuilder.AddKernelFromCuda(cudaSource, entryPoint, descriptor)
- [x] CudaEngine.LoadKernelFromSource() dispatches all three KernelSource variants
- [x] CudaContext owns IlgpuCompiler + NvrtcCache (lifetime management)
- [x] 37 tests passing (248 total, 9 skipped)

### Phase 5 Complete
- [ ] DX11 buffer sharing works
- [ ] DX11 texture sharing works
- [ ] Can render CUDA output in Stride
- [ ] Full pipeline: CUDA compute → Stride render
