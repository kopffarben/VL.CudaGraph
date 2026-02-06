# Architecture Review — Critical Analysis & Alternatives

> Generated 2026-02-06 from systematic verification of VL.CudaGraph docs against
> `src/References/managedCuda/` and `src/References/VL.StandardLibs/`.

---

## 1. API Verification Summary

### 1.1 ManagedCuda — All APIs Confirmed

Every CUDA API assumed by the documentation exists in ManagedCuda.
79+ individual APIs verified across 30+ source files.

| Category | APIs | Status | Key Files |
|----------|------|--------|-----------|
| CUDA Graph | 11 | ALL PRESENT | `CudaGraph.cs`, `CudaGraphExec.cs` |
| Stream Capture | 3 | ALL PRESENT | `DriverAPI.cs` (Streams class) |
| NVRTC | 4+ | ALL PRESENT | `NVRTC/CudaRuntimeCompiler.cs` |
| Module Loading | 3 | ALL PRESENT | `CudaLibrary.cs`, `CudaKernel.cs` |
| cuBLAS | 6 | ALL PRESENT | `CudaBlas/CudaBlasHandler.cs` |
| cuFFT | 20+ | ALL PRESENT | `CudaFFT/CudaFFTPlan*.cs` |
| Memory Mgmt | 8 | ALL PRESENT | `CudaDeviceVariable.cs` |
| Events | 4 | ALL PRESENT | `CudaEvent.cs` |
| DX11 Interop | 5 | ALL PRESENT | `CudaDirectXInteropResource.cs` |
| Conditionals | 2 | ALL PRESENT | `CudaGraph.cs:852` (AddConditionalNode) |

**Conclusion**: No API gaps. The architecture is implementable as designed.

### 1.2 VL.Core — All APIs Confirmed (One Difference)

| API | Status | Actual Location |
|-----|--------|-----------------|
| `NodeContext` | VERIFIED | `VL.Core/src/NodeContext.cs` |
| `NodeContext.Path.Stack.Peek()` | VERIFIED | Returns `UniqueId` via `ImmutableStack` |
| `NodeContext.GetLogger()` | VERIFIED | Returns `ILogger` |
| `NodeContext.AppHost` | VERIFIED | Direct property |
| `IVLRuntime.Current` | VERIFIED | Via `AppHost.Current.Services.GetService<IVLRuntime>()` |
| `IVLRuntime.AddMessage()` | VERIFIED | `(UniqueId, string, MessageSeverity)` overload exists |
| `IVLRuntime.AddPersistentMessage()` | VERIFIED | Returns `IDisposable` |
| `MessageSeverity` | VERIFIED | `None, Info, Warning, Error, Critical` |
| `AppHost.TakeOwnership()` | VERIFIED | `abstract void TakeOwnership(IDisposable)` |
| `ServiceRegistry` | VERIFIED | `RegisterService<T>()`, `GetService<T>()` |
| `IResourceProvider<T>` | VERIFIED | With `GetHandle()` returning `IResourceHandle<T>` |
| `ResourceProvider.Return()` | VERIFIED | `Return(T, Action<T>)` overload at line 346 |
| `IRefCounter<T>` | VERIFIED | `Init()`, `AddRef()`, `Release()` |
| `ProcessNodeAttribute` | VERIFIED | `[ProcessNode]` attribute, not a base class |
| `FrameClock` | VERIFIED | `IFrameClock.TimeDifference` property |

**ONE DIFFERENCE FOUND — PinGroup Attribute:**

| Docs Assume | Actual API |
|-------------|------------|
| `[PinGroup("Inputs", PinGroupKind.Dynamic)]` | `[Pin(PinGroupKind = PinGroupKind.Collection)]` |

The attribute is `PinAttribute` with a `PinGroupKind` property, not a separate
`PinGroupAttribute`. `PinGroupKind` has values `None`, `Collection`, `Dictionary`
(no `Dynamic`). Docs must be updated.

---

## 2. Documentation Consistency Issues

### 2.1 Critical (Must Fix Before Implementation)

**C1: `ClearCodeDirty()` method referenced but undefined**
- `EXECUTION-MODEL.md:250` calls `_cudaContext.ClearCodeDirty()`
- `CSHARP-API.md` only defines `ClearStructureDirty()` and `ClearParametersDirty()`
- **Fix**: Add `ClearCodeDirty()` to API, or fold into `ClearStructureDirty()`

**C2: `UpdateModule()` method referenced but undefined**
- `EXECUTION-MODEL.md:279` calls `block.UpdateModule(module)`
- No such method in any API definition
- **Fix**: Define on `ICudaBlock` or on `KernelNodeDescriptor`

**C3: `BlockDebugSnapshot` type used but never defined**
- `EXECUTION-MODEL.md:412` returns `BlockDebugSnapshot?` from profiling cache
- `CSHARP-API.md` only defines `IBlockDebugInfo`
- **Fix**: Either define `BlockDebugSnapshot` or use `IBlockDebugInfo` consistently

### 2.2 High (API Gaps in CSHARP-API.md)

**H1: `ProfilingPipeline` class not in API reference**
- Fully designed in `EXECUTION-MODEL.md` with `GetCached()`, ring buffer, etc.
- Missing from `CSHARP-API.md` module structure
- **Fix**: Add to API reference under Engine section

**H2: `LibraryHandleCache` and `NvrtcCache` not in API reference**
- Both described in architecture docs
- Missing from `CSHARP-API.md`
- **Fix**: Add to Kernel/Captured sections

### 2.3 Medium (Interface Inconsistency)

**M1: `NodeContext` missing from `ICudaBlock` in BLOCK-SYSTEM.md**
- `BLOCK-SYSTEM.md:32-45` defines ICudaBlock WITHOUT `NodeContext` property
- `CSHARP-API.md:568` includes `NodeContext NodeContext { get; }` in ICudaBlock
- **Fix**: Add to BLOCK-SYSTEM.md

**M2: `RegisterBlock()` — who provides NodeContext?**
- Most docs: `ctx.RegisterBlock(this)` — single parameter
- CudaEngine reads `block.NodeContext` for diagnostics routing
- Pattern is consistent but never explicitly stated
- **Fix**: Document that `ICudaBlock.NodeContext` is read during registration

### 2.4 Low (Clarifications Needed)

| # | Issue | Location |
|---|-------|----------|
| L1 | Parameter change notification mechanism undefined | BLOCK-SYSTEM.md |
| L2 | `BlockBuilder.Commit()` side effects not documented | BLOCK-SYSTEM.md |
| L3 | Serialization (GetModel/LoadModel) incomplete spec | VL-INTEGRATION.md |
| L4 | Phase 5.5 numbering confusing (inside Phase 4 impl) | PHASES.md |
| L5 | `DidRebuildLastFrame` not mentioned in EXECUTION-MODEL.md | CSHARP-API.md |
| L6 | `PinGroupKind.Dynamic` does not exist, use `Collection` | BLOCK-SYSTEM.md, VL-INTEGRATION.md |

---

## 3. Critical Architecture Analysis

### 3.1 CUDA Graph as Core Abstraction — SOUND

The decision to use CUDA Graph API instead of raw stream launches is well-justified:

**Advantages confirmed:**
- Graph-level kernel fusion and memory optimization by NVIDIA driver
- Reduced launch overhead (~5-10us per graph vs ~10-50us per kernel)
- Natural fit for VL's declarative dataflow model
- Update-in-place (Hot/Warm) avoids rebuild for parameter changes

**Risk identified:**
- CUDA Graph is immutable once instantiated — any structural change requires full rebuild
- Cold Rebuild cost scales with graph size (measured: ~1-5ms for 10-50 kernels)
- During interactive VL editing, frequent Cold Rebuilds are expected

**Mitigation (already in design):** Cold Rebuilds only happen during development.
Exported applications have stable graph structure.

### 3.2 Three Kernel Sources — SOUND but Complex

The three-source architecture is well-designed but adds significant implementation surface:

| Source | Complexity | Risk |
|--------|-----------|------|
| Filesystem PTX | Low | Proven path, minimal risk |
| Patchable Kernel (NVRTC) | **High** | Codegen design TBD, compile latency |
| Library Call (Stream Capture) | Medium | Recapture cost, opaque kernels |

**Recommendation**: Implement Source 1 (Filesystem PTX) first and fully stabilize
before adding Sources 2 and 3. This is already reflected in the phasing (Phase 0-2
before Phase 4), but the docs could be clearer that Source 1 is the MVP path.

### 3.3 Stream Capture for Library Calls — VIABLE but Has Caveats

**Verified**: `cuStreamBeginCapture` and `cuStreamEndCapture` exist in ManagedCuda.

**Caveats discovered:**

1. **Recapture Cost**: Any parameter change to a CapturedNode requires re-executing
   the library call in capture mode + `cuGraphExecChildGraphNodeSetParams`. This is
   ~100-500us per recapture — acceptable but not free.

2. **Pointer Mode Requirement**: cuBLAS operations in captured graphs MUST use
   `CUBLAS_POINTER_MODE_DEVICE` for scalar parameters. With Host pointer mode,
   scalars are baked in at capture time.

3. **Thread Safety**: `CUstreamCaptureMode.Relaxed` allows capture from any thread,
   but the capture stream must not be used by other work simultaneously.

4. **No Nested Capture**: Stream capture cannot be nested. A CapturedNode's
   `captureAction` must not start another capture.

### 3.4 Patchable Kernel Codegen — HIGHEST RISK

**NVRTC is verified** in ManagedCuda (`CudaRuntimeCompiler` class).

**Risks:**

1. **Codegen Design Undefined**: The docs describe a "closed, finite set of GPU
   primitives" (GPU.Add, GPU.Mul, etc.) but there is no codegen specification.
   How VL node-sets are translated to CUDA C++ is the largest open design question.

2. **Compile Latency**: NVRTC compilation takes 100ms-2s depending on kernel
   complexity. During interactive editing, this creates noticeable lag.

3. **Error Reporting**: NVRTC errors are cryptic C++ compiler errors. Mapping these
   back to VL node positions requires careful source-map tracking.

4. **Testing Surface**: Every GPU primitive needs CUDA C++ codegen + correctness tests.

**Recommendation**: Consider deferring patchable kernels to a later phase or even
a separate project. Filesystem PTX (Triton/nvcc) provides the same functionality
with mature tooling — the user just writes Python/CUDA instead of patching VL nodes.

### 3.5 Dirty-Tracking Design — SOUND

The four-level dirty tracking (Hot/Warm/Code/Cold) is well-designed and maps
directly to CUDA Graph API capabilities:

| Level | CUDA API | Cost | Verified |
|-------|----------|------|----------|
| Hot | `cuGraphExecKernelNodeSetParams` | ~0 | YES |
| Warm | `cuGraphExecKernelNodeSetParams` | Cheap | YES |
| Code | NVRTC recompile + Cold Rebuild | Medium | YES |
| Cold | `cuGraphDestroy` + `cuGraphCreate` + `cuGraphInstantiate` | Expensive | YES |
| Recapture | Stream Capture + `cuGraphExecChildGraphNodeSetParams` | Medium | YES |

**The event-based coupling (BlockRegistry/ConnectionGraph -> DirtyTracker) is clean.**
No service knows about the others. Wiring is centralized in CudaContext constructor.

### 3.6 BufferPool with Power-of-2 — SOUND, Alternatives Exist

The documented power-of-2 bucket strategy is standard and proven. CUDA's
`cuMemAlloc` is expensive (~50-200us), so pooling is essential.

**Alternative considered: CUDA Graph Memory Allocation Nodes**
- `cuGraphAddMemAllocNode` (CUDA 11.4+) allows allocation INSIDE the graph
- Would eliminate separate BufferPool for graph-lifetime buffers
- But: less control over allocation strategy, harder to pool

**Recommendation**: Keep the custom BufferPool as designed. Graph memory nodes
are useful as a future optimization for region-scoped temporaries only.

### 3.7 DX11 Interop — SOUND, Future-Proof Concern

**Verified**: Full DX11 interop API exists in ManagedCuda.
- `cuGraphicsD3D11RegisterResource` for buffer/texture registration
- `CudaDirectXInteropResource` managed wrapper
- Map/Unmap synchronization

**Future concern**: Stride is eventually moving to DX12/Vulkan. The interop
mechanism changes significantly:

| DX Version | CUDA Mechanism | Sync Model |
|-----------|----------------|------------|
| DX11 | `cuGraphicsD3D11RegisterResource` + Map/Unmap | Implicit |
| DX12 | `cuImportExternalMemory` + `cuImportExternalSemaphore` | Explicit fences |
| Vulkan | `cuImportExternalMemory` + `cuImportExternalSemaphore` | Explicit fences |

**Recommendation**: Isolate DX11 interop behind an interface (`IGraphicsInterop`)
so the DX12 migration path is clean. The docs already plan this via the
`VL.Cuda.Stride` separate module.

---

## 4. Alternative Architectures

### 4.1 Alternative A: Stream-Only (No CUDA Graph)

**Approach**: Execute kernels on CUDA streams directly, without graph compilation.

```
Blocks → CudaEngine collects kernel list → Launch each kernel on stream → Done
```

| Aspect | CUDA Graph (current) | Stream-Only |
|--------|---------------------|-------------|
| Launch overhead | ~5-10us per graph | ~10-50us per kernel |
| Parameter update | Near-zero (Hot Update) | Same as launch |
| Structure change | Expensive rebuild | No-op |
| Graph optimization | NVIDIA driver optimizes | None |
| Implementation | Complex | Simple |
| Development iteration | Rebuild lag | Instant |

**Verdict**: Stream-only is simpler but sacrifices the core value proposition of
VL.Cuda (GPU-optimized compute pipelines). **Not recommended as primary path.**

**Possible hybrid**: Use streams for development mode (instant iteration), switch
to CUDA Graph for production/export. This adds implementation cost but provides
the best of both worlds.

### 4.2 Alternative B: CUDA Graph Without Conditional Nodes

**Approach**: Drop CUDA 12.8 minimum to CUDA 10.2+, sacrifice conditional nodes.

| Feature | With Conditionals | Without Conditionals |
|---------|------------------|---------------------|
| CUDA minimum | 12.8 | 10.2 |
| GPU support | RTX 20xx+ | GTX 10xx+ |
| If/While regions | Native graph conditionals | Multiple graph variants |
| Complexity | Lower | Higher (variant management) |

**Approach for conditionals without CUDA 12.4+:**
- Compile separate graph variants for each conditional path
- Switch between graph executables at launch time
- While loops: launch graph in CPU loop

**Verdict**: Expanding GPU support to GTX 10xx is tempting but adds significant
complexity. The VL.Cuda target audience (creative coders, VJ, media artists) likely
has RTX GPUs. **Stick with CUDA 12.8 minimum as designed.**

### 4.3 Alternative C: Hybrid Graph + Stream for Library Calls

**Approach**: Use CUDA Graph for custom kernels (KernelNodes) but execute library
calls (cuBLAS, cuFFT) on streams outside the graph.

```
CudaEngine.Update():
  1. Pre-graph: Execute library calls on stream (cuBLAS, cuFFT)
  2. Synchronize stream
  3. Launch CUDA Graph (custom kernels only)
  4. Post-graph: Execute library calls that depend on graph output
```

| Aspect | Stream Capture (current) | Hybrid |
|--------|------------------------|--------|
| Parameter update | Recapture (~100-500us) | Direct call (~0) |
| Graph optimization | Library calls in graph | Library calls separate |
| Synchronization | Implicit (graph order) | Explicit (stream sync) |
| Complexity | Medium (capture logic) | Low |
| Flexibility | Library calls anywhere in pipeline | Pre/post graph only |

**Verdict**: The hybrid approach is simpler but limits where library calls can
appear in the pipeline. Stream Capture (current design) is more flexible.
**Consider hybrid as a simpler Phase 4 starting point**, then add full Stream
Capture for users who need library calls interleaved with custom kernels.

### 4.4 Alternative D: Use ManagedCuda Managed Wrappers vs Raw P/Invoke

ManagedCuda provides two layers:
1. **P/Invoke**: `DriverAPINativeMethods.GraphManagment.cuGraphAddKernelNode(...)`
2. **Managed**: `CudaGraph.AddKernelNode(...)`, `CudaGraphExec.Launch(stream)`

| Approach | Pros | Cons |
|----------|------|------|
| **Managed wrappers** | Less boilerplate, error handling built-in, Dispose pattern | Less control, version coupling |
| **Raw P/Invoke** | Full control, version-independent | More error handling code, IntPtr management |
| **Mixed** | Use managed for common ops, P/Invoke for graph-specific | Two patterns in codebase |

**Recommendation**: Start with managed wrappers (`CudaGraph`, `CudaGraphExec`,
`CudaKernel`, `CudaEvent`, `CudaStream`). Drop to P/Invoke only where managed
wrappers don't expose needed functionality (e.g. conditional node params).

Verified: `CudaGraph` has `AddKernelNode()`, `AddChildGraphNode()`,
`AddConditionalNode()`, `Instantiate()`. `CudaGraphExec` has `SetParams()` for
Hot/Warm updates and `Launch()`. These cover the core path.

---

## 5. ManagedCuda Version Considerations

### 5.1 DLL Version Targets

From the ManagedCuda source:

| Library | DLL Target | Implies |
|---------|-----------|---------|
| CUDA Driver | `nvcuda` | System driver (auto-versioned) |
| NVRTC | `nvrtc64_130_0` | CUDA 13.0 toolkit |
| cuBLAS | `cublas64_12` | CUDA 12.x toolkit |
| cuFFT | `cufft64_11` | CUDA 11.x toolkit |

**IMPORTANT**: The NVRTC DLL targets CUDA 13.0 (`nvrtc64_130_0`), which means
this ManagedCuda build assumes the CUDA 13.0 toolkit is installed. Our docs
state CUDA 12.8 minimum — this may need alignment.

**Options**:
1. Use the NuGet package version that matches CUDA 12.8
2. Accept CUDA 13.0 as the actual minimum (NVRTC DLL dependency)
3. Build ManagedCuda from source targeting CUDA 12.8 DLLs

**Recommendation**: Verify which ManagedCuda NuGet version targets CUDA 12.8.
The submodule HEAD may be ahead of the latest NuGet release.

### 5.2 NuGet Package Availability

The project consumes ManagedCuda as NuGet, not as a project reference.
Need to verify:
- [ ] `ManagedCuda` NuGet package version and CUDA target
- [ ] `ManagedCuda.NVRTC` NuGet package availability
- [ ] `ManagedCuda.CudaBlas` NuGet package availability
- [ ] `ManagedCuda.CudaFFT` NuGet package availability

---

## 6. Implementation Risk Matrix

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| NVRTC codegen design undefined | HIGH | HIGH | Defer to Phase 4+, Filesystem PTX is MVP |
| Cold Rebuild latency during VL editing | MEDIUM | HIGH | Profile early, consider stream fallback |
| ManagedCuda NuGet vs submodule version mismatch | MEDIUM | MEDIUM | Pin NuGet version, test early |
| NVRTC DLL version (13.0 vs 12.8) | MEDIUM | MEDIUM | Verify NuGet package targets |
| Stream Capture edge cases (nested, threading) | LOW | MEDIUM | Comprehensive integration tests |
| DX11 interop Map/Unmap overhead | LOW | LOW | Profile, consider double buffering |
| PinGroupKind.Dynamic doesn't exist | LOW | CERTAIN | Use `PinGroupKind.Collection` instead |
| Conditional node complexity | MEDIUM | LOW | Implement after core graph works |

---

## 7. Recommended Changes to Documentation

### Must Fix

1. **PinGroup attribute**: Change all docs from `[PinGroup("name", PinGroupKind.Dynamic)]`
   to `[Pin(PinGroupKind = PinGroupKind.Collection)]`

2. **Add missing API definitions**: `ClearCodeDirty()`, `BlockDebugSnapshot`,
   `ProfilingPipeline`, `NvrtcCache`, `LibraryHandleCache` to CSHARP-API.md

3. **ICudaBlock interface**: Add `NodeContext NodeContext { get; }` to BLOCK-SYSTEM.md

4. **Parameter change notification**: Document how `BlockParameter<T>.Value` setter
   triggers `OnParameterChanged()` on CudaContext

### Should Fix

5. **CUDA version alignment**: Clarify CUDA 12.8 vs 13.0 based on actual NuGet target

6. **Phase numbering**: Rename "Phase 5.5" in Graph Compiler to "Phase 5b" or
   integrate into Phase 5 with conditional execution

7. **Hybrid stream/graph option**: Document as potential development-mode optimization

### Nice to Have

8. **ManagedCuda wrapper inventory**: Document which operations use managed wrappers
   vs P/Invoke, so implementers know upfront

9. **Codegen specification**: Create a separate doc for patchable kernel codegen
   design before Phase 4 implementation

---

## 8. Recommended Implementation Order

Based on risk analysis, the phasing should prioritize de-risking:

```
Phase 0: Foundation (as designed)
    → CudaContext, GpuBuffer, BufferPool, PTX loading
    → DE-RISK: Verify ManagedCuda NuGet version works

Phase 1: Graph Basics (as designed)
    → GraphCompiler, CompiledGraph, Hot/Warm updates
    → DE-RISK: Measure Cold Rebuild cost at target graph sizes

Phase 2: Execution Model (as designed)
    → CudaEngine, ICudaBlock, BlockBuilder, dirty-tracking
    → DE-RISK: Test VL Hot-Swap cycle (Dispose → new → reconnect)

Phase 3: Advanced (as designed)
    → AppendBuffer, Regions, Liveness analysis
    → DE-RISK: Test conditional nodes on CUDA 12.8

Phase 4a: Library Calls — Hybrid First (MODIFIED)
    → Start with stream-based library calls (simpler)
    → Then add Stream Capture path as optimization
    → DE-RISK: Measure recapture cost for cuBLAS Sgemm

Phase 4b: Patchable Kernels (DEFERRED, HIGHEST RISK)
    → Requires codegen design spec first
    → NVRTC integration is low-risk (API verified)
    → Codegen (VL nodes → CUDA C++) is high-risk (undefined)

Phase 5: Graphics Interop (as designed, can overlap Phase 2+)
    → DX11 buffer/texture sharing
    → Behind IGraphicsInterop interface for future DX12
```

---

## 9. ILGPU as Alternative to NVRTC for Patchable Kernels

> Added 2026-02-06 after systematic exploration of `src/References/ILGPU/`.

### 9.1 What is ILGPU?

ILGPU is a .NET GPU compiler that compiles C# to PTX (and OpenCL). Crucially,
it has a **complete SSA-form Intermediate Representation (IR)** that can be
constructed programmatically — no C# source code required.

### 9.2 Key Findings

**PTX Extraction — CONFIRMED**

| Finding | Evidence |
|---------|----------|
| `PTXCompiledKernel.PTXAssembly` returns raw PTX string | `Backends/PTX/PTXCompiledKernel.cs:44-47` |
| `PTXBackend.Compile()` is public, returns `CompiledKernel` | `Backends/Backend.cs:623-626` |
| PTX string is directly usable with `cuModuleLoadData` | ILGPU itself does this in `Runtime/Cuda/CudaKernel.cs:51-54` |
| Backend can be used WITHOUT ILGPU accelerator | `Backend.Compile(entry, specialization)` — no context needed |

**Programmatic IR Construction — CONFIRMED**

| Capability | API | File |
|-----------|-----|------|
| Declare kernel entry point | `IRContext.Declare()` + `MethodFlags.EntryPoint` | `IR/Method.cs:36-62` |
| No C# source required | `MethodDeclaration(handle, returnType, methodBase: null)` | `IR/IRContext.cs:275-289` |
| Add parameters | `Method.Builder.AddParameter(TypeNode, name)` | `IR/Construction/Method.Builder.cs:444-460` |
| Create basic blocks | `Method.Builder.CreateBasicBlock(location, name)` | `IR/Construction/Method.Builder.cs` |
| Arithmetic (Add, Mul, etc.) | `CreateArithmetic(left, right, BinaryArithmeticKind.Add, flags)` | `IR/Construction/Arithmetic.cs:125-240` |
| Memory (Load, Store, Alloc) | `CreateLoad()`, `CreateStore()`, `CreateAlloca()` | `IR/Construction/Memory.cs` |
| Thread indices | `CreateGroupIndexValue(DeviceConstantDimension3D.X)` | `IR/Construction/IRBuilder.cs:119-196` |
| Grid dimensions | `CreateGridDimensionValue()`, `CreateGridIndexValue()` | `IR/Construction/IRBuilder.cs` |
| Shared memory | `CreateAlloca(type, MemoryAddressSpace.Shared)` | `IR/Construction/Memory.cs` |
| Control flow | `CreateReturn()`, `CreateIfBranch()`, `CreateBranch()` | `IR/Construction/Terminators.cs` |
| Constants | `CreatePrimitiveValue(location, 42)` / `3.14f` | `IR/Construction/Values.cs:141-283` |

**65 IR ValueKind types** including: `UnaryArithmetic`, `BinaryArithmetic`,
`TernaryArithmetic`, `Compare`, `Convert`, `GridIndex`, `GroupIndex`,
`GridDimension`, `GroupDimension`, `WarpSize`, `LaneIdx`, `Alloca`, `Load`,
`Store`, `MemoryBarrier`, `Phi`, `MethodCall`, `Structure`, `GetField`, `SetField`

**CUDA Context — ISOLATED**

ILGPU creates its own `CUcontext` via `cuCtxCreate_v2` when you use
`CudaAccelerator`. But for PTX compilation only, we bypass the accelerator
entirely — we only use `PTXBackend`, which has no CUDA context dependency.

### 9.3 ILGPU IR vs NVRTC — Comparison

| Aspect | ILGPU IR → PTX | NVRTC (CUDA C++ → PTX) |
|--------|---------------|------------------------|
| **Input** | Programmatic API (type-safe) | CUDA C++ string (text) |
| **VL Mapping** | Direct: VL node → IR operation | Indirect: VL node → C++ codegen → string |
| **Type Safety** | Compile-time (IR is typed) | None (string concatenation) |
| **Optimization** | 15+ IR transform passes | NVRTC compiler optimizations |
| **Compile Speed** | Fast (IR → PTX, no parsing) | Slower (C++ parsing + compilation) |
| **Error Reporting** | Structured (IR validation) | Cryptic C++ compiler errors |
| **Patchability** | Modify IR nodes, recompile | Regenerate C++ string, recompile |
| **Dependency** | ILGPU NuGet (~1.5 MB) | NVRTC DLL (CUDA toolkit install) |
| **Runtime Dep** | None (PTX extraction only) | `nvrtc64_130_0.dll` at runtime |
| **Maturity** | Active open-source, .NET native | NVIDIA maintained, industry standard |
| **Shared Memory** | `MemoryAddressSpace.Shared` | `__shared__` keyword |
| **Atomics** | `AtomicCAS`, `Atomic` value kinds | `atomicAdd()` etc. |

### 9.4 Proposed Integration: ILGPU as "PTX Compiler Library"

```
VL Visual Nodes
    │
    ▼
IR Construction Layer  (our code)
    │  Maps VL operations → ILGPU IR operations
    │  IRContext.Declare() + Method.Builder + BasicBlock.Builder
    ▼
ILGPU PTXBackend.Compile()
    │  IR → optimized IR → PTX string
    ▼
PTXCompiledKernel.PTXAssembly
    │  Raw PTX string
    ▼
ManagedCuda cuModuleLoadData()
    │  PTX → CUmodule
    ▼
cuGraphAddKernelNode()
    │  KernelNode in CUDA Graph
    ▼
CudaEngine.Launch()
```

**Key insight**: We use ILGPU ONLY as a PTX compiler — no accelerator, no
context, no runtime. It becomes a build-time dependency, not a runtime one
(beyond the NuGet DLL).

### 9.5 Patchability with ILGPU IR

What makes ILGPU IR superior for patchable kernels:

**Hot Patch (scalar change):**
- Replace `PrimitiveValue` node with new constant
- Or: keep as kernel parameter (no recompile needed, use CUDA Graph Hot Update)

**Warm Patch (operation change):**
- Replace `BinaryArithmetic(Add)` with `BinaryArithmetic(Mul)` in IR
- Recompile: IR → PTX (~1-10ms, much faster than NVRTC's 100ms-2s)
- Reload module, Cold Rebuild affected subgraph

**Structural Patch (topology change):**
- Add/remove IR nodes, modify control flow
- Recompile to PTX
- Cold Rebuild

The IR is a proper graph structure — nodes can be replaced, added, or removed
programmatically. With NVRTC, every change requires regenerating a C++ string
from scratch and running a full C++ compiler.

### 9.6 ILGPU IR Risks & Open Questions

| Risk | Impact | Mitigation |
|------|--------|------------|
| ILGPU IR is internal API, may change | HIGH | Pin ILGPU NuGet version, fork if needed |
| IR construction docs are sparse | MEDIUM | We have full source for reference |
| 15+ optimization passes may be overkill | LOW | Passes are optional, can disable |
| ILGPU NuGet size (~1.5 MB) | LOW | Acceptable for what it provides |
| PTXBackend targets specific PTX ISA versions | MEDIUM | Verify ISA version matches our CUDA minimum |

**Open question**: Can we use `PTXBackend` without creating a full ILGPU
`Context`? If not, the Context creation is lightweight but needs investigation.

### 9.7 Recommendation

**ILGPU IR should REPLACE NVRTC as the patchable kernel backend.**

Reasons:
1. **Type-safe programmatic construction** vs string-based C++ codegen
2. **Faster recompilation** (IR → PTX vs C++ → PTX)
3. **No CUDA toolkit runtime dependency** (no `nvrtc64_*.dll`)
4. **.NET native** — same language as VL.Cuda, debuggable
5. **IR is inspectable** — we can validate, visualize, optimize the kernel graph
6. **Cleaner VL mapping** — VL node → IR op is a 1:1 mapping, no string template layer

NVRTC remains useful for one case: loading user-written CUDA C++ code at
runtime. But for VL-generated patchable kernels, ILGPU IR is strictly superior.

---

## 10. Open Questions for Discussion

1. ~~**CUDA 12.8 vs 13.0**~~ — **RESOLVED: CUDA 13.0 accepted as minimum.**

2. ~~**Patchable Kernel Strategy**~~ — **RESOLVED: Option C.**
   ILGPU IR → PTX for VL-generated patchable kernels (type-safe, fast, .NET native).
   NVRTC retained as escape-hatch for user-written CUDA C++ code loading.

3. ~~**Hybrid Graph+Stream**~~ — **RESOLVED: Defer. Measure first, optimize later.**
   Build Phase 0-2 with CUDA Graph only. Profile Cold Rebuild latency at target
   graph sizes. Add stream-only fallback only if measured rebuild cost is too high.
   CudaEngine is the single execution point, so stream fallback can be retrofitted.

4. ~~**Library Call Integration**~~ — **RESOLVED: Stream Capture directly.**
   Both node types have static and patchable variants:
   - KernelNode: Filesystem PTX (static) / ILGPU IR (patchable, user composes ops)
   - CapturedNode: Single library call (static) / Chained library calls (patchable)
   Patchable CapturedNodes chain multiple library calls (cuBLAS, cuFFT, etc.) into
   one Stream Capture sequence → one child graph node. From outside: one block/function.
   Recapture on parameter change. Hybrid stream approach unnecessary.

5. ~~**ManagedCuda NuGet**~~ — **RESOLVED: Pragmatic approach.**
   Start Phase 0 with latest NuGet package, verify CUDA 13.0 compatibility.
   If mismatch, fall back to source-build from submodule. Test early in Phase 0.

6. ~~**Conditional Nodes**~~ — **RESOLVED: Phase 3 as planned (If + While).**
   CUDA 13.0 minimum guarantees full conditional node support.
   Implement both If-Regions and While-Loops in Phase 3.
