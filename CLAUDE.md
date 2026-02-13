# VL.Cuda — CUDA Graph Integration for vvvv gamma

## Project Overview

VL.Cuda brings GPU-accelerated computing to vvvv gamma through NVIDIA's CUDA Graph API. The system allows visual programmers to build and execute GPU compute pipelines using a node-based interface. Kernels are consumed as PTX + JSON metadata — the runtime is agnostic to how PTX is produced.

### Core Philosophy

```
Everything stays on GPU — no readback unless explicitly requested
Visual dataflow — mutations are invisible, data flows through links
Composition over inheritance — blocks combine, not inherit
Centralized execution — one Engine launches the entire CUDA Graph
Passive blocks — blocks describe work, they don't execute it
```

### Technology Stack

```
PTX + JSON          → Kernel binary + metadata (runtime input)
ILGPU IR → PTX      → Patchable kernel compilation (primary, type-safe, ~1-10ms)
NVRTC               → User CUDA C++ compilation (escape-hatch)
Stream Capture      → Library Call integration (cuBLAS, cuFFT, cuDNN)
ManagedCuda         → .NET CUDA bindings (CUDA 13.0 minimum)
ILGPU               → PTX backend only (no accelerator, no CUDA context)
VL.Cuda.Core        → Graph compiler, buffer pool, blocks
VL.Stride           → Graphics interop (DX11)
```

### Three Kernel Sources → Two Node Types (Static + Patchable)

The system supports three kernel sources that map to two CUDA Graph node types. Each node type has a static and a patchable variant:

```
1. Filesystem PTX  (Triton, nvcc, hand-written)  ─┐
                                                   ├→ KernelNode (Hot/Warm/Code/Cold)
2. Patchable Kernel (VL Node-Set → ILGPU IR → PTX)─┘  Static: from file
   + NVRTC escape-hatch for user CUDA C++              Patchable: from ILGPU IR

3. Library Call (cuBLAS, cuFFT, cuDNN)              → CapturedNode (Recapture/Cold)
   + Patchable Captured (chained library calls)        Static: single call
                                                       Patchable: chained sequence
```

See `docs/architecture/KERNEL-SOURCES.md` for the full design.

Filesystem PTX can be produced by any toolchain:
- **Triton** (Python) — recommended for rapid prototyping, auto-tuning
- **CUDA C/C++** (nvcc) — full control, industry standard
- **Hand-written PTX** — maximum control, niche use
- **Numba** (Python) — lightweight Python → PTX path

## Quick Start for Implementation

### Phase 0: Foundation
1. Read `docs/architecture/CORE-RUNTIME.md` — CudaContext, CudaStream, BufferPool
2. Read `docs/architecture/PTX-LOADER.md` — How PTX files are loaded and parsed
3. Implement: `CudaContext`, `GpuBuffer<T>`, `BufferPool`

### Phase 1: Graph Basics
1. Read `docs/architecture/GRAPH-COMPILER.md` — How graphs are built
2. Read `docs/architecture/KERNEL-SOURCES.md` — Three kernel sources, two node types
3. Implement: `KernelNode`, `Edge`, `GraphCompiler`, `CompiledGraph`

### Phase 2: Execution Model & VL Integration
1. Read `docs/architecture/EXECUTION-MODEL.md` — **How blocks, engine, and context interact**
2. Read `docs/architecture/VL-INTEGRATION.md` — Handle-flow pattern
3. Read `docs/architecture/BLOCK-SYSTEM.md` — ICudaBlock, BlockBuilder
4. Implement: `CudaEngine`, `ICudaBlock`, `BlockBuilder`, dirty-tracking

### Phase 3: Advanced
1. Read `docs/architecture/GRAPH-COMPILER.md#regions` — Conditional nodes
2. Implement: AppendBuffer, Regions (If, While), Liveness analysis

### Phase 4a: Library Calls (Stream Capture)
1. Read `docs/architecture/KERNEL-SOURCES.md` — CapturedNode, static + patchable
2. Implement: `StreamCaptureHelper`, `CapturedNodeDescriptor`, `AddCaptured()`
3. Implement: `LibraryHandleCache`, cuBLAS/cuFFT wrappers
4. Implement: Patchable CapturedNode (chained library call sequences)

### Phase 4b: Patchable Kernels (ILGPU IR + NVRTC)
1. Read `docs/architecture/KERNEL-SOURCES.md` — ILGPU IR vs NVRTC comparison
2. Implement: `IlgpuCompiler` (IR → PTX → CUmodule, with caching)
3. Implement: IR construction layer (VL node-set → ILGPU IR)
4. Implement: `NvrtcCache` (escape-hatch for user CUDA C++)

### Phase 5: Graphics Interop
1. Read `docs/architecture/GRAPHICS-INTEROP.md` — DX11/Stride sharing
2. Implement: CUDA-DX11 buffer/texture sharing

### Phase 6: Device Graph Launch & Dispatch Indirect
1. Read `docs/architecture/GRAPH-COMPILER.md#device-graph-launch--dispatch-indirect` — Three levels of dynamic work
2. Implement: AppendBuffer Counter as GPU pointer pin
3. Implement: DispatcherNode (built-in PTX, reads counter, sets target grid via `cudaGraphKernelNodeSetParam`)
4. Implement: DeviceLaunch graph instantiation + `cuGraphUpload()` integration

## Documentation Structure

```
CLAUDE.md                              ← You are here
docs/
  architecture/
    OVERVIEW.md                        ← High-level architecture
    EXECUTION-MODEL.md                 ← Block/Engine lifecycle, dirty-tracking
    CORE-RUNTIME.md                    ← Buffer, Context, Stream
    GRAPH-COMPILER.md                  ← Graph building & execution
    VL-INTEGRATION.md                  ← VL-specific patterns
    BLOCK-SYSTEM.md                    ← ICudaBlock, composition
    KERNEL-SOURCES.md                  ← Three sources, two node types, static + patchable
    PTX-LOADER.md                      ← PTX file parsing, kernel loading
    GRAPHICS-INTEROP.md                ← DX11/Stride integration
  api/
    CSHARP-API.md                      ← Complete C# API reference
  implementation/
    PHASES.md                          ← Implementation roadmap
  vl.reference/                        ← VL platform reference (for agents)
    VL.Overview.md                     ← Library dependency graph, key concepts
    VL.Compiler.md                     ← vvvvc.exe CLI compiler, .vl validation
    VL.Fileformat/
      VL.Fileformat.Description.md     ← .vl XML format spec (IDs, elements, patterns)
      VL.Fileformat.BestPractices.md   ← Layout conventions, spacing, sizing
    VL.StandardLibs/
      VL.CoreLib.md                    ← Foundation: types, Spread, math, animation, reactive
      VL.Stride.md                     ← 3D rendering, ECS, shaders, post-FX
      VL.Skia.md                       ← 2D rendering, layers, paint, input
      VL.ImGui.md                      ← Immediate-mode GUI, widgets, channels
      VL.Serialization.md              ← XML, FSPickler, MessagePack, Raw
      VL.Video.md                      ← Video playback, camera capture
      VL.IO.md                         ← File, HTTP, Redis, OSCQuery, Pipes, Dataflow
src/
  References/                          ← READ-ONLY git submodules
    VL.StandardLibs/                   ← VL.Core source (API reference)
    managedCuda/                       ← ManagedCuda source (API reference)
    ILGPU/                             ← ILGPU compiler source (API reference)
    The-Gray-Book/                     ← VL documentation, design guidelines
    stride/                            ← Stride 3D engine source (Graphics Interop reference)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CUDA Graph API (not streams) | Reduced launch overhead, graph-level optimization |
| Centralized execution (CudaEngine) | CUDA Graph is atomic — must compile and launch as one unit |
| Passive blocks (no GPU work) | Blocks are descriptions; Engine does all GPU work |
| Four-level dirty tracking | Hot Update/Warm Update/Code Rebuild/Cold Rebuild for KernelNodes; Recapture/Cold Rebuild for CapturedNodes |
| Handle-flow through VL links | Visual dataflow, no invisible mutations |
| PTX-agnostic runtime | Consumes PTX + JSON; source toolchain is user's choice |
| Three kernel sources, two node types | Each with static + patchable variant |
| ILGPU IR for patchable kernels | Type-safe, fast (~1-10ms), .NET native, no NVRTC DLL needed |
| NVRTC as escape-hatch | For user-written CUDA C++ code loading at runtime |
| Patchable CapturedNodes | Chained library calls as single block/function |
| Composition (not inheritance) | VL doesn't support inheritance well |
| Buffer pool with power-of-2 | Fast allocation, predictable memory |
| JSON alongside PTX | Human-readable metadata, editable without recompile |
| Event-based service coupling | BlockRegistry/ConnectionGraph fire events, DirtyTracker subscribes |
| NodeContext as first constructor param | VL convention: auto-injected identity + logging |
| IVLRuntime for diagnostics | Native VL error/warning display on nodes, no custom system |
| AppHost.TakeOwnership | Ensures CudaEngine cleanup on app shutdown |
| ResourceProvider at Stride boundary only | Internal: raw handles. External: VL-compatible lifetime |
| DeltaTime as normal VL pin | No special CUDA handling, user connects FrameClock |
| ShaderFX model for patchable kernels | Element-wise, implicit parallelism, GPU type enforcement |
| Three user levels | Consumer (use blocks), Composer (patch GPU nodes), Author (write PTX) |
| Raw CUgraph handles for CapturedNode | ManagedCuda.CudaGraph(CUgraph) constructor is internal in NuGet — use raw handles via DriverAPINativeMethods |
| Flat buffer binding indices | CapturedHandle.In/Out/Scalar return flat index into BufferBindings[]: [inputs..., outputs..., scalars...] |
| ILGPU ArrayView as 16-byte struct | ArrayView<T> compiles to `.param .align 8 .b8[16]` (ptr+length packed), NOT separate params |
| KernelSource discriminated union | FilesystemPtx / IlgpuMethod / NvrtcSource — clean dispatch in CudaEngine.LoadKernelFromSource |
| Code dirty → Cold Rebuild | Simplest correct behavior; partial rebuild is future optimization |
| LibraryHandleCache per CudaContext | Lazy-init expensive library handles (cuBLAS, cuFFT, etc.) — cached by config key for FFT plans |
| Three-level dispatch indirect | Level 1: Max-Grid+early-return (Phase 3), Level 2: Conditional Nodes (Phase 3.2), Level 3: Device-updatable nodes (Phase 6) |
| DispatcherNode as system-provided PTX | Built-in kernel reads GPU counter + calls `cudaGraphKernelNodeSetParam` — users don't write it |
| DeviceLaunch flag for Phase 6 only | Normal graphs don't pay DeviceLaunch restrictions; opt-in when DispatcherNode is present |
| ManagedCuda Device Graph APIs verified | All host-side APIs present (DeviceLaunch flag, cuGraphUpload, DeviceUpdatableKernelNode, ConditionalHandle). Device-side APIs are PTX intrinsics |

## Execution Model Summary

```
Blocks = passive ProcessNodes
    → Constructor(NodeContext, CudaContext): define kernels, pins, register
    → Update: read DebugInfo for VL tooltips, push parameter changes
    → Dispose: unregister from CudaContext

CudaEngine = active ProcessNode (one per CUDA pipeline)
    → Constructor: creates CudaContext, AppHost.TakeOwnership(this)
    → Update: collect params → dirty-check → rebuild/update → launch → report diagnostics
    → The ONLY component that talks to the GPU
    → Routes errors/warnings to VL nodes via IVLRuntime

CudaContext = facade over event-coupled services
    → BlockRegistry (fires StructureChanged) → DirtyTracker subscribes
    → ConnectionGraph (fires StructureChanged) → DirtyTracker subscribes
    → DirtyTracker, BufferPool, ModuleCache, DeviceContext
    → IlgpuCompiler, NvrtcCache (owned, disposed with context)

Update Levels (KernelNode — Filesystem PTX & Patchable Kernels):
    Hot    (scalar changed)         → cuGraphExecKernelNodeSetParams, ~0 cost
    Warm   (pointer/grid changed)   → cuGraphExecKernelNodeSetParams, cheap
    Code   (ILGPU IR recompile)     → Invalidate cache → Cold Rebuild (full graph; partial is future optimization)
    Cold   (structure changed)      → Full graph rebuild, expensive but OK during development

CudaEngine.Update() priority: Structure > Code > CapturedNodes > Parameters

Update Levels (CapturedNode — Library Calls, static + patchable chained):
    Recapture (param changed)       → Recapture + cuGraphExecChildGraphNodeSetParams, medium
    Cold      (structure changed)   → Full graph rebuild

Diagnostics:
    Errors/Warnings  → IVLRuntime.AddMessage() → VL node colors (red/orange)
    Timing/Stats     → ToString() / DebugInfo  → VL tooltips (on hover)
```

## Dependencies

```
ManagedCuda          — CUDA driver API bindings (CUDA 13.0 minimum)
ILGPU                — PTX backend only (IR → PTX compilation, no accelerator)
VL.Core              — NodeContext, IVLRuntime, AppHost, ResourceProvider, PinGroups
VL.Stride (optional) — Graphics interop (ResourceProvider boundary)
ManagedCuda.CUBLAS   — cuBLAS bindings (optional, for Library Calls via CapturedNode)
ManagedCuda.CUFFT    — cuFFT bindings (optional, for Library Calls via CapturedNode)
ManagedCuda.NVRTC    — NVRTC bindings (optional, escape-hatch for user CUDA C++)
```

## Reference Submodules (READ-ONLY)

```
src/References/
  VL.StandardLibs/     — VL.Core, VL.Stride, etc. (git submodule)
  managedCuda/         — ManagedCuda CUDA bindings (git submodule)
  ILGPU/               — ILGPU .NET GPU compiler (git submodule)
  The-Gray-Book/       — Official VL documentation, design guidelines, best practices (git submodule)
  stride/              — Stride 3D engine source (git submodule, Graphics Interop API reference)
```

**Purpose:** These submodules exist solely as API reference for development.
Search the source code to understand how VL.Core APIs (NodeContext, AppHost,
ResourceProvider, IVLRuntime, etc.), ManagedCuda APIs, and Stride internals
(DX11 interop, resource management, rendering pipeline) actually work. The-Gray-Book
contains official VL design guidelines, naming conventions, and best practices.

**Rules:**
- ⛔ **NEVER modify any file** in `src/References/` — these are upstream repos
- ✅ **Search and read** to understand API signatures, patterns, conventions
- ✅ **Our code** (`src/VL.Cuda.Core/`, etc.) consumes these as **NuGet packages**
- The submodules are not project references — they are not compiled as part of our solution
- ✅ **The-Gray-Book** is VL's official documentation — follow its design guidelines for node/pin naming, categories, documentation, etc.

## CUDA Requirements

- **Minimum:** CUDA 13.0 (hard requirement, aligned with ManagedCuda NVRTC DLL)
- **Compute Capability:** 7.5+ (Turing / RTX 20xx or newer)
- Full CUDA Graph feature set including improved conditional nodes
- No version-gating in code — all features assumed available

## File Conventions

```
*.ptx       — Compiled CUDA kernels (from Triton, nvcc, or any PTX source)
*.json      — Kernel metadata (beside PTX, same name)
*.vl        — VL patches using VL.Cuda
```

### Editing `.vl` Files

When generating or editing `.vl` files, **strictly follow** the specification in
[`docs/vl.reference/VL.Fileformat/VL.Fileformat.Description.md`](docs/vl.reference/VL.Fileformat/VL.Fileformat.Description.md).
This includes: ID encoding (Base64url, 16 bytes), element structure (Document, Patch, Canvas,
Node, Pin, Pad, Link, Fragment, Slot), property serialization, NodeReference/CategoryReference
patterns, and all element kinds. Non-conforming `.vl` files will fail to load in vvvv gamma.

After editing a `.vl` file, **always validate** it by compiling with `vvvvc.exe`:
```bash
tools\vvvv_gamma_7.1-0161-g176d67638c-win-x64\vvvvc.exe VL.CudaGraph.vl ^
  --package-repositories "D:\_MBOX\_CODE\_packages" --verbosity Information
```
For library packages (no entry point), the export step will crash with "Entry point not found" —
this is expected. The validation signal is the compilation line: `VL.CudaGraph.vl -> ...dll`.
See [`docs/vl.reference/VL.Compiler.md`](docs/vl.reference/VL.Compiler.md) for details.

## Pre-Commit Documentation Checklist

**Before every commit, documentation MUST be updated and verified.** Code and docs are always committed together — never commit code changes without updating affected docs.

### Mandatory Steps

1. **Check affected docs** — For every code change, identify which docs are impacted:
   - `docs/api/CSHARP-API.md` — Any new/changed/removed types, methods, properties, namespaces
   - `docs/architecture/EXECUTION-MODEL.md` — Changes to block/engine lifecycle, dirty-tracking, update levels
   - `docs/architecture/BLOCK-SYSTEM.md` — Changes to ICudaBlock, BlockBuilder, composition patterns
   - `docs/architecture/CORE-RUNTIME.md` — Changes to CudaContext, BufferPool, GpuBuffer, DeviceContext
   - `docs/architecture/GRAPH-COMPILER.md` — Changes to graph building, KernelNode, CompiledGraph
   - `docs/architecture/KERNEL-SOURCES.md` — Changes to kernel sources, node types, ILGPU/NVRTC
   - `docs/architecture/VL-INTEGRATION.md` — Changes to VL patterns, handle-flow, ProcessNode
   - `docs/architecture/PTX-LOADER.md` — Changes to PTX/JSON loading
   - `docs/architecture/GRAPHICS-INTEROP.md` — Changes to DX11/Stride interop
   - `docs/implementation/PHASES.md` — Phase status changes, new test counts, completed tasks
   - `CLAUDE.md` — New architecture decisions, design patterns, technology changes

2. **Update docs to match code** — Ensure signatures, class names, namespaces, behavior descriptions, and examples in docs reflect the actual implementation. Mark planned features clearly as *(Phase N+)*.

3. **Verify consistency** — Cross-check:
   - API doc type names match actual C# class/interface names
   - Phase status and test counts in PHASES.md are current
   - Code examples in docs compile against the actual API
   - No references to renamed/removed types or methods

4. **Do NOT update** `docs/vl.reference/` — These are external platform reference docs, not project docs.

### What Counts as "Affected"

- Added a new class → update CSHARP-API.md + relevant architecture doc
- Changed a method signature → update CSHARP-API.md
- Changed behavior/lifecycle → update the architecture doc that describes it
- Completed a phase task → update PHASES.md status
- New architecture decision → update CLAUDE.md "Key Design Decisions" table
- Changed test count → update PHASES.md and CSHARP-API.md header

## Testing

Run tests with:
```bash
dotnet test VL.Cuda.Tests
```

Key test areas:
- PTX parsing correctness
- Graph compilation validation
- Buffer pool allocation/release
- Type compatibility checking
- Dirty-tracking correctness (Hot/Warm/Code/Cold + Recapture)
- Cold rebuild / warm update paths
- ILGPU IR construction and PTX compilation (patchable kernels)
- NVRTC compilation and caching (user CUDA C++ escape-hatch)
- Stream Capture → CapturedNode lifecycle
- Recapture for CapturedNode parameter changes
- Mixed graph execution (KernelNodes + CapturedNodes)
- cuBLAS/cuFFT integration via CapturedNode
