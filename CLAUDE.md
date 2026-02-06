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
ManagedCuda         → .NET CUDA bindings
VL.Cuda.Core        → Graph compiler, buffer pool, blocks
VL.Stride           → Graphics interop (DX11)
```

PTX can be produced by any toolchain:
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
2. Implement: `KernelNode`, `Edge`, `GraphCompiler`, `CompiledGraph`

### Phase 2: Execution Model & VL Integration
1. Read `docs/architecture/EXECUTION-MODEL.md` — **How blocks, engine, and context interact**
2. Read `docs/architecture/VL-INTEGRATION.md` — Handle-flow pattern
3. Read `docs/architecture/BLOCK-SYSTEM.md` — ICudaBlock, BlockBuilder
4. Implement: `CudaEngine`, `ICudaBlock`, `BlockBuilder`, dirty-tracking

### Phase 3: Advanced
1. Read `docs/architecture/GRAPH-COMPILER.md#regions` — Conditional nodes
2. Implement: AppendBuffer, Regions (If, While), Liveness analysis

### Phase 4: Interop
1. Read `docs/architecture/GRAPHICS-INTEROP.md` — DX11/Stride sharing
2. Implement: CUDA-DX11 buffer/texture sharing

## Documentation Structure

```
CLAUDE.md                              ← You are here
docs/
  architecture/
    OVERVIEW.md                        ← High-level architecture
    EXECUTION-MODEL.md                 ← Block/Engine lifecycle, dirty-tracking ★ NEW
    CORE-RUNTIME.md                    ← Buffer, Context, Stream
    GRAPH-COMPILER.md                  ← Graph building & execution
    VL-INTEGRATION.md                  ← VL-specific patterns
    BLOCK-SYSTEM.md                    ← ICudaBlock, composition
    PTX-LOADER.md                      ← PTX parsing, kernel loading
    GRAPHICS-INTEROP.md                ← DX11/Stride integration
  api/
    CSHARP-API.md                      ← Complete C# API reference
  implementation/
    PHASES.md                          ← Implementation roadmap
src/
  References/                          ← READ-ONLY git submodules
    VL.StandardLibs/                   ← VL.Core source (API reference)
    managedCuda/                       ← ManagedCuda source (API reference)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CUDA Graph API (not streams) | Reduced launch overhead, graph-level optimization |
| Centralized execution (CudaEngine) | CUDA Graph is atomic — must compile and launch as one unit |
| Passive blocks (no GPU work) | Blocks are descriptions; Engine does all GPU work |
| Three-level dirty tracking | Hot/Warm/Cold updates minimize rebuild cost |
| Handle-flow through VL links | Visual dataflow, no invisible mutations |
| PTX-agnostic runtime | Consumes PTX + JSON; source toolchain is user's choice |
| Composition (not inheritance) | VL doesn't support inheritance well |
| Buffer pool with power-of-2 | Fast allocation, predictable memory |
| JSON alongside PTX | Human-readable metadata, editable without recompile |
| Event-based service coupling | BlockRegistry/ConnectionGraph fire events, DirtyTracker subscribes |
| NodeContext as first constructor param | VL convention: auto-injected identity + logging |
| IVLRuntime for diagnostics | Native VL error/warning display on nodes, no custom system |
| AppHost.TakeOwnership | Ensures CudaEngine cleanup on app shutdown |
| ResourceProvider at Stride boundary only | Internal: raw handles. External: VL-compatible lifetime |
| DeltaTime as normal VL pin | No special CUDA handling, user connects FrameClock |

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

Update Levels:
    Hot    (scalar changed)         → cuGraphExecKernelNodeSetParams, ~0 cost
    Warm   (pointer/grid changed)   → cuGraphExecKernelNodeSetParams, cheap
    Cold   (structure changed)      → Full graph rebuild, expensive but OK during development

Diagnostics:
    Errors/Warnings  → IVLRuntime.AddMessage() → VL node colors (red/orange)
    Timing/Stats     → ToString() / DebugInfo  → VL tooltips (on hover)
```

## Dependencies

```
ManagedCuda          — CUDA driver API bindings
VL.Core              — NodeContext, IVLRuntime, AppHost, ResourceProvider, PinGroups
VL.Stride (optional) — Graphics interop (ResourceProvider boundary)
```

## Reference Submodules (READ-ONLY)

```
src/References/
  VL.StandardLibs/     — VL.Core, VL.Stride, etc. (git submodule)
  managedCuda/         — ManagedCuda CUDA bindings (git submodule)
```

**Purpose:** These submodules exist solely as API reference for development.
Search the source code to understand how VL.Core APIs (NodeContext, AppHost,
ResourceProvider, IVLRuntime, etc.) and ManagedCuda APIs actually work.

**Rules:**
- ⛔ **NEVER modify any file** in `src/References/` — these are upstream repos
- ✅ **Search and read** to understand API signatures, patterns, conventions
- ✅ **Our code** (`src/VL.Cuda.Core/`, etc.) consumes these as **NuGet packages**
- The submodules are not project references — they are not compiled as part of our solution

## CUDA Requirements

- **Minimum:** CUDA 12.8 / Driver ≥ 570.x (hard requirement, no fallback paths)
- **Compute Capability:** 7.5+ (Turing / RTX 20xx or newer)
- Full CUDA Graph feature set including improved conditional nodes
- No version-gating in code — all features assumed available

## File Conventions

```
*.ptx       — Compiled CUDA kernels (from Triton, nvcc, or any PTX source)
*.json      — Kernel metadata (beside PTX, same name)
*.vl        — VL patches using VL.Cuda
```

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
- Dirty-tracking correctness
- Cold rebuild / warm update paths
