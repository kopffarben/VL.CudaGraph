# VL.Cuda — CUDA Graph Integration for vvvv gamma

## Project Overview

VL.Cuda brings GPU-accelerated computing to vvvv gamma through NVIDIA's CUDA Graph API. The system allows visual programmers to build and execute GPU compute pipelines using a node-based interface, with kernels authored in Triton (Python) and compiled to PTX.

### Core Philosophy

```
Everything stays on GPU — no readback unless explicitly requested
Visual dataflow — mutations are invisible, data flows through links
Composition over inheritance — blocks combine, not inherit
```

### Technology Stack

```
Triton (Python)     → PTX generation (compile-time)
PTX + JSON          → Kernel metadata + entry points
ManagedCuda         → .NET CUDA bindings
VL.Cuda.Core        → Graph compiler, buffer pool, blocks
VL.Stride           → Graphics interop (DX11)
```

## Quick Start for Implementation

### Phase 0: Foundation
1. Read `docs/architecture/CORE-RUNTIME.md` — CudaContext, CudaStream, BufferPool
2. Read `docs/architecture/PTX-LOADER.md` — How PTX files are loaded and parsed
3. Implement: `CudaContext`, `GpuBuffer<T>`, `BufferPool`

### Phase 1: Graph Basics
1. Read `docs/architecture/GRAPH-COMPILER.md` — How graphs are built
2. Implement: `KernelNode`, `Edge`, `GraphCompiler`, `CompiledGraph`

### Phase 2: VL Integration
1. Read `docs/architecture/VL-INTEGRATION.md` — Handle-flow pattern
2. Read `docs/architecture/BLOCK-SYSTEM.md` — ICudaBlock, BlockBuilder
3. Implement: `ICudaBlock`, `BlockBuilder`, PinGroups integration

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
    CORE-RUNTIME.md                    ← Buffer, Context, Stream
    GRAPH-COMPILER.md                  ← Graph building & execution
    VL-INTEGRATION.md                  ← VL-specific patterns
    BLOCK-SYSTEM.md                    ← ICudaBlock, composition
    PTX-LOADER.md                      ← PTX parsing, Triton workflow
    GRAPHICS-INTEROP.md                ← DX11/Stride integration
  api/
    CSHARP-API.md                      ← Complete C# API reference
  implementation/
    PHASES.md                          ← Implementation roadmap
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CUDA Graph API (not streams) | Reduced launch overhead, graph-level optimization |
| Handle-flow through VL links | Visual dataflow, no invisible mutations |
| PTX from Triton | High-level kernel authoring, automatic optimization |
| Composition (not inheritance) | VL doesn't support inheritance well |
| Buffer pool with power-of-2 | Fast allocation, predictable memory |
| JSON alongside PTX | Human-readable metadata, editable without recompile |

## Dependencies

```
ManagedCuda          — CUDA driver API bindings
VL.Core              — PinGroups, TypeRegistry
VL.Stride (optional) — Graphics interop
```

## CUDA Version Requirements

- **Minimum:** CUDA 12.4 (for Conditional Nodes in Graphs)
- **Recommended:** CUDA 12.8+ (improved conditional node features)
- **Compute Capability:** 7.0+ (Volta or newer)

## File Conventions

```
*.ptx       — Compiled CUDA kernels (from Triton)
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
