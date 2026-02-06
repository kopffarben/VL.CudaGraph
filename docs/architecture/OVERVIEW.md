# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VL / vvvv gamma                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    VL.Cuda.Core                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│   │
│  │  │ Block System │  │Graph Compiler│  │   Debug System       ││   │
│  │  │              │  │              │  │                      ││   │
│  │  │ ICudaBlock   │  │ Validation   │  │ Timing, Readback,    ││   │
│  │  │ BlockBuilder │  │ TopoSort     │  │ Structure Inspect    ││   │
│  │  │ Composition  │  │ CUDA Build   │  │                      ││   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘│   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│   │
│  │  │ Buffer Pool  │  │ PTX Loader   │  │   Handle System      ││   │
│  │  │              │  │              │  │                      ││   │
│  │  │ GpuBuffer<T> │  │ Parser       │  │ InputHandle<T>       ││   │
│  │  │ AppendBuffer │  │ ModuleCache  │  │ OutputHandle<T>      ││   │
│  │  │ RefCounting  │  │ Descriptors  │  │ PinGroups            ││   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘│   │
│  │  ┌──────────────────────────────────────────────────────────┐│   │
│  │  │ Execution Model                                          ││   │
│  │  │                                                          ││   │
│  │  │ CudaEngine (active)  ←→  Blocks (passive)               ││   │
│  │  │ Dirty-Tracking: Hot / Warm / Cold                        ││   │
│  │  │ See: EXECUTION-MODEL.md                                  ││   │
│  │  └──────────────────────────────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────┐  ┌────────────────────────────────────┐ │
│  │   VL.Cuda.Stride       │  │   VL.Cuda.Libraries (later)        │ │
│  │   (Graphics Interop)   │  │   (cuFFT, cuBLAS, cuRAND, NPP)     │ │
│  └────────────────────────┘  └────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │         ManagedCuda           │
                    │    (CUDA Driver API Bindings) │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │      NVIDIA CUDA Runtime      │
                    │      (Driver, GPU Hardware)   │
                    └───────────────────────────────┘
```

## Execution Model

> Detailed in `EXECUTION-MODEL.md`

VL.Cuda uses a centralized execution model with three actors:

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Emitter  │───▶│ Forces   │───▶│Integrate │   Passive Blocks
│(Process) │    │(Process) │    │(Process) │   describe GPU work,
│          │    │          │    │          │   show debug tooltips
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     │  CudaContext   │  (shared)     │
     └───────────────┼───────────────┘
                     │
                     ▼
             ┌──────────────┐
             │  CudaEngine  │   Active Engine
             │  (Process)   │   compiles + launches
             │  Update()    │   the CUDA Graph
             └──────────────┘
```

**Blocks** are passive VL ProcessNodes:
- Constructor: `Setup()` defines kernels, pins, connections → registers with CudaContext
- `Update()`: pushes parameter changes, reads DebugInfo for tooltips
- `Dispose()`: unregisters from CudaContext

**CudaEngine** is the single active ProcessNode:
- `Update()` every frame: collect params → dirty-check → rebuild or update → launch → distribute debug
- The ONLY component that does GPU work

**CudaContext** is internal shared state:
- Block registry, connection tracking, buffer pool
- Three-level dirty tracking: Hot (scalars), Warm (pointers), Cold (structure)

### Update Levels

| Level | Cost | Trigger | Action |
|-------|------|---------|--------|
| **Hot** | ~0 | Scalar value changed | `cuGraphExecKernelNodeSetParams` |
| **Warm** | Cheap | Buffer pointer changed, grid size changed | `cuGraphExecKernelNodeSetParams` |
| **Cold** | Expensive | Node/edge added/removed, Hot-Swap, PTX reload | Full `cuGraphInstantiate` |

Cold rebuilds happen during development (user editing the VL patch). In an exported application, the graph structure is typically stable and only Hot/Warm updates occur.

---

## Data Flow

```
Build-Time (Python/Triton):

    Triton Kernel (.py)
           │
           ▼
    triton.compile()
           │
           ▼
    ┌──────┴──────┐
    │    .ptx     │  +  kernel_name.json (metadata)
    └─────────────┘


Runtime (C#/VL):

    PTX + JSON Files
           │
           ▼
    ┌─────────────────┐
    │   PTX Loader    │
    │   ModuleCache   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐       ┌─────────────────┐
    │   ICudaBlock    │──────▶│   BlockBuilder  │
    │   (passive)     │       │   (Setup)       │
    └────────┬────────┘       └─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  CudaContext    │
    │  (Graph State)  │
    │  (Dirty Track)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  CudaEngine     │
    │  (active node)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Graph Compiler  │
    │ - Validation    │
    │ - Buffer Alloc  │
    │ - Topo Sort     │
    │ - CUDA Build    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ CompiledGraph   │
    │ (CUgraphExec)   │
    └────────┬────────┘
             │
             ▼
         GPU Launch
             │
             ▼
      Debug Info → back to Blocks (tooltips)
```

## Core Concepts

### 1. Everything is a Region

The graph is hierarchically organized as nested regions:

```
Root (GraphDescription)
├── Kernel nodes
├── Memset nodes  
├── Child Regions
│   ├── If (Conditional)
│   │   └── Body (sub-graph)
│   ├── IfElse (Conditional with else)
│   │   ├── TrueBody
│   │   └── FalseBody
│   ├── While (Loop)
│   │   └── Body (executed while condition true)
│   └── For (Bounded loop)
│       └── Body
└── Delegate (subgraph boundary)
```

### 2. Buffers Stay on GPU

```
External Buffer          Graph Buffer           Region Buffer
(User provides)          (Lives with graph)     (Temporary)

     │                        │                       │
     ▼                        ▼                       ▼
┌─────────┐             ┌─────────┐             ┌─────────┐
│ GpuBuf  │             │ GpuBuf  │             │ GpuBuf  │
│ Lifetime│             │ Lifetime│             │ Lifetime│
│=External│             │= Graph  │             │= Region │
└─────────┘             └─────────┘             └─────────┘
     │                        │                       │
     │                        │                       │
 User manages           Released when           Released after
 allocation/free        graph disposed          region completes
```

### 3. Handle-Flow Pattern

In VL, data flows through links. GPU buffers are represented by handles:

```
VL Patch:
                                
   ┌───────────┐         ┌───────────┐
   │  Emitter  │         │  Forces   │
   │           │ Handle  │           │
   │      Out ─┼────────▶┼─ In       │
   └───────────┘         └───────────┘

Behind the scenes:
   
   OutputHandle<GpuBuffer<Particle>> flows through VL link
                    │
                    ▼
   InputHandle<GpuBuffer<Particle>> receives it
                    │
                    ▼
   Graph Compiler sees the connection
   and creates edge between kernel nodes
```

### 4. Blocks Compose (Passively)

Complex GPU operations are built from simpler blocks. Blocks only describe the structure — they never execute:

```csharp
public class ParticleSystemBlock : ICudaBlock
{
    public ParticleSystemBlock(CudaContext ctx)
    {
        // Setup is declarative — no GPU work
        var builder = new BlockBuilder(ctx, this);
        
        var emitter = builder.AddChild<SphereEmitterBlock>();
        var forces = builder.AddChild<ForcesBlock>();
        var integrate = builder.AddChild<IntegrateBlock>();
        
        builder.ConnectChildren(emitter.Particles, forces.Particles);
        builder.ConnectChildren(forces.Particles, integrate.Particles);
        
        builder.ExposeInput("Config", emitter.Config);
        builder.ExposeOutput("Particles", integrate.Particles);
        
        builder.Commit();
        ctx.RegisterBlock(this);
    }
}
```

### 5. Centralized Execution

The CudaEngine is the only active component. This ensures:
- The CUDA Graph is always compiled and launched as one atomic unit
- No race conditions between blocks trying to access the GPU
- Clear execution order: collect → compile → launch → debug
- Predictable frame timing

## Module Dependencies

```
VL.Cuda.Core
    │
    ├── ManagedCuda (NuGet)
    │   └── CUDA Driver API
    │
    └── VL.Core
        └── PinGroups, TypeRegistry

VL.Cuda.Libraries (optional)
    │
    ├── VL.Cuda.Core
    │
    └── ManagedCuda libraries
        ├── ManagedCuda.CUBLAS
        ├── ManagedCuda.CUFFT
        ├── ManagedCuda.CURAND
        └── ManagedCuda.NPP

VL.Cuda.Stride (optional)
    │
    ├── VL.Cuda.Core
    │
    └── VL.Stride
        └── DirectX 11 interop
```

## Key Files

| File | Purpose |
|------|---------|
| `*.ptx` | Compiled CUDA kernel code |
| `*.json` | Kernel metadata (parameters, types, hints) |
| `CudaEngine.cs` | Active ProcessNode — compiles and launches graph |
| `CudaContext.cs` | Shared state, dirty-tracking, block registry |
| `GpuBuffer.cs` | Type-safe GPU memory wrapper |
| `BufferPool.cs` | Memory pooling with power-of-2 buckets |
| `GraphCompiler.cs` | Converts description to CUDA Graph |
| `PTXLoader.cs` | Parses PTX, extracts kernel info |
| `BlockBuilder.cs` | DSL for block construction |

## CUDA Version Features

| Feature | Minimum CUDA |
|---------|--------------|
| Basic Graph API | 10.0 |
| Graph Update (parameters) | 10.2 |
| Memory allocation in graph | 11.4 |
| Conditional nodes (If/While) | 12.4 |
| Improved conditionals | 12.8 |
