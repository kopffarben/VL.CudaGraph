# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VL / vvvv gamma                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    VL.Cuda.Core                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│   │
│  │  │ Block System │  │Graph Compiler│  │ Profiling &          ││   │
│  │  │              │  │              │  │ Diagnostics          ││   │
│  │  │ ICudaBlock   │  │ Validation   │  │ Timing (async GPU)   ││   │
│  │  │ BlockBuilder │  │ TopoSort     │  │ IVLRuntime (errors)  ││   │
│  │  │ Composition  │  │ CUDA Build   │  │ ToString (tooltips)  ││   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘│   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐│   │
│  │  │ Buffer Pool  │  │ PTX Loader   │  │   Handle System      ││   │
│  │  │              │  │              │  │                      ││   │
│  │  │ GpuBuffer<T> │  │ Parser       │  │ InputHandle<T>       ││   │
│  │  │ AppendBuffer │  │ ModuleCache  │  │ OutputHandle<T>      ││   │
│  │  │ Interop only:│  │ Descriptors  │  │ PinGroups            ││   │
│  │  │ RefCounting  │  │              │  │                      ││   │
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

**CudaContext** is a facade over event-coupled internal services:
- Public API: `RegisterBlock()`, `UnregisterBlock()`, `Connect()`, `Disconnect()`
- Internal: BlockRegistry (fires StructureChanged events)
- Internal: ConnectionGraph (fires StructureChanged events)
- Internal: DirtyTracker (subscribes to structure events)
- Internal: BufferPool, ModuleCache, DeviceContext
- See `CORE-RUNTIME.md` for the event-based coupling design

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
        ctx.RegisterBlock(this);   // public facade method
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
        ├── NodeContext (identity, first ctor parameter by VL convention)
        ├── IVLRuntime.AddMessage() (compilation errors → node tooltips)
        ├── AppHost.TakeOwnership() (lifetime guarantee at shutdown)
        ├── ServiceRegistry (app-global: DeviceInfo, DriverVersion)
        ├── IResourceProvider<T> (Stride interop boundary only)
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
| `CudaContext.cs` | Facade: public API over event-coupled internal services |
| `GpuBuffer.cs` | Type-safe GPU memory wrapper |
| `BufferPool.cs` | Memory pooling with power-of-2 buckets |
| `GraphCompiler.cs` | Converts description to CUDA Graph |
| `PTXLoader.cs` | Parses PTX, extracts kernel info |
| `BlockBuilder.cs` | DSL for block construction |

## CUDA Requirements

**Minimum: CUDA 12.8 / Driver ≥ 570.x / Compute Capability 7.5 (RTX 20xx+)**

We target CUDA 12.8 as hard minimum to use the full CUDA Graph feature set
without version-gating or fallback paths:

| Feature | Available Since | Status |
|---------|----------------|--------|
| Basic Graph API | 10.0 | ✅ included |
| Graph Update (parameters) | 10.2 | ✅ included |
| Memory allocation in graph | 11.4 | ✅ included |
| Conditional nodes (If/While) | 12.4 | ✅ included |
| Improved conditionals | 12.8 | ✅ included |

### Supported GPUs

| Generation | Architecture | Compute Capability | Supported |
|-----------|-------------|-------------------|:---------:|
| RTX 20xx | Turing | 7.5 | ✅ |
| RTX 30xx | Ampere | 8.6 | ✅ |
| RTX 40xx | Ada Lovelace | 8.9 | ✅ |
| RTX 50xx | Blackwell | 10.0 | ✅ |
| GTX 10xx | Pascal | 6.1 | ❌ |
| Older | — | < 6.1 | ❌ |
