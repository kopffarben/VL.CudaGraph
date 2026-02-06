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
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────┐  ┌────────────────────────────────────┐ │
│  │   VL.Cuda.Stride       │  │   VL.Cuda.Libraries (später)       │ │
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
    │   (VL Node)     │       │   (Setup)       │
    └────────┬────────┘       └─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  CudaContext    │
    │  (Graph Model)  │
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

### 4. Blocks Compose

Complex GPU operations are built from simpler blocks:

```csharp
public class ParticleSystemBlock : ICudaBlock
{
    public void Setup(BlockBuilder builder)
    {
        // Add child blocks
        var emitter = builder.AddChild<SphereEmitterBlock>();
        var forces = builder.AddChild<ForcesBlock>();
        var integrate = builder.AddChild<IntegrateBlock>();
        
        // Connect children
        builder.ConnectChildren(emitter.Particles, forces.Particles);
        builder.ConnectChildren(forces.Particles, integrate.Particles);
        
        // Expose to outside
        builder.ExposeInput("Config", emitter.Config);
        builder.ExposeOutput("Particles", integrate.Particles);
    }
}
```

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
| `CudaContext.cs` | Main entry point, graph management |
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
