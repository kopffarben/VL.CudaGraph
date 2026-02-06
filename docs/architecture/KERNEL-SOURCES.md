# Kernel Sources & Node Types

## Overview

The system supports **three kernel sources** that map to **two CUDA Graph node types**. Each node type has a **static** and a **patchable** variant. This design keeps the block system uniform while enabling both custom kernels and NVIDIA library operations.

```
KERNEL SOURCES                              NODE TYPE IN GRAPH
══════════════                              ══════════════════

1. Filesystem PTX ──────────────┐
   (Triton, nvcc, hand-written) │           KernelNode (static)
   .ptx + .json files           │
                                ├───→  KernelNode
2. Patchable Kernel ────────────┘      ├─ Hot:       Scalar changed         ~0 cost
   (VL Node-Set → ILGPU IR → PTX)     ├─ Warm:      Pointer changed        cheap
   Live or OnSave compilation          ├─ Code:      ILGPU recompile        fast (~1-10ms)
                                       └─ Cold:      Structure rebuild      expensive

3. Library Call ──────────────────→  CapturedNode (ChildGraphNode)
   (cuBLAS, cuFFT, cuDNN, cuRAND)     ├─ Recapture: Param changed          medium
   Stream Capture                      └─ Cold:      Structure rebuild      expensive

   Patchable Captured ────────────→  CapturedNode (chained library calls)
   (User-composed sequence of            Same update behavior as above,
    library calls → Stream Capture)      but captures the entire chain
```

### Static vs Patchable Variants

| | Static | Patchable |
|--|--------|-----------|
| **KernelNode** | Filesystem PTX (Triton, nvcc, etc.) | ILGPU IR (user composes ops visually in VL) |
| **CapturedNode** | Single library call | Chain of library calls (user composes sequence) |

All four variants are **fully integrated** into the block system. Blocks use `BlockBuilder` to declare their content — the Graph Compiler handles the rest.

---

## Source 1: Filesystem PTX (Static KernelNode)

Pre-compiled PTX files loaded from disk. This is the primary path for custom kernels and the MVP path.

**Toolchain:** Any — Triton (recommended), nvcc, Numba, hand-written PTX.

**Flow:**
```
Build-time:  Kernel source → compile → .ptx + .json
Runtime:     PTX Loader → ModuleCache → CUmodule → CUfunction → KernelNode
```

**BlockBuilder usage:**
```csharp
var kernel = builder.AddKernel("kernels/vector_add.ptx", "vector_add_f32");
builder.Input<float>("A", kernel.In(0));
builder.Output<float>("Sum", kernel.Out(2));
builder.InputScalar<int>("Count", kernel.In(3));
```

**Update behavior:**
- Hot: Scalar values (Count, alpha, beta) — `cuGraphExecKernelNodeSetParams`, ~0 cost
- Warm: Buffer pointers, grid size — `cuGraphExecKernelNodeSetParams`, cheap
- Cold: PTX file changed (FileWatcher) — full graph rebuild for affected block

> See `PTX-LOADER.md` for file format, parsing, and caching details.

---

## Source 2: Patchable Kernels (Patchable KernelNode)

Kernels generated at runtime from a visual node-set in VL. The user patches GPU operations visually, and the system constructs ILGPU IR programmatically, compiles it to PTX, and loads the result as a normal KernelNode.

**NVRTC** is retained as an escape-hatch for users who want to load their own CUDA C++ code at runtime.

### Primary Path: ILGPU IR → PTX

**Flow:**
```
VL Patch:    User edits GPU node-set (GPU.Add, GPU.Mul, GPU.Reduce, ...)
                 │
                 ▼
IR Build:    Node-set → ILGPU IR (programmatic, type-safe)
                 │  IRContext.Declare() + Method.Builder + BasicBlock.Builder
                 ▼
PTX Gen:     PTXBackend.Compile() → PTXCompiledKernel.PTXAssembly (string)
                 │  Fast: ~1-10ms (IR → PTX, no C++ parsing)
                 ▼
Loader:      cuModuleLoadData(ptxString) → CUmodule → CUfunction
                 │
                 ▼
Graph:       Normal KernelNode — same as Filesystem PTX from here on
```

**BlockBuilder usage:**
```csharp
// Patchable kernels use the same AddKernel path,
// but the PTX comes from ILGPU instead of a file.
var kernel = builder.AddKernel(ilgpuModule, "user_kernel", debugName: "MatMulPatch");
builder.Input<float>("A", kernel.In(0));
builder.Input<float>("B", kernel.In(1));
builder.Output<float>("C", kernel.Out(2));
```

**Update behavior:**
- Hot/Warm: Same as Filesystem PTX — full parameter patchability
- Code: User edits the node-set → ILGPU IR rebuild → new PTX → new CUmodule → Cold rebuild of affected block only (not the entire graph)

**Key constraint:** `cuGraphExecKernelNodeSetParams` cannot swap the `CUfunction` itself. A code change always requires a Cold rebuild of the block. However, all parameter updates remain Hot/Warm — only the kernel logic change triggers a rebuild.

### Why ILGPU IR (not NVRTC)

| Aspect | ILGPU IR → PTX | NVRTC (CUDA C++ → PTX) |
|--------|---------------|------------------------|
| **Input** | Programmatic API (type-safe) | CUDA C++ string (text) |
| **VL Mapping** | Direct: VL node → IR operation | Indirect: VL node → C++ codegen → string |
| **Type Safety** | Compile-time (IR is typed) | None (string concatenation) |
| **Compile Speed** | ~1-10ms (IR → PTX, no parsing) | ~100ms-2s (C++ parsing + compilation) |
| **Error Reporting** | Structured (IR validation) | Cryptic C++ compiler errors |
| **Patchability** | Modify IR nodes, recompile | Regenerate C++ string, recompile |
| **Dependency** | ILGPU NuGet (~1.5 MB) | NVRTC DLL (CUDA toolkit install) |
| **.NET Native** | Same language as VL.Cuda | C++ string templates in C# |

### ILGPU Integration: PTX Extraction Only

ILGPU is used **only as a PTX compiler** — no accelerator, no CUDA context, no runtime. The workflow:

1. Construct kernel IR via `IRContext.Declare()` + `Method.Builder`
2. Compile via `PTXBackend.Compile()` → `PTXCompiledKernel`
3. Extract PTX string via `.PTXAssembly`
4. Load into ManagedCuda via `cuModuleLoadData()`
5. Add to CUDA Graph via `cuGraphAddKernelNode()`

No CUDA context conflict with ManagedCuda because `PTXBackend` does not create a CUDA context.

### VL Node-Set → ILGPU IR Mapping

```
VL Node               ILGPU IR Builder Call
═══════               ════════════════════════
GPU.Add(a, b)     →   CreateArithmetic(a, b, BinaryArithmeticKind.Add)
GPU.Mul(a, b)     →   CreateArithmetic(a, b, BinaryArithmeticKind.Mul)
GPU.Sin(x)        →   CreateArithmetic(x, UnaryArithmeticKind.Sin)    (via math intrinsic)
GPU.Load(ptr)     →   CreateLoad(ptr)
GPU.Store(ptr, v) →   CreateStore(ptr, v)
GPU.ThreadIdx.X   →   CreateGroupIndexValue(DeviceConstantDimension3D.X)
GPU.BlockIdx.X    →   CreateGridIndexValue(DeviceConstantDimension3D.X)
GPU.SharedMem     →   CreateAlloca(type, MemoryAddressSpace.Shared)
GPU.Barrier       →   CreateBarrier()
GPU.AtomicAdd     →   CreateAtomic(AtomicKind.Add, ...)
GPU.If(cond)      →   CreateIfBranch(cond, trueBlock, falseBlock)
Constant(42)      →   CreatePrimitiveValue(location, 42)
```

65 IR ValueKind types available: arithmetic, memory, control flow, atomics, threading, device constants.

### Escape-Hatch: NVRTC for User CUDA C++

For users who want to load their own CUDA C++ code at runtime (not through the VL visual node-set), NVRTC remains available:

```csharp
// User-written CUDA C++ (not generated from VL nodes)
string userCudaSource = File.ReadAllText("my_kernel.cu");

var module = nvrtcCache.GetOrCompile(userCudaSource, "my_kernel", sm_75);
var kernel = builder.AddKernel(module, "my_kernel");
```

This path uses `NvrtcCache` (CUDA C++ → PTX via `nvrtcCompileProgram()`).

### Compilation Caching

Both ILGPU and NVRTC compilation results are cached:

```csharp
internal class IlgpuCompiler
{
    // Key: hash of IR description (node-set topology + types)
    // Value: compiled PTX string + CUmodule
    private ConcurrentDictionary<string, CachedModule> _cache;

    public CUmodule GetOrCompile(IKernelIRDescription irDesc, int targetSM);
    public void Invalidate(string descriptionHash);
}

internal class NvrtcCache
{
    // Key: hash of CUDA C++ source string
    // Value: compiled CUmodule
    private ConcurrentDictionary<string, CachedModule> _cache;

    public CUmodule GetOrCompile(string cudaSource, string entryPoint, int targetSM);
    public void Invalidate(string sourceHash);
}
```

---

## Source 3: Library Calls (CapturedNode)

Library Calls are invocations of NVIDIA libraries (cuBLAS, cuFFT, cuDNN, cuRAND, NPP) that cannot be expressed as explicit kernel nodes. These libraries use internal, opaque kernels that are invisible to the CUDA Graph API.

**Integration mechanism:** Stream Capture wraps the library call into a ChildGraphNode.

### Static CapturedNode (Single Library Call)

**Flow:**
```
Host-side:   cuStreamBeginCapture(stream, RELAXED)
                 │
                 ▼
Library:     cublasSgemm(handle, ..., stream)   // internal kernels captured
                 │
                 ▼
Host-side:   cuStreamEndCapture(stream) → childGraph
                 │
                 ▼
Graph:       CapturedNode (ChildGraphNode in CUDA Graph)
```

**BlockBuilder usage:**
```csharp
var op = builder.AddCaptured("MatMul", stream =>
{
    cublasSetStream(handle, stream);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}, new CapturedOpDescriptor
{
    DebugName = "cuBLAS_Sgemm",
    Inputs = new[] { ("A", typeof(float)), ("B", typeof(float)) },
    Outputs = new[] { ("C", typeof(float)) },
    Scalars = new[] { ("M", typeof(int)), ("N", typeof(int)), ("K", typeof(int)),
                      ("Alpha", typeof(float)), ("Beta", typeof(float)) }
});
```

### Patchable CapturedNode (Chained Library Calls)

A Patchable CapturedNode composes **multiple library calls into a single block**. The user defines the chain visually in VL — from outside, it's one block/function. Internally, the entire chain is captured via Stream Capture as one ChildGraphNode.

**Flow:**
```
VL Patch:    User composes library call sequence visually
                 │
                 ▼
Host-side:   cuStreamBeginCapture(stream, RELAXED)
                 │
                 ▼
Chain:       cublasSgemm(handle, A, B → temp1, stream)     // Step 1
             cublasSgeam(handle, temp1, C → temp2, stream) // Step 2
             cufftExecC2C(plan, temp2 → result, stream)    // Step 3
                 │
                 ▼
Host-side:   cuStreamEndCapture(stream) → childGraph
                 │
                 ▼
Graph:       CapturedNode (ChildGraphNode) — one node, multiple ops inside
```

**BlockBuilder usage:**
```csharp
var chain = builder.AddCaptured("MatMulFFT", stream =>
{
    // Step 1: Matrix multiply
    cublasSetStream(blasHandle, stream);
    cublasSgemm(blasHandle, ..., A, B, temp1);

    // Step 2: Scale result
    cublasSgeam(blasHandle, ..., temp1, temp2);

    // Step 3: FFT
    cufftSetStream(fftPlan, stream);
    cufftExecC2C(fftPlan, temp2, result, CUFFT_FORWARD);
}, new CapturedOpDescriptor
{
    DebugName = "MatMulFFT_Chain",
    Inputs = new[] { ("A", typeof(float)), ("B", typeof(float)) },
    Outputs = new[] { ("Result", typeof(float)) },
    Scalars = new[] { ("M", typeof(int)), ("N", typeof(int)), ("K", typeof(int)) }
});
```

**From VL's perspective:** One block with Inputs (A, B) and Output (Result). The internal chain is invisible — it's a patchable expression that compiles to a single CapturedNode.

**Update behavior (both static and patchable):**
- Recapture: Any parameter change (scalar or pointer) requires re-capture of the entire chain and `cuGraphExecChildGraphNodeSetParams`. More expensive than Hot/Warm but avoids a full Cold rebuild.
- Cold: Structural changes (chain composition changed) require full rebuild.

### Why Stream Capture is necessary

- cuBLAS, cuFFT, cuDNN are Host-side APIs — they cannot be called from PTX
- Their internal kernels are opaque — grid config, shared memory, and algorithm selection happen inside the library
- Stream Capture is the only CUDA-supported mechanism to include library calls in a CUDA Graph
- NVIDIA provides cuBLASDx/cuFFTDx (Device Extensions) but these require nvcc compilation with proprietary headers that NVRTC cannot access

### Stream Capture Helper

ManagedCuda exposes the stream capture P/Invoke bindings in `DriverAPINativeMethods.Streams` but does not wrap them in the `CudaStream` class. We use a thin internal helper:

```csharp
internal static class StreamCaptureHelper
{
    public static CUgraph CaptureToGraph(CUstream stream, Action<CUstream> work)
    {
        DriverAPINativeMethods.Streams.cuStreamBeginCapture(
            stream, CUstreamCaptureMode.Relaxed);

        work(stream);

        CUgraph graph = default;
        DriverAPINativeMethods.Streams.cuStreamEndCapture(stream, ref graph);
        return graph;
    }
}
```

### Library Handle Caching

Library handles (cublasHandle_t, cufftHandle, etc.) are expensive to create. They are cached per CudaContext:

```csharp
internal class LibraryHandleCache
{
    private ConcurrentDictionary<Type, object> _handles;

    public cublasHandle GetOrCreateBlas(CUstream stream);
    public cufftHandle GetOrCreateFFT(CUstream stream, int size, cufftType type);
}
```

### PointerMode Requirement

For cuBLAS operations in graphs, scalar parameters (alpha, beta) **must** use `CUBLAS_POINTER_MODE_DEVICE`:

```csharp
cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
```

With Host pointer mode, scalar values are baked into the captured graph at capture time — they cannot be updated without re-capture. With Device pointer mode, scalars live in GPU memory and can be updated via buffer writes (Warm update), avoiding re-capture for scalar-only changes.

---

## Comparison

| Aspect | Filesystem PTX | Patchable Kernel (ILGPU) | User CUDA C++ (NVRTC) | Library Call | Patchable Captured |
|--------|---------------|--------------------------|----------------------|-------------|-------------------|
| **Source** | .ptx + .json file | VL node-set → ILGPU IR | User .cu file → NVRTC | Single lib call | Chain of lib calls |
| **Node type** | KernelNode | KernelNode | KernelNode | CapturedNode | CapturedNode |
| **Variant** | Static | Patchable | Escape-hatch | Static | Patchable |
| **Hot Update** | ~0 cost | ~0 cost | ~0 cost | Recapture | Recapture |
| **Warm Update** | cheap | cheap | cheap | Recapture* | Recapture* |
| **Code Rebuild** | Cold (FileWatcher) | ILGPU ~1-10ms | NVRTC ~100ms-2s | N/A | N/A |
| **Compile speed** | N/A (pre-compiled) | Fast (~1-10ms) | Slow (~100ms-2s) | N/A | N/A |
| **Profiling** | Full (PerKernel) | Full (PerKernel) | Full (PerKernel) | PerBlock only | PerBlock only |
| **Grid config** | User-controlled | User-controlled | User-controlled | Library decides | Library decides |
| **Performance** | Depends on author | ~90-95% of library | Depends on author | 100% (vendor) | 100% (vendor) |
| **Complexity** | Low (just load) | Medium (IR builder) | Medium (NVRTC) | Medium (capture) | Higher (chain + capture) |

*With `CUBLAS_POINTER_MODE_DEVICE`, scalar-only changes can avoid Recapture.

---

## When to Use Which

```
Want full control over GPU ops?             → Filesystem PTX (Triton/nvcc, pre-compiled)
Rapid visual prototyping?                   → Patchable Kernel (ILGPU, instant VL iteration)
Need vendor-optimized FFT/GEMM?            → Library Call (cuFFT/cuBLAS, single op)
Complex multi-step library pipeline?        → Patchable Captured (chained cuBLAS/cuFFT)
Loading custom CUDA C++ at runtime?         → User CUDA C++ (NVRTC escape-hatch)
Simple element-wise math in VL?             → Patchable Kernel (trivial IR construction)
Production particle system?                 → Filesystem PTX (Triton, pre-compiled, tuned)
```

---

## Architecture Integration

Both node types (with both variants) live in the same block system and graph compiler:

```
┌─────────────────────────────────────────────────────────────┐
│  BlockBuilder                                                │
│                                                              │
│  .AddKernel(ptxPath, entry)       → KernelNodeDescriptor     │  Static KernelNode
│  .AddKernel(ilgpuModule, entry)   → KernelNodeDescriptor     │  Patchable KernelNode
│  .AddKernel(nvrtcModule, entry)   → KernelNodeDescriptor     │  User CUDA C++ (escape-hatch)
│  .AddCaptured(name, action)       → CapturedNodeDescriptor   │  Static CapturedNode
│  .AddCaptured(name, chainAction)  → CapturedNodeDescriptor   │  Patchable CapturedNode
│                                                              │
│  All produce INodeDescriptor → Graph Compiler handles all    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Graph Compiler                                              │
│                                                              │
│  Phase 1-5: Same for all (validation, alloc, topo, shape,   │
│             liveness)                                        │
│  Phase 5.5: Stream Capture — only for CapturedNodes          │
│  Phase 6:   CUDA Build — KernelNode via cuGraphAddKernelNode │
│                          CapturedNode via AddChildGraphNode  │
│  Phase 7:   Instantiate — same for all                       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Dirty Tracking                                              │
│                                                              │
│  KernelNode:    Hot → Warm → Code → Cold                     │
│  CapturedNode:  Recapture → Cold                             │
│                                                              │
│  Both managed by the same DirtyTracker, dispatched by        │
│  CudaEngine based on node type.                              │
└─────────────────────────────────────────────────────────────┘
```
