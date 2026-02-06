# Kernel Sources & Node Types

## Overview

The system supports **three kernel sources** that map to **two CUDA Graph node types**. This design keeps the block system uniform while enabling both custom kernels and NVIDIA library operations.

```
KERNEL SOURCES                              NODE TYPE IN GRAPH
══════════════                              ══════════════════

1. Filesystem PTX ──────────────┐
   (Triton, nvcc, hand-written) │
   .ptx + .json files           │
                                ├───→  KernelNode
2. Patchable Kernel ────────────┘      ├─ Hot:       Scalar changed         ~0 cost
   (VL Node-Set → NVRTC Codegen)      ├─ Warm:      Pointer changed        cheap
   Live or OnSave compilation          ├─ Code:      NVRTC recompile        medium
                                       └─ Cold:      Structure rebuild      expensive

3. Library Call ──────────────────→  CapturedNode (ChildGraphNode)
   (cuBLAS, cuFFT, cuDNN, cuRAND)     ├─ Recapture: Param changed          medium
   Stream Capture                      └─ Cold:      Structure rebuild      expensive
```

All three sources are **fully integrated** into the block system. Blocks use `BlockBuilder` to declare their content — the Graph Compiler handles the rest.

---

## Source 1: Filesystem PTX

Pre-compiled PTX files loaded from disk. This is the primary path for custom kernels.

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

## Source 2: Patchable Kernels

Kernels generated at runtime from a visual node-set in VL. The user patches GPU operations visually, and the system generates CUDA C++ source code, compiles it via NVRTC, and loads the result as a normal KernelNode.

**Flow:**
```
VL Patch:    User edits GPU node-set (GPU.Add, GPU.Mul, GPU.Reduce, ...)
                 │
                 ▼ Live / OnSave
Codegen:     Node-set → CUDA C++ source string
                 │
                 ▼
NVRTC:       nvrtcCompileProgram() → PTX bytes (or cubin)
                 │
                 ▼
Loader:      cuModuleLoadData(ptxBytes) → CUmodule → CUfunction
                 │
                 ▼
Graph:       Normal KernelNode — same as Filesystem PTX from here on
```

**BlockBuilder usage:**
```csharp
// Patchable kernels use the same AddKernel path,
// but the PTX comes from NVRTC instead of a file.
var kernel = builder.AddKernel(nvrtcModule, "user_kernel", debugName: "MatMulPatch");
builder.Input<float>("A", kernel.In(0));
builder.Input<float>("B", kernel.In(1));
builder.Output<float>("C", kernel.Out(2));
```

**Update behavior:**
- Hot/Warm: Same as Filesystem PTX — full parameter patchability
- Code: User edits the node-set → NVRTC recompile → new CUmodule → Cold rebuild of affected block only (not the entire graph)

**Key constraint:** `cuGraphExecKernelNodeSetParams` cannot swap the `CUfunction` itself. A code change always requires a Cold rebuild of the block. However, all parameter updates remain Hot/Warm — only the kernel logic change is expensive.

### NVRTC Integration

NVRTC (NVIDIA Runtime Compilation) is available in ManagedCuda as a separate module. It compiles CUDA C++ strings to PTX or cubin at runtime.

```csharp
// Simplified flow
string cudaSource = PatchableCodegen.Generate(nodeSet);

nvrtcCreateProgram(ref prog, cudaSource, "user_kernel", 0, null, null);
nvrtcCompileProgram(prog, numOptions, options);  // options: --gpu-architecture=sm_75
nvrtcGetPTX(prog, ptxBytes);

// From here: standard module loading
cuModuleLoadData(ref module, ptxBytes);
cuModuleGetFunction(ref func, module, "user_kernel");
```

### Codegen Node-Set

The patchable kernel system provides a **closed, finite set** of GPU primitives as VL nodes:

```
Element-wise:   GPU.Add, GPU.Mul, GPU.Div, GPU.Abs, GPU.Clamp, GPU.Lerp, ...
Reduction:      GPU.ReduceSum, GPU.ReduceMin, GPU.ReduceMax, ...
Scan:           GPU.PrefixSum, GPU.ExclusiveScan, ...
Math:           GPU.Sin, GPU.Cos, GPU.Exp, GPU.Sqrt, GPU.Pow, ...
Comparison:     GPU.Greater, GPU.Less, GPU.Equal, ...
Selection:      GPU.Where, GPU.Select, ...
Memory:         GPU.Gather, GPU.Scatter, ...
```

These are **not** open VL delegates (VL delegates are always open — they receive context from the VL tracer). Instead, these are a defined set of operations that the codegen knows how to translate to CUDA C++.

### Compilation Caching

NVRTC compilation results are cached alongside the ModuleCache:

```csharp
internal class NvrtcCache
{
    // Key: hash of generated CUDA source
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

**Update behavior:**
- Recapture: Any parameter change (scalar or pointer) requires re-capture of the library call and `cuGraphExecChildGraphNodeSetParams`. This is more expensive than Hot/Warm but avoids a full Cold rebuild.
- Cold: Structural changes require full rebuild.

**Why Stream Capture is necessary:**
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

| Aspect | Filesystem PTX | Patchable Kernel | Library Call |
|--------|---------------|-----------------|-------------------|
| **Source** | .ptx + .json file | VL node-set → NVRTC | cuBLAS/cuFFT/cuDNN |
| **Node type** | KernelNode | KernelNode | CapturedNode |
| **Hot Update** | ✅ ~0 cost | ✅ ~0 cost | ❌ Recapture needed |
| **Warm Update** | ✅ cheap | ✅ cheap | ❌ Recapture needed* |
| **Code Rebuild** | Cold Rebuild (FileWatcher) | Cold Rebuild (NVRTC recompile) | Cold Rebuild |
| **Profiling** | ✅ Full (PerKernel) | ✅ Full (PerKernel) | ⚠️ PerBlock only (opaque) |
| **Grid config** | ✅ User-controlled | ✅ User-controlled | ❌ Library decides |
| **Shape rules** | ✅ Via JSON | ✅ Via codegen | ❌ Hardcoded per op |
| **Performance** | Depends on author | ~90-95% of library | 100% (vendor-optimized) |
| **Debuggability** | ✅ Single kernel | ✅ Single kernel | ⚠️ Opaque child graph |
| **Complexity** | Low (just load) | Medium (codegen) | Medium (stream capture) |

*With `CUBLAS_POINTER_MODE_DEVICE`, scalar-only changes can avoid Recapture.

---

## When to Use Which

```
Want full control and patchability?          → Filesystem PTX or Patchable Kernel
Need vendor-optimized FFT/GEMM?             → Library Call (cuFFT/cuBLAS)
Prototyping a new algorithm?                 → Patchable Kernel (fastest iteration)
Production particle system?                  → Filesystem PTX (Triton, pre-compiled)
Complex deep learning inference?             → Library Call (cuDNN)
Simple element-wise math?                    → Patchable Kernel (trivial codegen)
```

---

## Architecture Integration

Both node types live in the same block system and graph compiler:

```
┌─────────────────────────────────────────────────────────────┐
│  BlockBuilder                                                │
│                                                              │
│  .AddKernel(ptxPath, entry)     → KernelNodeDescriptor       │
│  .AddKernel(nvrtcModule, entry) → KernelNodeDescriptor       │
│  .AddCaptured(name, action)     → CapturedNodeDescriptor     │
│                                                              │
│  Both produce INodeDescriptor → Graph Compiler handles both  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Graph Compiler                                              │
│                                                              │
│  Phase 1-5: Same for both (validation, alloc, topo, shape,  │
│             liveness)                                        │
│  Phase 5.5: Stream Capture — only for CapturedNodes          │
│  Phase 6:   CUDA Build — KernelNode via cuGraphAddKernelNode │
│                          CapturedNode via AddChildGraphNode  │
│  Phase 7:   Instantiate — same for both                      │
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
