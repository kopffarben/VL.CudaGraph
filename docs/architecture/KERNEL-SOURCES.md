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
// Patchable kernel via ILGPU: pass the C# method + user-facing descriptor.
// IlgpuCompiler auto-expands the descriptor for ILGPU's PTX layout.
// KernelHandle.In()/Out() indices are auto-remapped (+1 for _kernel_length).
var descriptor = new KernelDescriptor { EntryPoint = "user_kernel", BlockSize = 256 };
descriptor.Parameters.Add(new KernelParamDescriptor { Name = "A", IsPointer = true, Direction = ParamDirection.In });
descriptor.Parameters.Add(new KernelParamDescriptor { Name = "B", IsPointer = true, Direction = ParamDirection.In });
descriptor.Parameters.Add(new KernelParamDescriptor { Name = "C", IsPointer = true, Direction = ParamDirection.Out });

var kernel = builder.AddKernel(myKernelMethod, descriptor);
kernel.GridDimX = 1024;
builder.Input<float>("A", kernel.In(0));   // remapped to PTX param 1 (ArrayView struct)
builder.Input<float>("B", kernel.In(1));   // remapped to PTX param 2
builder.Output<float>("C", kernel.Out(2)); // remapped to PTX param 3
```

**Update behavior:**
- Hot/Warm: Same as Filesystem PTX -- full parameter patchability
- Code: User edits the C# method or node-set -> `CudaContext.OnCodeChanged()` -> Code dirty -> IlgpuCompiler cache invalidated -> Cold Rebuild (full graph rebuild in current implementation)

**Key constraint:** `cuGraphExecKernelNodeSetParams` cannot swap the `CUfunction` itself. A code change always requires a Cold rebuild. However, all parameter updates remain Hot/Warm -- only the kernel logic change triggers a rebuild. Partial rebuild (recompiling only the affected block) is a future optimization.

### Abstraction Level: Element-Wise (ShaderFX Model)

Patchable kernels use an **element-wise** abstraction — like ShaderFX for compute.
The user describes what happens to a single element. Grid configuration, thread
management, and parallelization are invisible and automatic.

```
VL Patch (user perspective):          CUDA (what actually runs):

  GPU.Mul(A, B) → C                    for each thread i in parallel:
                                            C[i] = A[i] * B[i]
```

The user **does not** think about:
- Thread blocks, grid dimensions, warps
- Memory coalescing, shared memory, barriers
- Thread indices, block indices

Grid size is derived automatically from input buffer dimensions. Block size
defaults to 256 (tunable as an advanced option on the block).

This matches the ShaderFX precedent where VL users compose pixel/vertex operations
without thinking about GPU execution details. See `VL-INTEGRATION.md` for the
full patching UX design.

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

For users who want to load their own CUDA C++ code at runtime (not through the VL visual node-set), NVRTC remains available via `BlockBuilder.AddKernelFromCuda()`:

```csharp
// User-written CUDA C++ (not generated from VL nodes)
string userCudaSource = File.ReadAllText("my_kernel.cu");
var descriptor = new KernelDescriptor { EntryPoint = "my_kernel", BlockSize = 256, ... };

var kernel = builder.AddKernelFromCuda(userCudaSource, "my_kernel", descriptor);
builder.Input<float>("A", kernel.In(0));
builder.Output<float>("B", kernel.Out(1));
```

This path uses `NvrtcCache` (CUDA C++ -> PTX via `CudaRuntimeCompiler` -> `PtxLoader.LoadFromBytes()`). The compute architecture (`compute_XY`) is auto-derived from `DeviceContext`. Compilation errors include the NVRTC log for diagnostics.

### Compilation Caching

Both ILGPU and NVRTC compilation results are cached:

```csharp
internal sealed class IlgpuCompiler : IDisposable
{
    // ILGPU Context + PTXBackend (no CUDA accelerator, pure compiler mode)
    private readonly IlgpuContext _ilgpuContext;
    private readonly PTXBackend _backend;
    private readonly DeviceContext _device;

    // Key: SHA256 hash of method (AssemblyQualifiedName.Name.MetadataToken)
    // Value: LoadedKernel (CUmodule + expanded KernelDescriptor)
    private ConcurrentDictionary<string, LoadedKernel> _cache;

    public LoadedKernel GetOrCompile(MethodInfo kernelMethod, KernelDescriptor descriptor);
    public bool Invalidate(string methodHash);
    public int CacheCount { get; }

    // Internal: compile to PTX string + entry point name
    internal (string PtxString, string EntryPointName) CompileToStringWithName(MethodInfo method);

    // Internal: expand descriptor for ILGPU PTX layout
    // Param 0: implicit _kernel_length (int32, 4 bytes)
    // Params 1..N: user params, ArrayView<T> → 16-byte struct {ptr, length}
    internal static KernelDescriptor ExpandDescriptorForIlgpu(KernelDescriptor original, string entryPointName);

    // Internal: compute index remap (user indices shift by 1 for _kernel_length)
    internal static int[] ComputeIndexRemap(KernelDescriptor original);

    // Internal: SHA256 hash from MethodInfo
    internal static string ComputeMethodHash(MethodInfo method);
}

internal sealed class NvrtcCache : IDisposable
{
    // Key: SHA256 hash of CUDA C++ source string
    // Value: LoadedKernel (CUmodule + KernelDescriptor)
    private ConcurrentDictionary<string, LoadedKernel> _cache;

    public LoadedKernel GetOrCompile(string cudaSource, string entryPoint, KernelDescriptor descriptor);
    public bool Invalidate(string sourceHash);
    public int CacheCount { get; }

    // Internal: compile to PTX bytes via CudaRuntimeCompiler
    internal byte[] CompileToBytes(string cudaSource, string entryPoint);
    internal static string ComputeSourceKey(string source);
}
```

Both compilers produce a `LoadedKernel` via `PtxLoader.LoadFromBytes()` — the same path as filesystem PTX, just with bytes instead of a file path. Both are owned by `CudaContext` and share the pipeline lifetime.

### KernelSource Discriminated Union

All three kernel sources are represented by a discriminated union (`KernelSource`), stored in `KernelHandle.Source` and `KernelEntry.Source`:

```csharp
public abstract class KernelSource
{
    public abstract string GetCacheKey();    // stable key for deduplication
    public abstract string GetDebugName();   // human-readable name for diagnostics

    public sealed class FilesystemPtx : KernelSource
    {
        public string PtxPath { get; }
        // GetCacheKey() → "file:{fullPath}"
    }

    public sealed class IlgpuMethod : KernelSource
    {
        public string MethodHash { get; }
        public MethodInfo KernelMethod { get; }
        // GetCacheKey() → "ilgpu:{methodHash}"
    }

    public sealed class NvrtcSource : KernelSource
    {
        public string SourceHash { get; }
        public string CudaSource { get; }
        public string EntryPoint { get; }
        // GetCacheKey() → "nvrtc:{sourceHash}"
    }
}
```

`CudaEngine.LoadKernelFromSource()` dispatches on the variant to route to the correct loader:
```csharp
return entry.Source switch
{
    KernelSource.FilesystemPtx fs    => Context.ModuleCache.GetOrLoad(fs.PtxPath),
    KernelSource.IlgpuMethod ilgpu   => Context.IlgpuCompiler.GetOrCompile(ilgpu.KernelMethod, entry.Descriptor!),
    KernelSource.NvrtcSource nvrtc   => Context.NvrtcCache.GetOrCompile(nvrtc.CudaSource, nvrtc.EntryPoint, entry.Descriptor!),
};
```

### ILGPU Parameter Layout

ILGPU compiles implicitly-grouped kernels with a different parameter layout than filesystem PTX:

```
Param 0:  _kernel_length  (.b32, 4 bytes) — implicit total element count (Index1D upper bound)
Param 1+: User parameters where each ArrayView<T> → 16-byte struct {void* ptr, long length}
          Scalars remain their natural size (4/8 bytes)
```

`IlgpuCompiler.ExpandDescriptorForIlgpu()` transforms the user-provided descriptor to match this layout. `IlgpuCompiler.ComputeIndexRemap()` produces an index translation table so `KernelHandle.In()/Out()` transparently remaps user-facing indices to expanded PTX indices (all shift by +1 for the implicit `_kernel_length`).

`CudaEngine.InitializeIlgpuParams()` sets the implicit parameters during ColdRebuild:
- `_kernel_length` is set to `GridDimX * BlockDimX` (total thread count)
- ArrayView structs get their length field set to `long.MaxValue` (pointers are overwritten by external buffer binding)

### Code Dirty Level

When a kernel's source code changes (e.g., user edits the C# method for an ILGPU kernel, or modifies a CUDA C++ string for NVRTC), the block calls `CudaContext.OnCodeChanged()`:

```csharp
// In CudaContext:
public void OnCodeChanged(Guid blockId, Guid handleId, KernelSource newSource)
{
    Dirty.MarkCodeDirty(new DirtyCodeEntry(blockId, handleId, newSource));
}

// DirtyCodeEntry identifies which kernel changed and what the new source is:
public readonly record struct DirtyCodeEntry(Guid BlockId, Guid KernelHandleId, KernelSource NewSource);
```

`CudaEngine.Update()` checks Code dirty after Structure but before CapturedNodes/Parameters:

```
Priority: Structure > Code > CapturedNodes > Parameters
```

`CudaEngine.CodeRebuild()` invalidates the affected compiler caches, then triggers a full ColdRebuild:

```csharp
private void CodeRebuild()
{
    foreach (var entry in Context.Dirty.GetDirtyCodeEntries())
    {
        switch (entry.NewSource)
        {
            case KernelSource.IlgpuMethod ilgpu:  Context.IlgpuCompiler.Invalidate(ilgpu.MethodHash); break;
            case KernelSource.NvrtcSource nvrtc:   Context.NvrtcCache.Invalidate(nvrtc.SourceHash); break;
            case KernelSource.FilesystemPtx fs:    Context.ModuleCache.Evict(fs.PtxPath); break;
        }
    }
    Context.Dirty.ClearCodeDirty();
    // Structure is now dirty → ColdRebuild will follow
}
```

**Current behavior:** Code dirty always triggers a full Cold Rebuild. Partial rebuild (recompiling only the affected block while keeping the rest of the graph) is a future optimization.

---

## Source 3: Library Calls (CapturedNode)

Library Calls are invocations of NVIDIA libraries (cuBLAS, cuFFT, cuSPARSE, cuRAND, cuSOLVER) that cannot be expressed as explicit kernel nodes. These libraries use internal, opaque kernels that are invisible to the CUDA Graph API. The current implementation provides wrappers for all five libraries with `CapturedNode` serving as the graph node type and `CapturedNodeDescriptor` describing the parameter layout.

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
var descriptor = new CapturedNodeDescriptor("cuBLAS.Sgemm",
    inputs: new[] { CapturedParam.Pointer("A", "float*"), CapturedParam.Pointer("B", "float*") },
    outputs: new[] { CapturedParam.Pointer("C", "float*") });

var op = builder.AddCaptured((stream, buffers) =>
{
    // buffers layout: [A, B, C] (inputs first, then outputs, then scalars)
    var blas = libs.GetOrCreateBlas();
    blas.Stream = stream;
    CudaBlasNativeMethods.cublasSgemm_v2(blas.CublasHandle, transA, transB,
        m, n, k, ref alpha, buffers[0], lda, buffers[1], ldb, ref beta, buffers[2], ldc);
}, descriptor);

// Bind block ports to captured operation parameters
builder.Input<float>("A", op.In(0));    // CapturedPin → flat index 0
builder.Input<float>("B", op.In(1));    // CapturedPin → flat index 1
builder.Output<float>("C", op.Out(0));  // CapturedPin → flat index 2
```

**Or use the high-level BlasOperations wrapper:**
```csharp
var sgemm = BlasOperations.Sgemm(builder, libs, m: 128, n: 128, k: 128);
builder.Input<float>("A", sgemm.In(0));
builder.Input<float>("B", sgemm.In(1));
builder.Output<float>("C", sgemm.Out(0));
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
var descriptor = new CapturedNodeDescriptor("MatMulFFT_Chain",
    inputs: new[] { CapturedParam.Pointer("A", "float*"), CapturedParam.Pointer("B", "float*") },
    outputs: new[] { CapturedParam.Pointer("Result", "float2*") });

var chain = builder.AddCaptured((stream, buffers) =>
{
    // Step 1: Matrix multiply
    var blas = libs.GetOrCreateBlas();
    blas.Stream = stream;
    CudaBlasNativeMethods.cublasSgemm_v2(blas.CublasHandle, ...,
        buffers[0], ..., buffers[1], ..., temp1, ...);

    // Step 2: FFT
    var plan = libs.GetOrCreateFFT1D(nx, cufftType.C2C);
    CudaFFTNativeMethods.cufftSetStream(plan.Handle, stream);
    plan.Exec(temp1, buffers[2], TransformDirection.Forward);
}, descriptor);

builder.Input<float>("A", chain.In(0));
builder.Input<float>("B", chain.In(1));
builder.Output<float>("Result", chain.Out(0));
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

ManagedCuda exposes the stream capture P/Invoke bindings in `DriverAPINativeMethods.Streams` but does not wrap them in the `CudaStream` class. `StreamCaptureHelper` is a thin internal static class that wraps stream capture, child graph node insertion, and recapture update:

```csharp
internal static class StreamCaptureHelper
{
    // Capture a work action between Begin/EndCapture. Caller owns the returned CUgraph.
    public static CUgraph CaptureToGraph(CUstream stream, Action<CUstream> work);

    // Destroy a CUgraph returned from CaptureToGraph.
    public static void DestroyGraph(CUgraph graph);

    // Add a child graph node to a parent graph (used by GraphCompiler).
    public static CUgraphNode AddChildGraphNode(
        CUgraph parentGraph, CUgraphNode[]? dependencies, CUgraph childGraph);

    // Update a child graph node in an executable graph (Recapture update).
    public static void UpdateChildGraphNode(
        CUgraphExec exec, CUgraphNode node, CUgraph newChildGraph);
}
```

Error handling: if the work action throws during capture, `StreamCaptureHelper` still calls `cuStreamEndCapture` and destroys the discarded graph to leave the stream in a valid state before re-throwing.

### Library Handle Caching

Library handles (cublasHandle_t, cufftHandle, etc.) are expensive to create. `LibraryHandleCache` provides lazy-initialized per-CudaContext caching for five CUDA library handle types:

```csharp
public sealed class LibraryHandleCache : IDisposable
{
    // cuBLAS — single handle, reused across all BLAS operations
    public CudaBlas GetOrCreateBlas();

    // cuFFT — plans cached by (nx, type, batch) since they are configuration-dependent
    public CudaFFTPlan1D GetOrCreateFFT1D(int nx, cufftType type, int batch = 1);

    // cuSPARSE — single context handle
    public CudaSparseContext GetOrCreateSparse();

    // cuRAND — device generator handle
    public CudaRandDevice GetOrCreateRand(GeneratorType type = GeneratorType.PseudoDefault);

    // cuSOLVER — dense solver handle
    public CudaSolveDense GetOrCreateSolveDense();
}
```

Each handle is created on first access and disposed when the cache is disposed. The cache is owned by `CudaContext` and shares the lifetime of the CUDA pipeline.

### Library Operation Wrappers

High-level static wrapper classes produce `CapturedHandle` entries that can be used directly with `BlockBuilder`:

| Class | Operations | Library |
|-------|-----------|---------|
| `BlasOperations` | Sgemm, Dgemm, Sgemv, Sscal | cuBLAS |
| `FftOperations` | Forward1D, Inverse1D, R2C1D, C2R1D | cuFFT |
| `SparseOperations` | SpMV | cuSPARSE |
| `RandOperations` | GenerateUniform, GenerateNormal | cuRAND |
| `SolveOperations` | Sgetrf, Sgetrs | cuSOLVER |

Each wrapper method takes a `BlockBuilder` and `LibraryHandleCache`, constructs the `CapturedNodeDescriptor`, and returns a `CapturedHandle` ready for port binding. Example:

```csharp
var sgemm = BlasOperations.Sgemm(builder, libs, m: 128, n: 128, k: 128);
builder.Input<float>("A", sgemm.In(0));
builder.Input<float>("B", sgemm.In(1));
builder.Output<float>("C", sgemm.Out(0));
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

## Device Graph Launch & Dispatch Indirect *(Phase 6 — planned)*

> **Prerequisite:** Conditional Nodes (Phase 3.2). See `GRAPH-COMPILER.md` for the full three-level design.

Device Graph Launch enables GPU kernels to control graph execution — including dynamically setting grid dimensions of other kernel nodes. This is the CUDA equivalent of DX11/DX12 **DispatchIndirect**.

### ManagedCuda API Coverage (Verified)

All required host-side APIs are present in ManagedCuda 13.0:

| API | Purpose |
|-----|---------|
| `CUgraphInstantiate_flags.DeviceLaunch` | Enable device-side graph launch |
| `cuGraphInstantiateWithParams()` | Instantiate with DeviceLaunch flag |
| `cuGraphUpload()` | Upload graph to device after device-side modifications |
| `CUlaunchAttributeID.DeviceUpdatableKernelNode` | Mark kernel as updateable from device |
| `CUgraphDeviceNode` | Opaque handle passed to device kernels |
| `cuGraphConditionalHandleCreate()` | Conditional control from device |
| `CUgraphConditionalNodeType.If/While/Switch` | All conditional types |

Device-side APIs (`cudaGraphLaunch`, `cudaGraphKernelNodeSetParam`, `cudaGraphSetConditional`) are **PTX intrinsics** called from within kernel code — they do not need .NET bindings.

### Impact on Kernel Sources

Device-updatable nodes and conditional handles affect all three kernel sources:

| Source | Device Graph Impact |
|--------|-------------------|
| **Filesystem PTX** | PTX author includes `cudaGraphKernelNodeSetParam` / `cudaGraphSetConditional` intrinsics directly |
| **Patchable Kernel (ILGPU)** | Dispatcher kernel is a special built-in PTX (not user-patchable). Target kernels are marked device-updatable via launch attributes |
| **Library Call (CapturedNode)** | CapturedNodes inside conditional bodies — entire body is skipped/executed based on device-side condition |

### New Node Type: DispatcherNode *(Phase 6)*

A DispatcherNode is a specialized KernelNode that reads a GPU counter and sets the grid dimensions of a target kernel node. It bridges AppendBuffer output (variable count) with the next processing stage (exact thread count).

```
┌─────────────────────────────────────────────────────────┐
│  DispatcherNode (built-in PTX, device-updatable)         │
│                                                          │
│  Input:  uint* counter   (GPU pointer from AppendBuffer) │
│  Input:  CUgraphDeviceNode targetNode                    │
│  Param:  uint blockSize  (256 default)                   │
│                                                          │
│  PTX body:                                               │
│    gridDim = ceil(*counter / blockSize)                   │
│    cudaGraphKernelNodeSetParam(targetNode, gridDim, ...) │
└─────────────────────────────────────────────────────────┘
```

This is a **system-provided kernel** — users don't write it. It's part of the VL.Cuda runtime.

---

## Architecture Integration

Both node types (with both variants) live in the same block system and graph compiler:

```
┌─────────────────────────────────────────────────────────────────┐
│  BlockBuilder                                                     │
│                                                                   │
│  .AddKernel(ptxPath)                       → KernelHandle         │  Static KernelNode
│  .AddKernel(ilgpuModule, entry)            → KernelHandle         │  Patchable KernelNode *(4b)*
│  .AddKernel(nvrtcModule, entry)            → KernelHandle         │  User CUDA C++ *(4b)*
│  .AddCaptured(action, descriptor)          → CapturedHandle       │  Static CapturedNode
│  .AddCaptured(chainAction, descriptor)     → CapturedHandle       │  Patchable CapturedNode
│                                                                   │
│  KernelHandle.In()/Out()       → KernelPin    (kernel param ref)  │
│  CapturedHandle.In()/Out()/Scalar() → CapturedPin (flat index)    │
│                                                                   │
│  Both handle types → IGraphNode → Graph Compiler handles all      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Graph Compiler                                                   │
│                                                                   │
│  Phase 1-4: Same for all (validation, alloc, topo, memset)        │
│  Phase 5b:  Stream Capture — CapturedNodes on dedicated stream    │
│             CapturedNode.Capture(stream) → AddChildGraphNode      │
│  Phase 5c:  Kernel node deps can reference captured node handles  │
│  Phase 6:   CUDA Build — KernelNode via cuGraphAddKernelNode      │
│  Phase 7:   Instantiate — same for all                            │
│                                                                   │
│  CompiledGraph stores capturedNodeHandles: Dict<Guid, CUgraphNode>│
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Dirty Tracking (DirtyTracker — four levels)                      │
│                                                                   │
│  Priority:     Structure > Code > CapturedNodes > Parameters      │
│                                                                   │
│  KernelNode:    Hot → Warm → Code → Cold                          │
│  CapturedNode:  Recapture → Cold                                  │
│                                                                   │
│  DirtyTracker.IsCodeDirty → CudaEngine.CodeRebuild()              │
│    Invalidates compiler cache → marks structure dirty → ColdRebuild│
│  DirtyTracker.AreCapturedNodesDirty → CudaEngine.RecaptureNodes() │
│  All managed by the same DirtyTracker, dispatched by              │
│  CudaEngine based on node type.                                   │
└─────────────────────────────────────────────────────────────────┘
```
