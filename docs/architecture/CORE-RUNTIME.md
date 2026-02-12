# Core Runtime

## CudaContext

The `CudaContext` is the central domain container for all GPU operations within a single CUDA device. It is created by `CudaEngine` and flows to blocks via VL pin connections.

> **Note:** CudaContext is created internally by CudaEngine. Users don't create it directly. See `EXECUTION-MODEL.md` for the full lifecycle.

### Internal Architecture

CudaContext is a facade that composes several internal services. These services communicate via events — no service holds a direct reference to another.

```
┌──────────────────────────────────────────────────────────────┐
│ CudaContext (Facade)                                          │
│                                                               │
│  ┌──────────────┐   StructureChanged    ┌──────────────────┐ │
│  │ BlockRegistry │──── event ──────────▶│  DirtyTracker    │ │
│  └──────────────┘                       │                  │ │
│  ┌──────────────┐   StructureChanged    │  structureDirty  │ │
│  │ConnectionGraph│──── event ──────────▶│  parameterDirty  │ │
│  └──────────────┘                       └──────────────────┘ │
│  ┌──────────────┐                                            │
│  │ BufferPool    │  (CUDA-specific pooling, own impl)        │
│  └──────────────┘                                            │
│  ┌──────────────┐                                            │
│  │ ModuleCache   │  (PTX compilation cache)                  │
│  └──────────────┘                                            │
│  ┌──────────────┐                                            │
│  │IlgpuCompiler │  (ILGPU IR → PTX for patchable kernels)    │
│  └──────────────┘                                            │
│  ┌──────────────┐                                            │
│  │ NvrtcCache    │  (NVRTC escape-hatch for user CUDA C++)    │
│  └──────────────┘                                            │
│  ┌──────────────────┐                                         │
│  │LibraryHandleCache│  (cuBLAS/cuFFT handle cache)            │
│  └──────────────────┘                                         │
│  ┌──────────────┐                                            │
│  │ DeviceContext │  (CUcontext, streams, device handles)     │
│  └──────────────┘                                            │
│                                                               │
│  Public API:  DeviceId, Pool, ProfilingLevel                  │
│  Internal:    Registry, Topology, Dirty, Modules, Device      │
└──────────────────────────────────────────────────────────────┘
```

### Event-Based Coupling

Internal services communicate through events, wired up in the CudaContext constructor:

```csharp
public class CudaContext : IDisposable
{
    // Public API (visible to VL users)
    public int DeviceId { get; }
    public BufferPool Pool { get; }
    public ProfilingLevel ProfilingLevel { get; set; }

    // Internal services (visible to CudaEngine, GraphCompiler)
    internal BlockRegistry Registry { get; }
    internal ConnectionGraph Topology { get; }
    internal DirtyTracker Dirty { get; }
    internal ModuleCache Modules { get; }
    internal DeviceContext Device { get; }

    internal CudaContext(CudaEngineOptions options)
    {
        Device = new DeviceContext(options.DeviceId);
        Pool = new BufferPool(Device);
        Modules = new ModuleCache(Device);
        Ilgpu = new IlgpuCompiler(Device);
        Nvrtc = new NvrtcCache(Device);
        LibHandles = new LibraryHandleCache(Device);
        Registry = new BlockRegistry();
        Topology = new ConnectionGraph();
        Dirty = new DirtyTracker();

        // Event wiring — the ONLY place services are connected
        Registry.StructureChanged += () => Dirty.MarkStructureDirty();
        Topology.StructureChanged += () => Dirty.MarkStructureDirty();
    }

    public void Dispose()
    {
        Pool.Dispose();
        Modules.Dispose();
        Ilgpu.Dispose();
        Nvrtc.Dispose();
        LibHandles.Dispose();
        Device.Dispose();
    }
}
```

### Why Events, Not Direct References

BlockRegistry and ConnectionGraph both trigger structure-dirty independently. With events:
- Neither service knows DirtyTracker exists
- New consumers (e.g. future validation cache) subscribe without changing existing code
- The wiring is explicit and centralized in one constructor
- Testing is simpler — services can be tested in isolation

### Service Responsibilities

| Service | Responsibility |
|---------|---------------|
| **BlockRegistry** | Block registration/unregistration, NodeContext storage for error routing |
| **ConnectionGraph** | Topology tracking (edges between blocks), connection validation |
| **DirtyTracker** | Dirty flags per node type (KernelNode: Hot Update/Warm Update/Code Rebuild/Cold Rebuild; CapturedNode: Recapture/Cold Rebuild) |
| **BufferPool** | GPU memory allocation with power-of-2 bucketing |
| **ModuleCache** | PTX loading, CUmodule caching, kernel descriptor storage |
| **IlgpuCompiler** | ILGPU IR → PTX compiler for patchable kernels (primary path) |
| **NvrtcCache** | NVRTC compilation cache for user CUDA C++ (escape-hatch) |
| **LibraryHandleCache** | Cached library handles (cublasHandle_t, cufftHandle, etc.) |
| **DeviceContext** | CUcontext, default stream, capture stream, device properties |

### CUDA Context Constraint

A CUDA Graph's nodes must all reside on a single device. Therefore one CudaContext always maps to exactly one CUDA device context (`CUcontext`). Multiple CudaEngines in the same patch are separate worlds — cross-engine buffer access is not possible and is validated during graph compilation.

### Block Registration

Blocks use the public facade methods on CudaContext. These delegate to internal services and fire the appropriate events:

```csharp
// Block constructor registers (internally: Registry fires StructureChanged → DirtyTracker):
ctx.RegisterBlock(this);  // reads this.NodeContext for identity/diagnostics

// Block Dispose unregisters (internally: Registry fires StructureChanged → DirtyTracker):
ctx.UnregisterBlock(this.Id);

// VL link drawing (internally: Topology fires StructureChanged → DirtyTracker):
ctx.Connect(srcId, "Out", tgtId, "In");
ctx.Disconnect(srcId, "Out", tgtId, "In");
```

### Dirty Tracking

```csharp
internal class DirtyTracker
{
    public bool IsStructureDirty { get; private set; } = true;
    public bool AreParametersDirty { get; private set; }

    // Called via events from BlockRegistry / ConnectionGraph
    public void MarkStructureDirty() => IsStructureDirty = true;

    // Called when scalar/pointer values change
    public void MarkParametersDirty() => AreParametersDirty = true;

    public void ClearAll()
    {
        IsStructureDirty = false;
        AreParametersDirty = false;
    }

    public void ClearParameters() => AreParametersDirty = false;
}
```

### Global Services via AppHost.ServiceRegistry

Some information is truly app-global and belongs in VL's ServiceRegistry rather than per-CudaContext:

```csharp
// Registered once at startup (e.g. in a VL static initializer)
appHost.Services.RegisterService<CudaDeviceInfo>(new CudaDeviceInfo());
appHost.Services.RegisterService<CudaDriverVersion>(new CudaDriverVersion());

// Accessible anywhere via AppHost
var deviceInfo = AppHost.Current.Services.GetService<CudaDeviceInfo>();
```

Per-instance state (pool, registry, dirty, device) stays in CudaContext — because multiple CudaContexts per app must be possible (multi-GPU).

---

## CudaStream

Streams enable asynchronous operations and can be used for:
- Overlapping compute with data transfer
- Multiple independent execution paths

```csharp
// Create stream
using var stream = CudaStream.Create(ctx);

// Execute on stream
engine.Update(stream);

// Sync (blocking)
stream.Synchronize();

// Sync (async)
await stream.SynchronizeAsync();
```

---

## GpuBuffer\<T\>

Type-safe wrapper for GPU memory. The generic parameter ensures type safety at compile time.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Id` | `Guid` | Unique identifier |
| `Pointer` | `CUdeviceptr` | Raw GPU pointer |
| `SizeInBytes` | `long` | Total size in bytes |
| `ElementCount` | `int` | Number of elements |
| `ElementType` | `DType` | Data type enum |
| `Shape` | `BufferShape` | Dimensional information |
| `Lifetime` | `BufferLifetime` | Management mode |
| `State` | `BufferState` | Current state |

### Data Types (DType)

```csharp
public enum DType
{
    U8, U16, U32, U64,
    S8, S16, S32, S64,
    F16, F32, F64,
    BF16,
    Complex64, Complex128
}
```

### Lifetime Management

```csharp
public enum BufferLifetime
{
    External,   // User-managed: user allocates and frees
    Graph,      // Graph-managed: freed when graph disposed
    Region      // Region-managed: freed after region completes
}
```

### Upload/Download

```csharp
// Upload from CPU
float[] hostData = new float[1024];
buffer.Upload(hostData);

// Download to CPU (blocking)
float[] result = new float[1024];
buffer.Download(result);

// Download async
float[] result = await buffer.DownloadAsync();
```

### Views

```csharp
// Slice (view into existing buffer)
var slice = buffer.Slice(offset: 100, count: 200);

// Reinterpret (same memory, different type)
GpuBuffer<byte> bytes = buffer.Reinterpret<byte>();
```

### Buffer States

```csharp
public enum BufferState
{
    Valid,
    Uninitialized,
    Released
}
```

---

## AppendBuffer\<T\>

A specialized buffer for GPU-side append operations (streaming output). Uses composition: wraps a `GpuBuffer<T>` for data and a `GpuBuffer<uint>` for the atomic counter. Kernels append elements via `atomicAdd` on the counter, then write to `Data[index]`.

### Structure

```
┌────────────────────────────────────────────┐
│         Data: GpuBuffer<T>                 │
│  [elem0][elem1][elem2][...][    unused    ]│
└────────────────────────────────────────────┘
                    ▲
                    │ CurrentCount
┌────────────────────┐
│Counter: GpuBuffer<uint>│  ← GPU-side atomic counter (single uint32)
└────────────────────┘
```

### IAppendBuffer Interface

Type-erased interface for systems that track append buffers without knowing `T`:

```csharp
public interface IAppendBuffer
{
    GpuBuffer CounterBuffer { get; }   // The single-uint counter buffer
    CUdeviceptr CounterPointer { get; } // Shortcut: CounterBuffer.Pointer
    int MaxCapacity { get; }
    uint ReadCount();                   // Blocking readback of counter value
}
```

`GraphCompiler` and `CudaEngine` use `IAppendBuffer` to generate memset nodes and read back counts without knowing the element type.

### C# API

```csharp
public sealed class AppendBuffer<T> : IAppendBuffer, IDisposable where T : unmanaged
{
    public GpuBuffer<T> Data { get; }           // Element storage
    public GpuBuffer<uint> Counter { get; }     // Atomic counter (1 element)
    public int MaxCapacity { get; }

    // IAppendBuffer (type-erased)
    GpuBuffer IAppendBuffer.CounterBuffer => Counter;
    CUdeviceptr IAppendBuffer.CounterPointer => Counter.Pointer;
    uint IAppendBuffer.ReadCount() => ReadCount();

    // Typed API
    public uint ReadCount();                    // Blocking: downloads counter from GPU
    public Task<uint> ReadCountAsync();         // Async variant
}
```

### Acquisition

```csharp
// BufferPool allocates both the data buffer and the 1-element counter buffer
var particles = pool.AcquireAppend<Particle>(maxCapacity: 100000, BufferLifetime.Graph);

// Access the two underlying buffers
CUdeviceptr dataPtr = particles.Data.Pointer;
CUdeviceptr counterPtr = particles.Counter.Pointer;
```

### Counter Reset

Counter reset is **not** done manually. The `GraphCompiler` automatically inserts a `cuGraphAddMemsetNode` for each append buffer's counter before kernel execution. This ensures the counter is zeroed at the start of every graph launch. See `GRAPH-COMPILER.md` for the memset node phase.

### Readback

After graph launch, `CudaEngine` automatically reads back append counts and stores them in `BlockDebugInfo.AppendCounts`:

```csharp
// CudaEngine does this after Launch():
foreach (var appendInfo in description.AppendBuffers)
{
    uint count = appendInfo.Buffer.ReadCount();
    debugInfo.AppendCounts[appendInfo.PortName] = count;
}

// Block's VL Update() reads it for tooltip display:
var counts = block.DebugInfo.AppendCounts;  // Dictionary<string, uint>
```

---

## BufferShape

Describes multi-dimensional buffer layout.

### Construction

```csharp
var shape1d = new BufferShape(1024);
var shape2d = new BufferShape(width: 1920, height: 1080);
var shape3d = new BufferShape(64, 64, 64);
var shape = new BufferShape(dims: new[] { 1920, 1080 }, strides: new[] { 1, 1920 });
```

### Operations

```csharp
var transposed = shape.Transpose();
var flat = shape.Flatten();
var strided = shape.WithStride(new[] { 1, 2048 });
```

---

## BufferPool

Manages GPU memory allocation with power-of-2 bucketing for fast reuse. This is a custom implementation — VL's `ResourceProvider` pooling is not used internally because GPU memory management requires CUDA-specific knowledge (allocation costs, stream synchronization, power-of-2 bucketing).

### Interop Boundary

When buffers leave the CUDA system (e.g. to VL.Stride), they are wrapped in `IResourceProvider<GpuBuffer<T>>` for VL-compatible lifetime management. See `GRAPHICS-INTEROP.md` for details.

```
Internal:  BufferPool.Acquire() → GpuBuffer<T>              (own pooling)
External:  ResourceProvider.Return(buffer, dispose) → IRP   (VL-compatible lifetime)
```

### Bucket Strategy

```
Size        Bucket
─────────────────────
<= 256B      → 256B
<= 512B      → 512B
<= 1KB       → 1KB
...
<= 1GB       → 1GB
> 1GB        → exact size
```

### Acquire/Release

```csharp
var buffer = pool.Acquire<float>(elementCount: 1024, BufferLifetime.Graph);
var buffer2d = pool.Acquire<float>(new BufferShape(64, 64), BufferLifetime.Graph);
var append = pool.AcquireAppend<Particle>(maxCapacity: 10000, BufferLifetime.Graph);
pool.Release(buffer);
```

### Statistics

```csharp
long totalBytes = pool.TotalAllocatedBytes;
long usedBytes = pool.CurrentlyUsedBytes;
int bufferCount = pool.BufferCount;
```
