# Core Runtime

## CudaContext

The `CudaContext` is the main entry point for all GPU operations. It manages:
- CUDA device selection
- Block registration and connections
- Graph compilation and execution
- Buffer pool access
- Debug information

### Creation

```csharp
// Simple creation (device 0)
using var ctx = CudaContext.Create();

// With options
using var ctx = CudaContext.Create(new CudaContextOptions
{
    DeviceId = 0,
    Profiling = ProfilingLevel.PerBlock,
    BufferPoolInitialSizeMB = 512,
    EnableDebug = true
});
```

### Block Management

```csharp
// Create blocks
var emitter = ctx.CreateBlock<SphereEmitterBlock>();
var forces = ctx.CreateBlock<ForcesBlock>();

// Connect blocks (for runtime/UI scenarios)
ctx.Connect(emitter.Id, "Particles", forces.Id, "Particles");

// Remove blocks
ctx.RemoveBlock(emitter.Id);
```

### Execution Cycle

```csharp
// Compile (when structure changes)
if (ctx.NeedsRecompile)
{
    ctx.Compile();
}

// Execute (every frame)
ctx.Execute();
```

### Child Contexts

Composite blocks create child contexts to encapsulate their internal graphs:

```csharp
// Inside a composite block's Setup method
var childCtx = builder.Context.CreateChildContext(BlockId, "MyBlock");
```

---

## CudaStream

Streams enable asynchronous operations and can be used for:
- Overlapping compute with data transfer
- Multiple independent execution paths

```csharp
// Create stream
using var stream = CudaStream.Create(ctx);

// Execute on stream
ctx.Execute(stream);

// Sync (blocking)
stream.Synchronize();

// Sync (async)
await stream.SynchronizeAsync();
```

---

## GpuBuffer<T>

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
    // Unsigned integers
    U8, U16, U32, U64,
    
    // Signed integers
    S8, S16, S32, S64,
    
    // Floating point
    F16, F32, F64,
    
    // Special
    BF16,              // Brain float 16
    Complex64,         // float + float
    Complex128         // double + double
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
    Valid,          // Contains valid data
    Uninitialized,  // Allocated but not written
    Released        // Returned to pool, do not use
}
```

---

## AppendBuffer<T>

A specialized buffer for GPU-side append operations (streaming output).

### Structure

```
┌────────────────────────────────────────────┐
│              Data Buffer                    │
│  [elem0][elem1][elem2][...][    unused    ]│
└────────────────────────────────────────────┘
                    ▲
                    │ CurrentCount
┌────────────────────┐
│  Counter (uint32)  │  ← GPU-side atomic counter
└────────────────────┘
```

### Usage in Kernels (PTX)

```ptx
// Kernel appends an element
.global .u32 counter;
.global .f32 data[];

// Atomic increment, get old value as index
atom.global.inc.u32 %idx, [counter], %max_capacity;
st.global.f32 [data + %idx*4], %value;
```

### C# API

```csharp
// Create
var particles = pool.AcquireAppend<Particle>(maxCapacity: 100000, BufferLifetime.Graph);

// Get count (sync readback - use sparingly!)
int count = particles.ReadCount();

// Get count async
int count = await particles.ReadCountAsync();

// Reset for next frame
particles.ResetCount();
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `MaxCapacity` | `int` | Maximum elements |
| `CountPointer` | `CUdeviceptr` | Pointer to counter |

---

## BufferShape

Describes multi-dimensional buffer layout.

### Construction

```csharp
// 1D
var shape1d = new BufferShape(1024);

// 2D (row-major by default)
var shape2d = new BufferShape(width: 1920, height: 1080);

// 3D
var shape3d = new BufferShape(64, 64, 64);

// With explicit strides
var shape = new BufferShape(
    dims: new[] { 1920, 1080 },
    strides: new[] { 1, 1920 }  // Column-major
);
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Rank` | `int` | Number of dimensions (1-3) |
| `Dim0`, `Dim1`, `Dim2` | `int` | Dimension sizes |
| `Stride0`, `Stride1`, `Stride2` | `int` | Element strides |
| `TotalElements` | `long` | Total element count |

### Operations

```csharp
// Transpose (swap dims/strides)
var transposed = shape.Transpose();

// Flatten to 1D
var flat = shape.Flatten();

// Custom strides
var strided = shape.WithStride(new[] { 1, 2048 });
```

---

## BufferPool

Manages GPU memory allocation with power-of-2 bucketing for fast reuse.

### Bucket Strategy

```
Size        Bucket
─────────────────────
<= 256B      → 256B
<= 512B      → 512B
<= 1KB       → 1KB
<= 2KB       → 2KB
...
<= 1GB       → 1GB
> 1GB        → exact size
```

### Acquire/Release

```csharp
// Acquire from pool
var buffer = pool.Acquire<float>(elementCount: 1024, BufferLifetime.Graph);

// Acquire with shape
var buffer2d = pool.Acquire<float>(new BufferShape(64, 64), BufferLifetime.Graph);

// Acquire append buffer
var append = pool.AcquireAppend<Particle>(maxCapacity: 10000, BufferLifetime.Region);

// Release back to pool (usually automatic)
pool.Release(buffer);
```

### Statistics

```csharp
long totalBytes = pool.TotalAllocatedBytes;
long usedBytes = pool.CurrentlyUsedBytes;
int bufferCount = pool.BufferCount;

// Per-bucket stats
foreach (var (bucketSize, count) in pool.BucketStats)
{
    Console.WriteLine($"Bucket {bucketSize}B: {count} buffers");
}
```

### Implementation Notes

1. **Power-of-2 bucketing**: Reduces fragmentation
2. **Best-fit within bucket**: Finds smallest available buffer
3. **Lazy allocation**: Only allocates when needed
4. **Reference counting**: Tracks buffer usage across graph
5. **Automatic release**: Region buffers freed after scope

### Memory Layout

```
Pool Structure:

Bucket 256B:    [buf][buf][buf][   free   ]
Bucket 512B:    [buf][  free  ]
Bucket 1KB:     [buf][buf]
Bucket 2KB:     [   free   ]
...

Each buffer:
┌──────────────────────────────────────┐
│ Header (internal)                     │
│ - Id: Guid                           │
│ - ElementType: DType                 │
│ - RefCount: int                      │
│ - State: BufferState                 │
├──────────────────────────────────────┤
│ GPU Memory (CUdeviceptr)              │
│ [data data data data ...]            │
└──────────────────────────────────────┘
```
