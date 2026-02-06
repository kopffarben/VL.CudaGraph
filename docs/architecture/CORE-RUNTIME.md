# Core Runtime

## CudaContext

The `CudaContext` is the shared state object for all GPU operations. It is created by `CudaEngine` and passed to blocks via VL pin connections. It manages:
- Block registry and connection tracking
- Dirty-tracking (structure + parameters)
- Buffer pool access
- Debug information

> **Note:** CudaContext is created internally by CudaEngine. Users don't create it directly. See `EXECUTION-MODEL.md` for the full lifecycle.

### Creation (by CudaEngine)

```csharp
// CudaEngine creates the context:
var engine = new CudaEngine(new CudaContextOptions
{
    DeviceId = 0,
    Profiling = ProfilingLevel.PerBlock,
    BufferPoolInitialSizeMB = 512,
    EnableDebug = true
});

// Blocks receive context via VL pin:
var ctx = engine.Context;
```

### Block Registration

```csharp
// Block constructor registers:
ctx.RegisterBlock(this);    // → structureDirty = true

// Block Dispose unregisters:
ctx.UnregisterBlock(id);    // → structureDirty = true

// VL link drawing triggers:
ctx.Connect(srcId, "Out", tgtId, "In");     // → structureDirty = true
ctx.Disconnect(srcId, "Out", tgtId, "In");  // → structureDirty = true
```

### Dirty Tracking

```csharp
// CudaEngine checks each frame:
if (ctx.IsStructureDirty)    // → Cold Rebuild
if (ctx.AreParametersDirty)  // → Hot/Warm Update
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
engine.Update(stream);

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

### C# API

```csharp
var particles = pool.AcquireAppend<Particle>(maxCapacity: 100000, BufferLifetime.Graph);

int count = particles.ReadCount();
int count = await particles.ReadCountAsync();
particles.ResetCount();
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

Manages GPU memory allocation with power-of-2 bucketing for fast reuse.

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
var append = pool.AcquireAppend<Particle>(maxCapacity: 10000, BufferLifetime.Region);
pool.Release(buffer);
```

### Statistics

```csharp
long totalBytes = pool.TotalAllocatedBytes;
long usedBytes = pool.CurrentlyUsedBytes;
int bufferCount = pool.BufferCount;
```
