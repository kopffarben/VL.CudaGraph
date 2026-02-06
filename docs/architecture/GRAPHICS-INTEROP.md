# Graphics Interop

## Overview

VL.Cuda needs to share GPU resources with VL.Stride for visualization. This enables:
- Rendering CUDA-computed particles in Stride
- Using Stride textures as CUDA kernel inputs
- Zero-copy sharing (no CPU roundtrip)

```
┌─────────────────┐         ┌─────────────────┐
│   VL.Cuda       │ ◄─────► │   VL.Stride     │
│                 │  Share  │                 │
│  GpuBuffer<T>   │         │  DX11 Buffer    │
│  CudaTexture2D  │         │  DX11 Texture   │
│  CudaTexture3D  │         │  DX11 Texture   │
└─────────────────┘         └─────────────────┘
         │                           │
         └───────────┬───────────────┘
                     │
                     ▼
              GPU Memory
            (shared, no copy)
```

---

## ResourceProvider Boundary

This is the **only** place in the system where VL's `IResourceProvider<T>` is used. Internal CUDA buffer handling uses the custom BufferPool. At the Stride boundary, buffers are wrapped for VL-compatible lifetime management.

### Boundary Map

| Boundary | Direction | Mechanism |
|----------|-----------|-----------|
| Block ↔ Block | internal | Raw `OutputHandle<GpuBuffer<T>>`, no ResourceProvider |
| **CUDA → Stride** | **out** | **Wrap in `IResourceProvider<GpuBuffer<T>>`** |
| **Stride → CUDA** | **in** | **Consume `IResourceHandle`, release at frame end** |
| CUDA → CPU (Download) | out | Synchronous copy, no ResourceProvider |
| External → CUDA | in | `BufferLifetime.External`, no ResourceProvider |

### CUDA → Stride (wrapping outgoing buffers)

```csharp
// ToDX11Buffer node wraps internal buffer for Stride consumption
public IResourceProvider<GpuBuffer<T>> WrapForStride(GpuBuffer<T> buffer)
{
    return ResourceProvider.Return(buffer, disposeAction: b => Pool.Release(b));
}
```

Stride nodes receive an `IResourceProvider<GpuBuffer<T>>` and obtain handles via `.GetHandle()`. When the handle is disposed, the buffer returns to the pool.

### Stride → CUDA (consuming incoming resources)

```csharp
// FromDX11Texture node consumes a Stride resource
public void UseStrideTexture(IResourceProvider<SharedTexture2D<T>> provider)
{
    using var handle = provider.GetHandle();
    var texture = handle.Resource;
    texture.MapForCuda();
    // ... use in kernel ...
    texture.UnmapFromCuda();
}
```

### IRefCounter Registration

For external consumers that need ref-counting on CUDA buffers:

```csharp
// Registered once at CudaEngine startup
appHost.Factory.RegisterService<GpuBuffer, IRefCounter<GpuBuffer>>(
    _ => new GpuBufferRefCounter(pool)
);
```

This allows VL nodes outside the CUDA system to safely hold references to GPU buffers.

---

## Requirements

### Target Renderer
- **Current**: VL.Stride uses DirectX 11
- **Future**: DirectX 12 (when Stride migrates)

### Sharing Directions

| Direction | Use Case |
|-----------|----------|
| CUDA → Stride | Render particles, visualize compute results |
| Stride → CUDA | Use textures as kernel inputs |

### Data Types

| CUDA Type | DX11 Type |
|-----------|-----------|
| `GpuBuffer<T>` | `ID3D11Buffer` |
| `CudaTexture2D<T>` | `ID3D11Texture2D` |
| `CudaTexture3D<T>` | `ID3D11Texture3D` |

### Critical Requirements

- ✅ **Zero-copy sharing**: No CPU roundtrip
- ✅ **Buffer sharing**: Not just textures
- ✅ **Bidirectional**: Both directions supported

---

## CUDA-DX11 Interop Mechanism

CUDA provides native DirectX 11 interop through the CUDA Driver API.

### API Functions

```c
// Register a DX11 resource with CUDA
cuGraphicsD3D11RegisterResource(
    CUgraphicsResource* pCudaResource,
    ID3D11Resource* pD3DResource,
    unsigned int Flags
);

// Map resources for CUDA access
cuGraphicsMapResources(
    unsigned int count,
    CUgraphicsResource* resources,
    CUstream stream
);

// Get device pointer for buffers
cuGraphicsResourceGetMappedPointer(
    CUdeviceptr* pDevPtr,
    size_t* pSize,
    CUgraphicsResource resource
);

// Get CUDA array for textures
cuGraphicsSubResourceGetMappedArray(
    CUarray* pArray,
    CUgraphicsResource resource,
    unsigned int arrayIndex,
    unsigned int mipLevel
);

// Unmap when CUDA is done
cuGraphicsUnmapResources(
    unsigned int count,
    CUgraphicsResource* resources,
    CUstream stream
);
```

### Registration Flags

```csharp
public enum CUgraphicsRegisterFlags
{
    None = 0,
    ReadOnly = 1,    // CUDA will only read
    WriteDiscard = 2 // CUDA will write, DX11 content discarded
}
```

---

## C# Interface

### IDX11SharedResource

```csharp
public interface IDX11SharedResource : IDisposable
{
    // The underlying DX11 resource
    IntPtr DX11Resource { get; }

    // CUDA device pointer (valid only when mapped)
    CUdeviceptr CudaPointer { get; }

    // Map for CUDA access
    void MapForCuda(CudaStream? stream = null);

    // Unmap (return to DX11)
    void UnmapFromCuda(CudaStream? stream = null);

    // State
    bool IsMappedForCuda { get; }
}
```

### CudaDX11Interop Static Class

```csharp
public static class CudaDX11Interop
{
    /// <summary>
    /// Register a DX11 buffer for CUDA access
    /// </summary>
    public static SharedBuffer<T> RegisterBuffer<T>(
        IntPtr dx11Buffer,
        CudaContext ctx,
        CUgraphicsRegisterFlags flags = CUgraphicsRegisterFlags.None)
        where T : unmanaged;

    /// <summary>
    /// Register a DX11 Texture2D for CUDA access
    /// </summary>
    public static SharedTexture2D<T> RegisterTexture2D<T>(
        IntPtr dx11Texture,
        CudaContext ctx,
        CUgraphicsRegisterFlags flags = CUgraphicsRegisterFlags.None)
        where T : unmanaged;

    /// <summary>
    /// Register a DX11 Texture3D for CUDA access
    /// </summary>
    public static SharedTexture3D<T> RegisterTexture3D<T>(
        IntPtr dx11Texture,
        CudaContext ctx,
        CUgraphicsRegisterFlags flags = CUgraphicsRegisterFlags.None)
        where T : unmanaged;

    /// <summary>
    /// Create a new buffer that can be shared between CUDA and DX11
    /// </summary>
    public static SharedBuffer<T> CreateSharedBuffer<T>(
        int elementCount,
        CudaContext ctx,
        ID3D11Device dx11Device)
        where T : unmanaged;
}
```

### SharedBuffer\<T\>

```csharp
public class SharedBuffer<T> : IDX11SharedResource, IDisposable where T : unmanaged
{
    public IntPtr DX11Resource { get; }
    public CUdeviceptr CudaPointer { get; private set; }

    public int ElementCount { get; }
    public long SizeInBytes { get; }

    private CUgraphicsResource _cudaResource;
    private bool _isMapped;

    public bool IsMappedForCuda => _isMapped;

    public void MapForCuda(CudaStream? stream = null)
    {
        if (_isMapped) return;

        cuGraphicsMapResources(1, ref _cudaResource, stream?.Handle ?? IntPtr.Zero);

        CUdeviceptr ptr;
        size_t size;
        cuGraphicsResourceGetMappedPointer(out ptr, out size, _cudaResource);

        CudaPointer = ptr;
        _isMapped = true;
    }

    public void UnmapFromCuda(CudaStream? stream = null)
    {
        if (!_isMapped) return;

        cuGraphicsUnmapResources(1, ref _cudaResource, stream?.Handle ?? IntPtr.Zero);

        CudaPointer = default;
        _isMapped = false;
    }

    public void Dispose()
    {
        if (_isMapped) UnmapFromCuda();
        cuGraphicsUnregisterResource(_cudaResource);
    }
}
```

---

## Usage Patterns

### Pattern 1: CUDA Compute → Stride Render

```csharp
// 1. Get DX11 buffer from Stride
var strideBuffer = strideRenderer.GetParticleBuffer();

// 2. Register with CUDA
var shared = CudaDX11Interop.RegisterBuffer<Particle>(
    strideBuffer.NativePointer,
    cudaCtx,
    CUgraphicsRegisterFlags.WriteDiscard);

// 3. Map for CUDA
shared.MapForCuda();

// 4. Use in CUDA kernel
particleKernel.Run(shared.CudaPointer, count);

// 5. Unmap (back to DX11)
shared.UnmapFromCuda();

// 6. Stride renders the buffer
strideRenderer.DrawParticles();
```

### Pattern 2: Stride Texture → CUDA Input

```csharp
// 1. Get DX11 texture from Stride
var strideTexture = strideRenderer.GetInputTexture();

// 2. Register with CUDA
var shared = CudaDX11Interop.RegisterTexture2D<float4>(
    strideTexture.NativePointer,
    cudaCtx,
    CUgraphicsRegisterFlags.ReadOnly);

// 3. Map for CUDA
shared.MapForCuda();

// 4. Use in CUDA kernel
imageProcessKernel.Run(shared.CudaArray, width, height, output);

// 5. Unmap
shared.UnmapFromCuda();
```

### Pattern 3: Persistent Sharing

For frequently shared resources, keep them registered:

```csharp
public class ParticleSystem : IDisposable
{
    private SharedBuffer<Particle> _particles;

    public void Initialize(StrideDevice stride, CudaContext cuda)
    {
        // Create buffer once
        _particles = CudaDX11Interop.CreateSharedBuffer<Particle>(
            elementCount: MaxParticles,
            cuda,
            stride.Device);
    }

    public void Update()
    {
        // Map
        _particles.MapForCuda();

        // CUDA compute
        _emitter.Run(_particles.CudaPointer);
        _forces.Run(_particles.CudaPointer);
        _integrate.Run(_particles.CudaPointer);

        // Unmap
        _particles.UnmapFromCuda();
    }

    public void Render(StrideRenderer renderer)
    {
        // DX11 render (buffer is already accessible)
        renderer.DrawInstancedParticles(_particles.DX11Resource, _particleCount);
    }

    public void Dispose()
    {
        _particles.Dispose();
    }
}
```

---

## Synchronization

### Issue

CUDA and DX11 run asynchronously on the GPU. We need to ensure:
- CUDA finishes before DX11 reads
- DX11 finishes before CUDA writes

### Solution: Map/Unmap

The `MapResources` and `UnmapResources` calls provide implicit synchronization:
- `MapForCuda`: Waits for pending DX11 work, then CUDA can access
- `UnmapFromCuda`: Waits for pending CUDA work, then DX11 can access

### Advanced: Explicit Sync

For finer control (later):

```csharp
// Create DX11 fence
var dx11Fence = dx11Device.CreateFence();

// Signal from CUDA
cuda.SignalExternalFence(dx11Fence, value);

// Wait in DX11
dx11Context.Wait(dx11Fence, value);
```

---

## Texture Format Mapping

| DX11 Format | CUDA Element Type |
|-------------|-------------------|
| `DXGI_FORMAT_R32_FLOAT` | `float` |
| `DXGI_FORMAT_R32G32_FLOAT` | `float2` |
| `DXGI_FORMAT_R32G32B32A32_FLOAT` | `float4` |
| `DXGI_FORMAT_R8_UNORM` | `uchar` (normalize in kernel) |
| `DXGI_FORMAT_R8G8B8A8_UNORM` | `uchar4` (normalize in kernel) |
| `DXGI_FORMAT_R16_FLOAT` | `half` |
| `DXGI_FORMAT_R16G16B16A16_FLOAT` | `half4` |

---

## VL Patch Integration

```
┌─────────────────────────────────────────────────────────────────┐
│ CUDA Particle System                                            │
│                                                                  │
│  ┌────────────┐      ┌────────────┐      ┌──────────────────┐  │
│  │  Emitter   │─────▶│   Forces   │─────▶│ ToDX11Buffer     │  │
│  └────────────┘      └────────────┘      └────────┬─────────┘  │
│                                                    │            │
└────────────────────────────────────────────────────┼────────────┘
                                                     │
                                                     ▼
                                   IResourceProvider<GpuBuffer<T>>
                                      (VL-compatible lifetime)
                                                     │
                                                     ▼
                                            ┌────────────────┐
                                            │  Stride Render │
                                            └────────────────┘
```

The `ToDX11Buffer` node wraps the internal `GpuBuffer<T>` in an `IResourceProvider` before handing it to Stride. This is the ResourceProvider boundary — see `VL-INTEGRATION.md` for the full boundary design.

> **CapturedNode outputs follow the same path.** A CapturedNode (cuBLAS, cuFFT) produces standard `GpuBuffer<T>` outputs that are indistinguishable from KernelNode outputs. The ToDX11Buffer bridge works identically regardless of which kernel source produced the buffer. See `KERNEL-SOURCES.md` for the three kernel sources.

### Block: ToDX11Buffer

```csharp
public class ToDX11BufferBlock : ICudaBlock
{
    private SharedBuffer<T> _shared;

    public void Setup(BlockBuilder builder)
    {
        // Input from CUDA graph
        var cudaIn = builder.Input<T>("Particles", ...);

        // This block bridges to DX11
        // Output is wrapped as IResourceProvider for Stride
    }

    // Special handling in graph compiler:
    // After CUDA execution, the buffer is available for DX11
}
```

---

## Later Considerations

### DX12 Migration

When Stride moves to DX12, the interop changes:

```csharp
// DX12 uses external memory import
cuImportExternalMemory(
    CUexternalMemory* extMem,
    const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc);

cuExternalMemoryGetMappedBuffer(
    CUdeviceptr* devPtr,
    CUexternalMemory extMem,
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc);
```

### Resource Ownership

**Question**: Who allocates the resource?

| Approach | Pros | Cons |
|----------|------|------|
| DX11 allocates | Works with existing Stride resources | Must adapt to Stride's lifetime |
| CUDA allocates | Full control | Must expose to DX11 |
| Shared heap | Clean ownership | More complex |

**Current approach**: Support both, prefer DX11-allocated for integration.

### Lifetime Management

```csharp
public class SharedResourceManager : IDisposable
{
    private List<IDX11SharedResource> _resources = new();

    public T Register<T>(IntPtr dx11Resource, ...) where T : IDX11SharedResource
    {
        var shared = CreateShared<T>(dx11Resource, ...);
        _resources.Add(shared);
        return shared;
    }

    public void Dispose()
    {
        foreach (var res in _resources)
            res.Dispose();
        _resources.Clear();
    }
}
```

---

## Error Handling

```csharp
public class InteropException : Exception
{
    public InteropError Error { get; }
    public IntPtr Resource { get; }
}

public enum InteropError
{
    ResourceNotRegistered,
    ResourceAlreadyMapped,
    ResourceNotMapped,
    IncompatibleFormat,
    DeviceMismatch,
    DX11Error,
    CUDAError
}
```

---

## Performance Tips

1. **Register once, use many times**: Registration has overhead
2. **Batch map/unmap**: Map multiple resources in one call
3. **Use streams**: Overlap CUDA compute with transfer
4. **Consider double-buffering**: CUDA writes buffer A while DX11 renders buffer B
5. **Profile**: Measure actual overhead of map/unmap
