# VL Integration

## Execution Model

> **See `EXECUTION-MODEL.md` for the full execution model.**
> **See `KERNEL-SOURCES.md` for the three kernel sources (Filesystem PTX, Patchable Kernels, Library Calls).**
> **See `../vl.reference/` for VL platform reference** (.vl file format, StandardLibs APIs, layout conventions).

The key insight: Blocks are **passive** — they describe GPU work but never execute it. A single **CudaEngine** node compiles and launches the CUDA Graph each frame.

```
VL Patch:

┌──────────────┐
│  CudaEngine  │──── Context ──────────────────────┐
│  (active)    │                                    │
└──────────────┘                                    │
                                                    │
       ┌────────────────────────────────────────────┤
       │              │              │              │
       ▼              ▼              ▼              │
┌──────────┐   ┌──────────┐  ┌──────────┐         │
│ Emitter  │──▶│ Forces   │─▶│Integrate │         │
│(passive) │   │(passive) │  │(passive) │         │
│ ctx ◀────────│ ctx ◀───────│ ctx ◀────┘─────────┘
└──────────┘   └──────────┘  └──────────┘
```

The CudaContext flows from Engine to Blocks via VL pin connections. Blocks register themselves in their constructor and unregister on Dispose.

---

## VL.Core Integration

VL.Cuda integrates with VL.Core at specific, well-defined points. The integration is moderate — enough to feel native, but not so deep that we depend on VL internals that may change.

### What We Use from VL.Core

| VL.Core Feature | Our Usage |
|----------------|-----------|
| **NodeContext** | Block/Engine identity (UniqueId), logging (ILogger), AppHost access |
| **IVLRuntime** | Error/warning routing to VL nodes (AddMessage, AddPersistentMessage) |
| **AppHost.TakeOwnership** | Ensures CudaEngine cleanup on app shutdown |
| **ServiceRegistry** | App-global singletons (CudaDeviceInfo, CudaDriverVersion) |
| **IResourceProvider\<T\>** | Buffer wrapping at Stride interop boundary only |
| **IRefCounter\<T\>** | Registration for external buffer consumers (Stride) |

### What We Do NOT Use

| VL.Core Feature | Reason |
|----------------|--------|
| **ResourceProvider pooling** | GPU memory needs CUDA-specific pooling (power-of-2, stream sync) |
| **Using\<T\>/Producing\<T\>** | Buffer lifetime is Graph/Region scoped, not VL-frame scoped |
| **FrameClock subscriptions** | CudaEngine is a ProcessNode — VL calls Update() each frame |
| **Channel system** | Parameters flow through VL pins, not VL channels |

### NodeContext — Constructor Pattern

Every VL-created instance receives a `NodeContext` as its first constructor parameter. VL injects this automatically — VL users never see it in the patch. C# users see it in the signature.

```csharp
// VL injects nodeContext automatically. C# users provide it explicitly.
public class EmitterBlock : ICudaBlock
{
    public EmitterBlock(NodeContext nodeContext, CudaContext cudaContext)
    {
        _nodeContext = nodeContext;
        _logger = nodeContext.GetLogger();

        Setup(new BlockBuilder(cudaContext, this));
        cudaContext.RegisterBlock(this);
    }

    public void Dispose()
    {
        _cudaContext.UnregisterBlock(this.Id);
    }
}
```

The NodeContext provides:
- **Identity**: `nodeContext.Path.Stack.Peek()` → `UniqueId` for addressing errors to the correct VL node
- **Logging**: `nodeContext.GetLogger()` → standard `ILogger`
- **App access**: `nodeContext.AppHost` → ServiceRegistry, IVLRuntime

### Diagnostics — IVLRuntime

Errors and warnings are routed to VL nodes via `IVLRuntime`. VL displays these as colored node borders (orange = warning, red = error). This replaces a custom diagnostics system.

```csharp
// CudaEngine routes errors to VL nodes after each frame
private void ReportDiagnostics()
{
    foreach (var reg in _blockRegistry.Values)
    {
        var diag = GetDiagnostics(reg.Block.Id);
        var elementId = reg.NodeContext.Path.Stack.Peek();

        if (diag.HasError)
        {
            _persistentMessages[reg.Block.Id] ??=
                _runtime.AddPersistentMessage(
                    new Message(elementId, MessageSeverity.Error, diag.ErrorText));
        }
        else
        {
            // Clear persistent error if resolved
            _persistentMessages[reg.Block.Id]?.Dispose();
            _persistentMessages[reg.Block.Id] = null;

            if (diag.HasWarning)
                _runtime.AddMessage(elementId, diag.WarningText, MessageSeverity.Warning);
        }
    }
}
```

For timing and buffer statistics, blocks use `ToString()` / DebugInfo — this is the standard VL tooltip mechanism that only runs when the user hovers a node. See `EXECUTION-MODEL.md` for profiling details.

### ResourceProvider Boundary

VL's `IResourceProvider<T>` is used **only** at the Stride interop boundary — nowhere else.

| Boundary | Direction | Mechanism |
|----------|-----------|-----------|
| Block ↔ Block | internal | Raw `OutputHandle<GpuBuffer<T>>`, no ResourceProvider |
| CUDA → Stride | out | Wrap in `IResourceProvider<GpuBuffer<T>>` |
| Stride → CUDA | in | Consume `IResourceHandle`, release at frame end |
| CUDA → CPU (Download) | out | Synchronous copy, no ResourceProvider |
| External → CUDA | in | `BufferLifetime.External`, no ResourceProvider |

See `GRAPHICS-INTEROP.md` for the full interop design.

### Multi-Engine Constraint

A CUDA Graph's nodes must all reside on a single device. Therefore:
- Each CudaEngine = one CudaContext = one CUDA device
- Two CudaEngines in the same patch are completely separate worlds
- Cross-engine buffer connections are physically impossible
- The GraphCompiler validates this and reports a clear error via IVLRuntime

---

## Source-Package Setup (Validated)

VL.CudaGraph is distributed as a **source-package** — a folder with a `.vl` file, `lib/` DLLs, and a `deployment/` nuspec. VL discovers it via `--package-repositories` pointing to the parent folder.

### File Structure

```
VL.CudaGraph/
  VL.CudaGraph.vl              ← VL document (dependencies, category)
  lib/net8.0/
    VL.Cuda.Core.dll            ← Built DLL (OutputPath in .csproj)
    VL.Cuda.Core.xml            ← XML docs (GenerateDocumentationFile)
    VL.Cuda.Core.deps.json      ← Dependency resolution
  deployment/
    VL.CudaGraph.nuspec         ← Package metadata (for nuget pack)
  help/
    help.xml                    ← Help patches (empty for now)
  src/
    VL.Cuda.Core/               ← C# source
```

### .vl File — Dependencies

The `.vl` file declares dependencies that VL resolves at load time:

```xml
<NugetDependency Location="VL.CoreLib" Version="2025.7.0" />
<PlatformDependency Location="./lib/net8.0/VL.Cuda.Core.dll" IsForward="true" />
<NugetDependency Location="ManagedCuda-13" Version="13.0.64" />
```

Key rules:
- **`PlatformDependency`** for our own DLL — NOT `AssemblyReference`
- **`IsForward="true"`** is required to make types visible in VL's node browser
- **Explicit `NugetDependency` for transitive deps** (ManagedCuda-13) — VL doesn't resolve from `deps.json` alone
- **Don't hand-edit `.vl` files** — VL regenerates all IDs and LanguageVersion on open

### .csproj Output Configuration

```xml
<OutputPath>..\..\lib\net8.0</OutputPath>
<AppendTargetFrameworkToOutputPath>false</AppendTargetFrameworkToOutputPath>
<AppendRuntimeIdentifierToOutputPath>false</AppendRuntimeIdentifierToOutputPath>
<GenerateDocumentationFile>true</GenerateDocumentationFile>
```

- `OutputPath` places DLL directly in `lib/net8.0/` (VL source-package convention)
- `GenerateDocumentationFile` produces XML docs that VL reads for tooltips

### Assembly Attributes

```csharp
// Properties/AssemblyInfo.cs
using VL.Core.Import;

[assembly: ImportAsIs(Namespace = "VL.Cuda.Core")]
```

- `ImportAsIs` makes all public types available in VL
- `Namespace` stripping: `VL.Cuda.Core.Engine.CudaEngine` → `CudaEngine [Engine]` in VL

### ProcessNode Attribute

```csharp
[ProcessNode(HasStateOutput = true)]
public sealed class CudaEngine : IDisposable
```

- `[ProcessNode]` marks a class as a VL process node (Create → Update → Dispose lifecycle)
- `HasStateOutput = true` exposes the instance as an output pin (for flowing CudaContext to blocks)
- Requires `[assembly:ImportAsIs]` to work

### Launch via --package-repositories

```
vvvv.exe --package-repositories "D:\_MBOX\_CODE\_packages"
```

VL scans the path for folders containing `.vl` files and treats them as source packages.

---

## Core Pattern: Handle-Flow

VL (vvvv gamma) is a visual, node-based programming environment. The key principle for VL.Cuda integration is that **data flows visibly through links**.

### The Problem with Mutation

In traditional GPU programming, buffers are often mutated in-place:

```csharp
// BAD for VL: mutation is invisible
void ApplyForces(GpuBuffer particles)
{
    // particles modified in-place
    // no visible output
}
```

In a visual dataflow system, this is confusing because:
- The user doesn't see data flowing
- Side effects are hidden
- The graph structure doesn't match the data flow

### The Solution: Handle-Flow

Every GPU operation takes input handles and produces output handles:

```
VL Patch:

   ┌───────────┐         ┌───────────┐         ┌───────────┐
   │  Emitter  │         │  Forces   │         │ Renderer  │
   │           │ Handle  │           │ Handle  │           │
   │      Out ─┼────────▶┼─ In   Out─┼────────▶┼─ In       │
   └───────────┘         └───────────┘         └───────────┘

Each link carries an OutputHandle<GpuBuffer<T>>
The visual flow matches the actual data dependencies
```

### Handle Types

```csharp
// Output from a block
OutputHandle<GpuBuffer<Particle>> particles;

// Input to a block (references an output)
InputHandle<GpuBuffer<Particle>> particleInput;

// Connection
particleInput.Source = particles;  // VL link does this
```

---

## CUDA Delegate Pattern

In VL, a "Delegate" is a pure function region. For CUDA, this means:

```
┌─────────────────────────────────────────────────────────────────┐
│  CUDA Delegate                                                   │
│                                                                  │
│  Inputs → [GPU Operations] → Outputs                             │
│                                                                  │
│  • No state between calls                                        │
│  • All inputs/outputs through pins                               │
│  • Can be called multiple times per frame                        │
│  • Graph compiled once, executed many times                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Delegate vs Process

| Aspect | Delegate | Process |
|--------|----------|---------|
| State | Stateless | Stateful |
| GPU Graph | Compiled once | May need recompile |
| Use case | Pure transforms | Accumulation, feedback |

---

## PinGroups for Dynamic Pins

VL's PinGroups allow nodes to have a variable number of pins. This is essential for CUDA blocks that expose kernel parameters.

### Example: Kernel Node

A kernel might have varying parameters:
```
VectorAdd kernel:      Particles kernel:
  ┌──────────┐           ┌──────────┐
  │VectorAdd │           │Particles │
  │          │           │          │
─▶│ A        │         ─▶│ Positions│
─▶│ B        │         ─▶│ Velocities│
  │ Count    │─scalar  ─▶│ Forces   │
  │     Sum ─│▶          │ DeltaTime│─scalar
  └──────────┘           │ Count    │─scalar
                         │   Pos'  ─│▶
                         │   Vel'  ─│▶
                         └──────────┘
```

### Implementation with PinGroups

```csharp
public class KernelBlock : ICudaBlock, IDisposable
{
    private readonly CudaContext _ctx;

    [Pin(PinGroupKind = PinGroupKind.Collection)]
    public IEnumerable<IBlockPort> Inputs => _inputs;

    [Pin(PinGroupKind = PinGroupKind.Collection)]
    public IEnumerable<IBlockPort> Outputs => _outputs;

    private List<BlockPort> _inputs;
    private List<BlockPort> _outputs;

    public KernelBlock(NodeContext nodeContext, CudaContext ctx, string ptxPath, string entryPoint)
    {
        _ctx = ctx;
        Setup(ptxPath, entryPoint);
        _ctx.RegisterBlock(this);
    }

    private void Setup(string ptxPath, string entryPoint)
    {
        var builder = new BlockBuilder(_ctx, this);
        var kernel = builder.AddKernel(ptxPath, entryPoint);

        // Create pins based on kernel descriptor
        foreach (var param in kernel.Descriptor.Parameters)
        {
            if (param.IsPointer)
            {
                if (param.Direction == ParamDirection.In)
                    _inputs.Add(builder.Input(param.Name, kernel.In(param.Index)));
                else if (param.Direction == ParamDirection.Out)
                    _outputs.Add(builder.Output(param.Name, kernel.Out(param.Index)));
                else // InOut
                {
                    _inputs.Add(builder.Input(param.Name, kernel.In(param.Index)));
                    _outputs.Add(builder.Output(param.Name, kernel.Out(param.Index)));
                }
            }
            else
            {
                // Scalar parameter becomes a config pin
                builder.InputScalar(param.Name, kernel.In(param.Index));
            }
        }

        builder.Commit();
    }

    public void Dispose() => _ctx.UnregisterBlock(this.Id);
}
```

---

## DeltaTime — A Normal VL Pin

DeltaTime is not a special CUDA concept. It flows as a normal scalar parameter from VL's FrameClock node to blocks that need it:

```
VL Patch:

┌───────────┐
│ FrameClock │─── TimeDiff ──▶┌──────────┐
└───────────┘                 │ Simulate │
                              │ DeltaT ◀─┘
                              └──────────┘
```

This is important for multi-machine installations where VL's clock can be set externally. The user decides where time comes from — the CUDA system doesn't impose a time source.

---

## Feedback Pattern

For stateful operations (accumulation, simulation), VL uses explicit feedback:

### FrameDelay Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   ┌────────────┐     ┌────────────┐     ┌────────────┐      │
│   │FrameDelay  │     │  Simulate  │     │            │      │
│   │            │     │            │     │            │      │
│───┤ In    Out ─┼────▶┤ In    Out ─┼────▶┤            │      │
│   └────────────┘     └────────────┘     │            │      │
│         ▲                               │            │      │
│         └───────────────────────────────┼────────────┘      │
│                    feedback link        │                    │
│                                         ▼                    │
│                                    Output                    │
└─────────────────────────────────────────────────────────────┘
```

### PingPong Pattern (Double Buffering)

For operations that read and write the same logical data:

```csharp
public class PingPongBlock : ICudaBlock
{
    private GpuBuffer<T> _bufferA;
    private GpuBuffer<T> _bufferB;
    private bool _ping;

    public GpuBuffer<T> CurrentRead => _ping ? _bufferA : _bufferB;
    public GpuBuffer<T> CurrentWrite => _ping ? _bufferB : _bufferA;

    public void Swap() => _ping = !_ping;
}
```

```
Frame 0: Read A, Write B
Frame 1: Read B, Write A (swapped)
Frame 2: Read A, Write B (swapped)
...
```

---

## InOut Parameters in VL

When a kernel has an InOut parameter (reads and writes same buffer), VL shows it as both input and output:

```
PTX Kernel:
  scale_inplace(float* data, float factor, int n)
  // data is read AND written

VL Node:
  ┌──────────────┐
  │  Scale       │
  │              │
─▶│ Data    Data │▶   // Same buffer, visible flow
─▶│ Factor       │
─▶│ Count        │
  └──────────────┘
```

The graph compiler knows it's the same buffer and doesn't allocate twice.

---

## No Programmatic Node Creation

VL nodes are created in the visual patch, not in code. The CudaContext API supports both:

1. **Design-time (VL Patch)**: User places nodes, draws links
2. **Runtime (Code/UI)**: Programmatic connection for exported apps

```csharp
// Runtime API for UI-driven composition
// Blocks register themselves via constructor
var emitter = new SphereEmitterBlock(nodeContext, ctx);
var forces = new ForcesBlock(nodeContext, ctx);
var renderer = new ParticleRendererBlock(nodeContext, ctx);

// Connect blocks (like drawing a link, but in code)
ctx.Connect(emitter.Id, "Particles", forces.Id, "Particles");
ctx.Connect(forces.Id, "Particles", renderer.Id, "Particles");

// CudaEngine handles compile + execute in its Update()
```

---

## VL-Specific Constraints

### No Inheritance

VL doesn't support class inheritance well. Use:
- Interfaces (`ICudaBlock`)
- Composition (child blocks)
- Delegates (lambdas)

```csharp
// GOOD: Composition
public class ComplexBlock : ICudaBlock
{
    public ComplexBlock(NodeContext nodeContext, CudaContext ctx)
    {
        var builder = new BlockBuilder(ctx, this);
        var child1 = builder.AddChild<SimpleBlock1>();
        var child2 = builder.AddChild<SimpleBlock2>();
        builder.ConnectChildren(child1.Output, child2.Input);
        builder.Commit();
        ctx.RegisterBlock(this);
    }
}

// BAD: Inheritance (doesn't work well in VL)
public class DerivedBlock : BaseBlock { ... }
```

### Serialization

For saving/loading configurations:

```csharp
// Save
var model = ctx.GetModel();
model.SaveToFile("particle_system.json");

// Load
var model = GraphModel.LoadFromFile("particle_system.json");
ctx.LoadModel(model);
```

The model contains:
- Block types and IDs
- Parameter values
- Connections between blocks

It does NOT contain:
- Internal kernel wiring (that's in the block code)
- Buffer contents (runtime data)

---

## Debug Integration

VL shows debug information as tooltips on nodes and pins. Two separate mechanisms:

### Errors/Warnings → IVLRuntime

Routed by CudaEngine to VL's node display. Errors turn nodes red, warnings turn them orange:

| Source | Severity | Example |
|--------|----------|---------|
| PTX load failure | Error | "Failed to load kernel.ptx: file not found" |
| CUDA launch error | Error | "CUDA_ERROR_LAUNCH_FAILED in particle_emit" |
| Graph validation | Error | "Type mismatch: GpuBuffer\<float4\> → GpuBuffer\<float3\>" |
| Cross-engine connection | Error | "Block uses different CudaContext than engine" |
| Missing connection | Warning | "Required input 'Particles' not connected" |
| Buffer near capacity | Warning | "AppendBuffer at 95% capacity (95000/100000)" |

### Timing/Stats → ToString/DebugInfo

Read from the async profiling cache, only when user hovers a node:

```
Node Tooltip (Block.ToString):
┌─────────────────────────────────────────┐
│  ParticleEmitter                        │
│  ⏱ 0.42 ms (avg: 0.38 ms)              │
│  📦 Particles: 45,231 / 100,000        │
│  ✅ OK                                  │
└─────────────────────────────────────────┘

Link Tooltip (OutputHandle.ToString):
┌─────────────────────────────────────────┐
│  Particles (GpuBuffer<float4>)          │
│  Count: 45,231 elements                 │
│  Size: 724 KB (1024 KB allocated)       │
└─────────────────────────────────────────┘

Engine Tooltip (CudaEngine.ToString):
┌─────────────────────────────────────────┐
│  CudaEngine: 4 blocks, 12 kernels      │
│  Launch: 2.1ms                          │
│  Pool: 12 MB / 64 MB                   │
└─────────────────────────────────────────┘
```

---

## VL Naming Conventions & Design Guidelines

> **Based on The-Gray-Book design guidelines.** See `src/References/The-Gray-Book/reference/extending/design-guidelines.md` for full reference.

### Node Names

A node name consists of three components: **Name (Version) [Category]**

#### Name Component
- Use **CamelCasing**, no spaces
- **Process nodes** (stateful): use nouns → `Sequencer`, `Emitter`, `Integrator`
- **Operation nodes** (stateless): prefer verbs → `Add`, `Scale`, `Transform`
- Avoid `As..` prefixes like `AsString`. Use `To..` or `From..` instead

#### Version Component (optional)
- Most nodes should have **no version**
- Use versions only when specializing an existing node
- Simpler variant = no version, specialized variant = descriptive version
- Examples: `Emitter` vs `Emitter (Sphere)`, `Add` vs `Add (Saturated)`

#### Category Component (required)
- Use existing categories when possible: `[Math]`, `[Collections]`, `[Animation]`, `[Reactive]`
- For VL.Cuda nodes, use `[CUDA]` as root category
- Subcategories with dot notation: `[CUDA.Kernels]`, `[CUDA.Buffers]`, `[CUDA.Libraries]`
- Avoid excessive subcategory nesting (max 2-3 levels)

### Pin Names

- Use **spaces** to separate words, all starting with **upper case**
- Good: `Particle Count`, `Delta Time`, `Grid Size`
- Bad: `particleCount`, `delta_time`, `gridSize`
- Avoid generic names like `Do`, `Update` unless following established patterns
- For vector components: `X`, `Y`, `Z`, `W` (not `x`, `y`, `z`, `w`)

#### Pin Order
- **Main input** on the left
- **Configuration/parameters** in the middle
- **Reset/Control** typically on the right
- **Outputs** ordered by importance (main result first)

### Constructor Node Naming

Three naming patterns for constructor operations:

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Create** | Complex datatypes with functionality | `Create [Particle]` |
| **Join/Split** | Container datatypes (property bags) | `Vector2 (Join)`, `Vector2 (Split)` |
| **FromX/ToX** | Type conversions | `FromHSL [Color.RGBA]`, `ToRadians [Math]` |

For VL.Cuda:
```
Create [GpuBuffer<T>]           // Main buffer constructor
Upload [CUDA.Buffers]           // CPU → GPU conversion
Download [CUDA.Buffers]         // GPU → CPU conversion
FromCpuArray [CUDA.Buffers]     // Alternative to Upload
ToCpuArray [CUDA.Buffers]       // Alternative to Download
```

### XML Documentation

Use standard XML doc comments — VL reads them automatically:

```csharp
/// <summary>Emits particles from a sphere surface</summary>
/// <remarks>
/// Particles are distributed uniformly across the sphere surface.
/// Use Emission Rate to control density.
/// </remarks>
/// <param name="radius">Sphere radius in world units</param>
/// <param name="emissionRate">Particles per second</param>
/// <returns>Buffer containing newly emitted particles</returns>
public class SphereEmitter : ICudaBlock
{
    // ...
}
```

VL displays:
- **Summary** in NodeBrowser and as first line of tooltip
- **Remarks** in extended tooltip (multi-line, shows on hover)
- **Param** descriptions for each pin's tooltip
- **Returns** for output pin tooltips

> **IMPORTANT:** XML docs are only generated if the C# project has `<GenerateDocumentationFile>true</GenerateDocumentationFile>` in the .csproj.

### Tags for Node Search

- Tags are **lowercase, space-separated** search terms
- Apply via `[Tags("term1 term2 term3")]` attribute
- **Don't duplicate** terms already in name/version/category
- Example: `[Tags("gpu compute parallel kernel")]`

### Standard Datatypes

Prefer established VL types on node boundaries:

| Use Case | VL Type | Avoid |
|----------|---------|-------|
| Booleans | `bool` | Custom enum |
| Integers | `int` | `int32`, `Int32` |
| Floats | `float` | `float32`, `Float32` |
| Vectors | `Vector2`, `Vector3`, `Vector4` | Custom structs |
| Matrices | `Matrix4x4` | Custom matrix types |
| Strings | `string` | Custom text types |
| Collections | `Spread<T>`, `IEnumerable<T>` | Arrays, Lists |
| File paths | `Path` (VL.IO) | `string` |
| Colors | `RGBA` (VL.CoreLib) | Custom color types |
| Angles | **Cycles** (0-1 = full rotation) | Radians, degrees |

For VL.Cuda:
- Internal GPU buffers: `GpuBuffer<T>` (our type)
- CPU↔GPU boundary: `Spread<T>` (VL) ↔ `GpuBuffer<T>` (CUDA)
- Scalar parameters: use standard VL types (`float`, `int`, `Vector3`, etc.)

### Process Nodes

Mark classes intended as VL process nodes with `[ProcessNode]` attribute:

```csharp
[ProcessNode]
public class ParticleEmitter : ICudaBlock
{
    private int _particleCount;

    public ParticleEmitter(NodeContext nodeContext, CudaContext ctx)
    {
        // Constructor runs once on node creation
    }

    public void Update(float deltaTime, int emissionRate)
    {
        // Update runs every frame
        _particleCount += (int)(emissionRate * deltaTime);
    }

    public void Dispose()
    {
        // Cleanup when node is deleted
    }
}
```

Key points:
- `[ProcessNode]` only works if assembly has `[assembly:ImportAsIs]` attribute
- Constructor runs **once** when node is created
- `Update` (or any public method) runs **every frame** via VL pins
- Use fields for state that persists between frames
- VL automatically creates an input pin for any `Update` method parameter
- Methods with no outputs get a `bool` enable pin (default: false)
- Methods with Input/Output of same type get an `Apply` pin (default: true)

### Async Operations Pattern

For operations that may take time, follow this output pattern:

```csharp
public class AsyncLoader
{
    public void Load(out bool inProgress, out bool onCompleted,
                     out bool success, out string error)
    {
        // inProgress: true while operation is running
        // onCompleted: bang when operation finishes
        // success: true if completed successfully
        // error: error message if failed
    }
}
```

VL users recognize this pattern and know how to handle it.

### Exception Handling

For VL.Cuda blocks:
- **Validation errors** (missing inputs, type mismatches) → report via `IVLRuntime.AddMessage()` with `MessageSeverity.Error`
- **CUDA errors** (launch failures, out of memory) → report via `IVLRuntime.AddMessage()` with `MessageSeverity.Error`
- **Warnings** (deprecated features, suboptimal config) → report via `IVLRuntime.AddMessage()` with `MessageSeverity.Warning`
- **Info/stats** (timing, buffer usage) → return via `ToString()` for tooltips

Don't throw exceptions from `Update()` methods — VL can't recover. Always report via IVLRuntime.

### Pin Visibility

Operations with a single `Input` → `Output` flow automatically get an **Apply** pin:

```csharp
// Automatically gets Apply pin (default: true)
public GpuBuffer<float> Scale(GpuBuffer<float> input, float factor)
{
    // When Apply = false, input is passed through unchanged
    // When Apply = true, operation executes
}
```

Operations with no output get an **enable pin** named after the method:

```csharp
// Gets a "Clear" pin (default: false)
public void Clear()
{
    // Only runs when pin is true
}
```

### Observable Pattern

For events/notifications, always return `IObservable<T>`:

```csharp
public class BufferMonitor
{
    private Subject<int> _capacityExceeded = new();

    public IObservable<int> OnCapacityExceeded => _capacityExceeded;

    public void CheckCapacity(int current, int max)
    {
        if (current > max * 0.9f)
            _capacityExceeded.OnNext(current);
    }
}
```

VL has native Reactive nodes (`HoldLatest`, `KeepLatest`, etc.) that work seamlessly with `IObservable<T>`.

---

## Typical VL Patch Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌──────────────┐                                               │
│  │  CudaEngine  │──── Context ─────────────────────────┐        │
│  │  ⏱ 0.8ms     │                                      │        │
│  └──────────────┘                                      │        │
│                                                         │        │
│  ┌──────────────┐                                      │        │
│  │  FrameClock  │─── TimeDiff ──────────────────┐      │        │
│  └──────────────┘                               │      │        │
│                                                  │      │        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │ ┌──────────┐ │
│  │ Emitter  │───▶│ Forces   │───▶│Integrate │   │ │ Renderer │ │
│  │          │    │          │    │          │   │ │          │ │
│  │ Config ◀─│    │ Gravity◀─│    │ DeltaT ◀─┼───┘ │ Camera ◀─│ │
│  │ ctx   ◀──│────│ ctx   ◀──│────│ ctx   ◀──│─────│ ctx   ◀──│ │
│  └──────────┘    └──────────┘    └──────────┘     └──────────┘ │
│       ▲               ▲                                │        │
│       │               │                                ▼        │
│    External        Parameters                       Output      │
│    Buffer          from VL                         to Stride    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

External connections:
- **Context**: CudaEngine → all blocks (registration)
- **Config/Parameters**: VL values → CUDA scalars (including DeltaTime from FrameClock)
- **Buffers**: Can flow in/out of the graph
- **Renderer output**: Goes to VL.Stride for visualization (via ResourceProvider wrapping)

---

## Patching UX — ShaderFX Model for Compute

> This section describes how VL.Cuda **feels** to patch, from a VL user's perspective.

### Precedent: ShaderFX

VL already has a GPU patching paradigm: **ShaderFX** (in VL.StandardLibs). Users compose
shader operations visually — they never write HLSL. The same model applies to VL.Cuda
compute kernels.

Key principles inherited from ShaderFX:
- GPU operations are a **closed, finite node-set** (not arbitrary VL code)
- Pins carry **GPU types** — VL's type system prevents mixing CPU and GPU
- The user only sees **inputs and outputs** — no intermediate GPU state inspection
- The patch **describes** work — it doesn't execute it directly
- Parallelization is **implicit** — the user thinks per-element, not per-thread

### Uniform Block Experience

From the outside, **all four block variants look identical** — a block with typed pins:

```
From outside (all variants look the same):

┌──────────────┐
│  VectorScale  │
│               │
│  Data ◀───────│  GpuBuffer<float>
│  Scale ◀──────│  float
│               │
│  Result ──────│▶ GpuBuffer<float>
└──────────────┘
```

The user doesn't know or care whether the block contains:
- Filesystem PTX (pre-compiled)
- ILGPU IR (patchable compute)
- A library call chain (patchable captured)
- User CUDA C++ (NVRTC escape-hatch)

### Type System Enforcement

GPU/CPU separation is enforced by VL's type system — no special "GPU mode" needed:

```
GpuBuffer<float>  ←→  GpuBuffer<float>     ✅ connects
GpuBuffer<float>  ←→  Spread<float>        ❌ type mismatch, won't connect
GpuBuffer<float>  ←→  GpuBuffer<int>       ❌ type mismatch, won't connect
```

To move data between CPU and GPU, the user must use explicit conversion nodes:

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Spread   │────▶│ Upload   │────▶│ GPU.Mul  │────▶│ Download │──▶ Spread
│ <float>  │     │ (CPU→GPU)│     │          │     │ (GPU→CPU)│
└──────────┘     └──────────┘     └──────────┘     └──────────┘
```

This is intentional — GPU readback is expensive and should be a conscious decision.

### Patchable Compute Kernels (Element-Wise)

When the user opens a patchable compute block, they see GPU operation nodes.
The abstraction level is **element-wise** — one operation describes what happens
to a single element. Grid configuration and thread management are invisible.

```
Inside "ParticleForces" patch:

              ┌───────────┐     ┌───────────────┐
  Positions ──┤ GPU.Sub   ├────▶┤ GPU.Normalize ├──┐
  Center ─────┤           │     └───────────────┘  │
              └───────────┘                         │
                                                    │  ┌───────────┐
                                                    ├─▶┤ GPU.Mul   ├── Force
  Strength ─────────────────────────────────────────┘  │           │
                                                       └───────────┘
```

The user patches like they would for CPU math — but with GPU-typed nodes:

| GPU Node | Operation | Analogy |
|----------|-----------|---------|
| `GPU.Add` | Element-wise addition | Like `+` but for GPU buffers |
| `GPU.Mul` | Element-wise multiply | Like `*` but for GPU buffers |
| `GPU.Sub` | Element-wise subtract | Like `-` but for GPU buffers |
| `GPU.Normalize` | Normalize vector | Like `Normalize` but GPU |
| `GPU.Lerp` | Linear interpolation | Like `Lerp` but GPU |
| `GPU.Sin` / `GPU.Cos` | Trigonometry | Like `Sin`/`Cos` but GPU |
| `GPU.Clamp` | Clamp to range | Like `Clamp` but GPU |
| `GPU.Select` | Conditional select | Like `If` but GPU |
| `GPU.ReduceSum` | Sum all elements | Parallel reduction |
| `GPU.PrefixSum` | Running sum | Parallel scan |

Each node maps 1:1 to an ILGPU IR operation internally. The user never sees IR,
PTX, or CUDA concepts. Grid size is derived automatically from buffer dimensions.

### Patchable Library Call Chains

When the user opens a patchable captured block, they see library operation nodes.
Same principle — place nodes, connect pins:

```
Inside "MatMulFFT" patch:

          ┌──────────────┐     ┌──────────────┐
  A ──────┤ cuBLAS.Sgemm ├────▶┤ cuFFT.Forward├── Result
  B ──────┤              │     │              │
  M ──────┤              │     │   Size ◀─────│
  N ──────┤              │     └──────────────┘
  K ──────┤              │
          └──────────────┘
```

Library nodes are a separate closed set:

| Library Node | Operation |
|-------------|-----------|
| `cuBLAS.Sgemm` | Matrix multiply (float) |
| `cuBLAS.Dgemm` | Matrix multiply (double) |
| `cuBLAS.Saxpy` | Vector scale + add |
| `cuFFT.Forward` | FFT forward transform |
| `cuFFT.Inverse` | FFT inverse transform |

From outside, a library chain block looks identical to a compute block — same
pin types, same connection rules. The difference is purely internal (Stream
Capture vs ILGPU IR).

### What the User Cannot Do Inside GPU Patches

GPU patches are **not** general-purpose VL patches. The user cannot:

- Use CPU nodes (Spread operations, string manipulation, IO, etc.)
- Inspect intermediate values via tooltips (data is on GPU)
- Use VL delegates or stateful operations
- Create dynamic graph structures (no ForEach, no Reactive)

This is the same constraint as ShaderFX — and VL users already understand it.
The type system enforces it naturally: CPU-typed pins don't connect to GPU-typed pins.

### Three Levels of GPU Usage

| Level | User | What they do |
|-------|------|-------------|
| **Consumer** | Most VL users | Place pre-made blocks, connect pins, tweak parameters |
| **Composer** | Advanced users | Patch custom GPU logic using GPU nodes (element-wise) |
| **Author** | C#/CUDA developers | Write PTX (Triton/nvcc), load via filesystem, or write CUDA C++ via NVRTC |

Level 1 (Consumer) requires zero GPU knowledge. Level 2 (Composer) requires
understanding of element-wise parallel thinking (same as ShaderFX). Level 3
(Author) requires CUDA expertise but offers maximum control.
