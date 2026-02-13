# Block System

## Overview

The Block System provides a composable way to describe GPU compute pipelines. Blocks are **passive data-containers** that encapsulate kernels and their connections. They do not execute GPU work — that is the responsibility of the `CudaEngine` (see `EXECUTION-MODEL.md`).

```
Block Hierarchy:

SimpleBlock              CompositeBlock
┌──────────────────┐     ┌──────────────────────────────────────┐
│                  │     │                                      │
│  [Kernel1]       │     │  ┌─────────┐    ┌─────────┐         │
│  [Kernel2]       │     │  │ Child1  │───▶│ Child2  │         │
│                  │     │  └─────────┘    └─────────┘         │
│  Inputs/Outputs  │     │       ▲              │              │
│  exposed directly│     │       │              ▼              │
│                  │     │  Exposed Input   Exposed Output     │
└──────────────────┘     └──────────────────────────────────────┘

Blocks are descriptions — they never call CUDA APIs directly.
The CudaEngine reads these descriptions to compile and launch the graph.
```

---

## ICudaBlock Interface

Every block implements this interface:

```csharp
public interface ICudaBlock : IDisposable
{
    // Identity
    Guid Id { get; }
    string TypeName { get; }
    NodeContext NodeContext { get; }

    // Ports
    IReadOnlyList<IBlockPort> Inputs { get; }
    IReadOnlyList<IBlockPort> Outputs { get; }
    IReadOnlyList<IBlockParameter> Parameters { get; }

    // Debug (written by CudaEngine after each frame)
    IBlockDebugInfo DebugInfo { get; set; }
}
```

Note: There is no `Setup()` method in the interface. Setup happens in the **constructor**.

> **NodeContext**: VL's `NodeContext` is auto-injected as the first constructor parameter. It provides identity, logging, and app access. See `VL-INTEGRATION.md`.

---

## Block Lifecycle

```
Constructor(NodeContext nodeContext, CudaContext ctx):
    1. Store nodeContext (for identity/logging/app access)
    2. Create BlockBuilder(ctx, this)
    3. Define kernels, pins, connections (declarative)
    4. builder.Commit()   ← stores BlockDescription, wires param events
    5. ctx.RegisterBlock(this)  ← fires StructureChanged → DirtyTracker

VL Update() (every frame, optional):
    → Read DebugInfo for tooltip display
    → Push parameter value changes (triggers ValueChanged → OnParameterChanged)
    → NO GPU work

Dispose():
    → ctx.UnregisterBlock(this.Id)  ← removes from registry + connections + description
```

---

## BlockBuilder

The `BlockBuilder` is the DSL for describing blocks. It's used in the constructor and provides methods for:
- Adding kernels
- Defining inputs/outputs
- Connecting internal nodes
- Adding child blocks (composition)
- Exposing child ports

### Simple Block Example

```csharp
public class VectorAddBlock : ICudaBlock, IDisposable
{
    private readonly CudaContext _ctx;

    public Guid Id { get; } = Guid.NewGuid();
    public string TypeName => "VectorAdd";
    public NodeContext NodeContext { get; }

    public IReadOnlyList<IBlockPort> Inputs => _inputs;
    public IReadOnlyList<IBlockPort> Outputs => _outputs;
    public IReadOnlyList<IBlockParameter> Parameters => _parameters;
    public IBlockDebugInfo DebugInfo { get; set; }

    private readonly List<IBlockPort> _inputs = new();
    private readonly List<IBlockPort> _outputs = new();
    private readonly List<IBlockParameter> _parameters = new();

    public VectorAddBlock(NodeContext nodeContext, CudaContext ctx)
    {
        _ctx = ctx;
        NodeContext = nodeContext;

        var builder = new BlockBuilder(ctx, this);

        // Add the kernel (entry point extracted from JSON metadata)
        var kernel = builder.AddKernel("kernels/vector_add.ptx");

        // Define buffer ports bound to kernel parameters
        _inputs.Add(builder.Input<float>("A", kernel.In(0)));
        _inputs.Add(builder.Input<float>("B", kernel.In(1)));
        _outputs.Add(builder.Output<float>("C", kernel.Out(2)));

        // Scalar parameter — changes trigger Hot Update (no rebuild)
        _parameters.Add(builder.InputScalar<uint>("N", kernel.In(3), 1024u));

        builder.Commit();
        ctx.RegisterBlock(this);
    }

    public void Dispose() => _ctx.UnregisterBlock(Id);
}
```

### Composite Block Example *(Phase 3 — not yet implemented)*

```csharp
public class ParticleSystemBlock : ICudaBlock, IDisposable
{
    private readonly CudaContext _ctx;

    public Guid Id { get; } = Guid.NewGuid();
    public string TypeName => "ParticleSystem";
    public IBlockDebugInfo DebugInfo { get; set; }

    // ... interface implementation ...

    public ParticleSystemBlock(NodeContext nodeContext, CudaContext ctx)
    {
        _ctx = ctx;

        var builder = new BlockBuilder(ctx, this);

        // Add child blocks
        var emitter = builder.AddChild<SphereEmitterBlock>();
        var gravity = builder.AddChild<GravityForceBlock>();
        var drag = builder.AddChild<DragForceBlock>();
        var integrate = builder.AddChild<IntegrateBlock>();

        // Connect children internally
        builder.ConnectChildren(emitter, "Particles", gravity, "Particles");
        builder.ConnectChildren(gravity, "Particles", drag, "Particles");
        builder.ConnectChildren(drag, "Particles", integrate, "Particles");

        // Expose input from emitter
        var configIn = builder.ExposeInput("EmitterConfig", emitter, "Config");

        // Expose parameters from children
        builder.ExposeParameter<float>("Gravity", gravity, "Strength");
        builder.ExposeParameter<float>("DragCoeff", drag, "Coefficient");
        builder.ExposeParameter<float>("DeltaTime", integrate, "DeltaTime");

        // Expose output from integrate
        var particlesOut = builder.ExposeOutput("Particles", integrate, "Particles");

        _inputs.Add(configIn);
        _outputs.Add(particlesOut);

        builder.Commit();
        ctx.RegisterBlock(this);
    }

    public void Dispose() => _ctx.UnregisterBlock(this.Id);
}
```

---

## BlockBuilder Methods

### Kernel Operations (Implemented)

```csharp
// Add a kernel from filesystem PTX (entry point extracted from JSON metadata)
KernelHandle AddKernel(string ptxPath);
```

Set grid dimensions on the returned handle before `Commit()`:
```csharp
var kernel = builder.AddKernel("kernels/my_kernel.ptx");
kernel.GridDimX = 256;  // default is 1
```

### Input Definitions (Implemented)

```csharp
// Buffer input port bound to a kernel parameter. Returns BlockPort.
BlockPort Input<T>(string name, KernelPin pin) where T : unmanaged;

// Scalar input parameter. Returns BlockParameter<T> with change tracking.
BlockParameter<T> InputScalar<T>(string name, KernelPin pin, T defaultValue = default) where T : unmanaged;
```

### Output Definitions (Implemented)

```csharp
// Buffer output port bound to a kernel parameter. Returns BlockPort.
BlockPort Output<T>(string name, KernelPin pin) where T : unmanaged;

// Append output port — wraps an AppendBuffer<T> (data + atomic counter).
// Automatically creates a companion "{name} Count" output port for the current count.
// The counter is reset to 0 by a memset node before each graph launch (see GRAPH-COMPILER.md).
AppendOutputPort AppendOutput<T>(string name, KernelPin dataPin, KernelPin counterPin, int maxCapacity)
    where T : unmanaged;
```

### Internal Connections (Implemented)

```csharp
// Connect two kernel parameters within this block (via KernelPin references)
void Connect(KernelPin source, KernelPin target);
```

### Captured Operations (Implemented — Phase 4a)

```csharp
// Add a captured library operation. Returns a handle for binding ports.
// The captureAction receives the stream and a flat buffer bindings array.
CapturedHandle AddCaptured(Action<CUstream, CUdeviceptr[]> captureAction, CapturedNodeDescriptor descriptor);

// Input/Output overloads accept CapturedPin for binding to captured operation parameters.
BlockPort Input<T>(string name, CapturedPin pin) where T : unmanaged;
BlockPort Output<T>(string name, CapturedPin pin) where T : unmanaged;
```

**CapturedHandle** provides `In()`, `Out()`, and `Scalar()` methods that return `CapturedPin` values pointing to flat buffer binding indices. Buffer bindings follow descriptor order: `[inputs..., outputs..., scalars...]`.

```csharp
var descriptor = new CapturedNodeDescriptor("cuBLAS.Sgemm",
    inputs: new[] { CapturedParam.Pointer("A", "float*"), CapturedParam.Pointer("B", "float*") },
    outputs: new[] { CapturedParam.Pointer("C", "float*") });

var op = builder.AddCaptured((stream, buffers) =>
{
    var blas = libs.GetOrCreateBlas();
    blas.Stream = stream;
    CudaBlasNativeMethods.cublasSgemm_v2(blas.CublasHandle, ...
        buffers[0], ..., buffers[1], ..., buffers[2], ...);
}, descriptor);

builder.Input<float>("A", op.In(0));    // flat index 0
builder.Input<float>("B", op.In(1));    // flat index 1
builder.Output<float>("C", op.Out(0));  // flat index 2 (inputs.Count + 0)
```

On `Commit()`, each `CapturedHandle` produces a `CapturedEntry` stored in `BlockDescription.CapturedEntries`. This contains the handle ID, descriptor, and capture action needed by `CudaEngine` to create `CapturedNode` instances during `ColdRebuild`.

See `KERNEL-SOURCES.md` for the full three-source architecture and library operation wrappers.

### Planned (Phase 3+ / Phase 4b)

The following BlockBuilder methods are designed but not yet implemented:

```csharp
// Phase 3: Composite blocks
T AddChild<T>() where T : ICudaBlock;
void ConnectChildren(ICudaBlock source, string sourcePort, ICudaBlock target, string targetPort);
InputHandle<T> ExposeInput<T>(string name, InputHandle<T> childInput);
OutputHandle<T> ExposeOutput<T>(string name, OutputHandle<T> childOutput);
BlockParameter<T> ExposeParameter<T>(string name, ICudaBlock child, string childParamName);

// Phase 4b: Patchable Kernels (ILGPU IR + NVRTC)
KernelHandle AddKernel(CUmodule ilgpuModule, string entryPoint);
KernelHandle AddKernel(CUmodule nvrtcModule, string entryPoint, bool isNvrtc = true);
```

---

## KernelHandle

Represents a kernel added to a block via `BlockBuilder.AddKernel()`:

```csharp
public sealed class KernelHandle
{
    public Guid Id { get; }
    public string PtxPath { get; }
    public KernelDescriptor Descriptor { get; }

    // Grid dimensions (default 1×1×1). Set before Commit().
    public uint GridDimX { get; set; }
    public uint GridDimY { get; set; }
    public uint GridDimZ { get; set; }

    // Access kernel pins by parameter index
    public KernelPin In(int index);   // → KernelPin(this.Id, index)
    public KernelPin Out(int index);  // → KernelPin(this.Id, index)
}
```

> **Note**: `In()` and `Out()` return the same `KernelPin` type — the semantic difference (input vs output) is captured by how the pin is used in `builder.Input<T>()` vs `builder.Output<T>()`.

### Grid Configuration (Phase 2)

Phase 2 uses simple `uint GridDimX/Y/Z` on KernelHandle. These are stored in `KernelEntry` within `BlockDescription` and applied to `KernelNode` during `ColdRebuild`.

**Planned (Phase 3+):** `GridConfig` class with `GridSizeMode.Auto` (calculate from buffer size) and `GridSizeMode.Fixed`.

---

## CapturedHandle

Represents a captured library operation added to a block via `BlockBuilder.AddCaptured()`:

```csharp
public sealed class CapturedHandle
{
    public Guid Id { get; }
    public CapturedNodeDescriptor Descriptor { get; }
    public Action<CUstream, CUdeviceptr[]> CaptureAction { get; }

    // Access captured params by category index → returns flat buffer binding index
    public CapturedPin In(int index);      // → flat index = index
    public CapturedPin Out(int index);     // → flat index = Inputs.Count + index
    public CapturedPin Scalar(int index);  // → flat index = Inputs.Count + Outputs.Count + index
}
```

`CapturedPin` is analogous to `KernelPin` but references a flat index into the buffer bindings array rather than a kernel parameter index. The bindings array layout is `[inputs..., outputs..., scalars...]` matching the descriptor order.

```csharp
public readonly struct CapturedPin
{
    public Guid CapturedHandleId { get; }
    public int ParamIndex { get; }            // Flat index into BufferBindings
    public CapturedPinCategory Category { get; }  // Input, Output, or Scalar
}
```

### CapturedEntry

On `Commit()`, each `CapturedHandle` produces a `CapturedEntry` stored in `BlockDescription.CapturedEntries`:

```csharp
public sealed class CapturedEntry
{
    public Guid HandleId { get; }
    public CapturedNodeDescriptor Descriptor { get; }
    public Action<CUstream, CUdeviceptr[]> CaptureAction { get; }

    // StructuralEquals ignores HandleId, compares debug name + param counts
    public bool StructuralEquals(CapturedEntry? other);
}
```

---

## Block Ports

### IBlockPort

```csharp
public interface IBlockPort
{
    Guid BlockId { get; }
    string Name { get; }
    PortDirection Direction { get; }
    PinType Type { get; }
}
```

### BlockPort

Non-generic concrete port — maps a named port to a specific kernel node's parameter.

```csharp
public sealed class BlockPort : IBlockPort
{
    public Guid BlockId { get; }
    public string Name { get; }
    public PortDirection Direction { get; }
    public PinType Type { get; }

    // Internal: which kernel node/param this maps to (set by BlockBuilder)
    internal Guid KernelNodeId { get; set; }
    internal int KernelParamIndex { get; set; }
}
```

### AppendOutputPort

Returned by `BlockBuilder.AppendOutput<T>()`. Represents a streaming append output with an auto-generated count companion port:

```csharp
public sealed class AppendOutputPort : IBlockPort
{
    public Guid BlockId { get; }
    public string Name { get; }                    // e.g. "Particles"
    public PortDirection Direction => PortDirection.Output;
    public PinType Type { get; }                   // Buffer<T>

    public string CountPortName { get; }           // Auto-generated: "Particles Count"
    public int MaxCapacity { get; }

    // Internal: kernel parameter mappings
    internal Guid KernelNodeId { get; set; }
    internal int DataParamIndex { get; set; }      // Kernel param for data buffer
    internal int CounterParamIndex { get; set; }   // Kernel param for counter buffer
}
```

In VL, an append output appears as two pins:
```
┌───────────────────────┐
│  ParticleEmitter      │
│                       │
│       Particles ○──── │  ← AppendBuffer<Particle>.Data pointer
│  Particles Count ○─── │  ← uint, auto-readback after launch
│                       │
└───────────────────────┘
```

### AppendBufferInfo

Stored in `BlockDescription` to track which outputs are append buffers. Used by `GraphCompiler` to generate memset nodes and by `CudaEngine` for auto-readback:

```csharp
public sealed class AppendBufferInfo
{
    public string PortName { get; }             // "Particles"
    public IAppendBuffer Buffer { get; }        // Type-erased append buffer
    public Guid KernelNodeId { get; }           // Which kernel writes to this
    public int DataParamIndex { get; }
    public int CounterParamIndex { get; }
    public int MaxCapacity { get; }
}
```

---

## Block Parameters

For scalar values that can be changed at runtime without graph rebuild (Hot Update):

```csharp
public interface IBlockParameter
{
    string Name { get; }
    Type ValueType { get; }
    object Value { get; set; }
    bool IsDirty { get; }
    void ClearDirty();
}

public sealed class BlockParameter<T> : IBlockParameter where T : unmanaged
{
    public string Name { get; }
    public Type ValueType => typeof(T);
    public T TypedValue { get; set; }        // Fires ValueChanged on change
    public event Action<BlockParameter<T>>? ValueChanged;
    public bool IsDirty { get; }
    public void ClearDirty();

    internal Guid KernelNodeId { get; set; }     // mapped kernel
    internal int KernelParamIndex { get; set; }  // mapped param index
}
```

When `TypedValue` is set, the `ValueChanged` event fires → `BlockBuilder.Commit()` wired it to call `CudaContext.OnParameterChanged(blockId, paramName)` → `DirtyTracker.MarkParameterDirty()`. CudaEngine picks this up in its next `Update()` and applies a Hot Update — no graph rebuild needed.

Usage in VL:
```
┌─────────────────────┐
│  GravityForce       │
│                     │
│  Strength: [9.81]   │  ← BlockParameter<float>, Hot Update
│  Direction: [▼ Y]   │  ← BlockParameter<Vector3>, Hot Update
│                     │
└─────────────────────┘
```

---

## Block Debug Info

Written by CudaEngine after each frame launch:

**Current implementation (Phase 2 + Phase 3.1):**

```csharp
public interface IBlockDebugInfo
{
    BlockState State { get; }
    string? StateMessage { get; }
    TimeSpan LastExecutionTime { get; }
    IReadOnlyDictionary<string, uint> AppendCounts { get; }  // Phase 3.1
}

public sealed class BlockDebugInfo : IBlockDebugInfo
{
    public BlockState State { get; set; } = BlockState.NotCompiled;
    public string? StateMessage { get; set; }
    public TimeSpan LastExecutionTime { get; set; }
    public Dictionary<string, uint> AppendCounts { get; set; } = new();  // Phase 3.1
}

public enum BlockState
{
    OK,          // Running normally
    Warning,     // Something noteworthy
    Error,       // Compilation/runtime error
    NotCompiled  // Graph not yet compiled
}
```

`AppendCounts` is populated by `CudaEngine` after each graph launch. Each entry maps an append output port name to the counter value read back from GPU. For example, `AppendCounts["Particles"]` = 4200 means 4200 particles were emitted this frame. Blocks read this in their VL `Update()` for tooltip display.

**Planned additions (Phase 3+):** `AverageExecutionTime`, `Buffers`, `Kernels`, `Children` lists for hierarchical profiling.

---

## Composition Patterns *(Phase 3 — not yet implemented)*

### Sequential Pipeline

```csharp
public ParticleChainBlock(NodeContext nodeContext, CudaContext ctx)
{
    var builder = new BlockBuilder(ctx, this);

    var step1 = builder.AddChild<Step1Block>();
    var step2 = builder.AddChild<Step2Block>();
    var step3 = builder.AddChild<Step3Block>();

    builder.ConnectChildren(step1, "Out", step2, "In");
    builder.ConnectChildren(step2, "Out", step3, "In");

    builder.ExposeInput("In", step1, "In");
    builder.ExposeOutput("Out", step3, "Out");

    builder.Commit();
    ctx.RegisterBlock(this);
}
```

### Fan-Out / Fan-In

```csharp
public FanOutInBlock(NodeContext nodeContext, CudaContext ctx)
{
    var builder = new BlockBuilder(ctx, this);

    var split = builder.AddChild<SplitBlock>();
    var processA = builder.AddChild<ProcessABlock>();
    var processB = builder.AddChild<ProcessBBlock>();
    var merge = builder.AddChild<MergeBlock>();

    // Fan-out
    builder.ConnectChildren(split, "OutA", processA, "In");
    builder.ConnectChildren(split, "OutB", processB, "In");

    // Fan-in
    builder.ConnectChildren(processA, "Out", merge, "InA");
    builder.ConnectChildren(processB, "Out", merge, "InB");

    builder.ExposeInput("In", split, "In");
    builder.ExposeOutput("Out", merge, "Out");

    builder.Commit();
    ctx.RegisterBlock(this);
}
```

### Nested Composition

Blocks can contain blocks that contain blocks:

```csharp
public class Level1Block : ICudaBlock
{
    public Level1Block(NodeContext nodeContext, CudaContext ctx)
    {
        var builder = new BlockBuilder(ctx, this);
        var level2 = builder.AddChild<Level2Block>();  // Level2 contains Level3
        // ...
        builder.Commit();
        ctx.RegisterBlock(this);
    }
}
```

---

## Context Threading & DAG Branching *(Architecture — see VL-UX.md)*

Inside the CudaGraph Region, every block node follows the **context threading pattern**:

- **First input pin**: context (PinGroup — execution ordering)
- **First output pin**: context (single — signals completion)
- **Remaining pins**: buffer I/O and scalar parameters

### PinGroup on Context Input

The context input is a `PinGroup` (`[Pin(PinGroupKind = PinGroupKind.Collection)]`). Users add/remove dependency pins with Ctrl+/Ctrl-. Default is one pin.

```csharp
[ProcessNode]
public class CudaKernelNode
{
    [Pin(PinGroupKind = PinGroupKind.Collection)]
    public IEnumerable<CudaContext> ContextInputs { get; }  // Multiple ctx-in

    public CudaContext ContextOutput { get; }                // Single ctx-out
    // ... buffer pins, scalar pins
}
```

### Implicit Buffer Dependencies

Buffer links create implicit execution dependencies. If KernelB reads KernelA's output `GpuBuffer<T>`, B automatically depends on A. The ctx PinGroup is only needed for non-data ordering (e.g., two independent memsets before a kernel).

### Graph Compiler Resolution

The GraphCompiler reads both ctx links and buffer links to build CUDA Graph dependency edges. See `VL-UX.md` for the full DAG branching design.

---

## Best Practices

1. **Keep blocks focused**: One block = one logical operation
2. **Use composition**: Build complex from simple
3. **Expose meaningful ports**: Hide internal complexity
4. **Document parameters**: Use description strings
5. **Consider reusability**: Design for multiple use cases
6. **Match VL patterns**: Fit with visual programming style
7. **Constructor = Setup**: All description happens in the constructor
8. **Always Dispose**: Unregister from CudaContext to trigger rebuild
9. **Context threading**: First pin in = ctx, first pin out = ctx (inside CudaGraph Region)
10. **Buffer links = implicit deps**: Don't require redundant ctx links for data-connected nodes
