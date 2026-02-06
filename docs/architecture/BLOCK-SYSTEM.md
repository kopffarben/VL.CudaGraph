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
```

### Internal Connections (Implemented)

```csharp
// Connect two kernel parameters within this block (via KernelPin references)
void Connect(KernelPin source, KernelPin target);
```

### Planned (Phase 3+)

The following BlockBuilder methods are designed but not yet implemented:

```csharp
// Phase 3: Composite blocks
T AddChild<T>() where T : ICudaBlock;
void ConnectChildren(ICudaBlock source, string sourcePort, ICudaBlock target, string targetPort);
InputHandle<T> ExposeInput<T>(string name, InputHandle<T> childInput);
OutputHandle<T> ExposeOutput<T>(string name, OutputHandle<T> childOutput);
BlockParameter<T> ExposeParameter<T>(string name, ICudaBlock child, string childParamName);

// Phase 4a: Library Calls via Stream Capture
CapturedHandle AddCaptured(string name, Action<CUstream> captureAction, CapturedOpDescriptor descriptor);

// Phase 4b: Patchable Kernels (ILGPU IR + NVRTC)
KernelHandle AddKernel(CUmodule ilgpuModule, string entryPoint);
KernelHandle AddKernel(CUmodule nvrtcModule, string entryPoint, bool isNvrtc = true);
```

See `KERNEL-SOURCES.md` for the full three-source architecture.

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

**Current implementation (Phase 2):**

```csharp
public interface IBlockDebugInfo
{
    BlockState State { get; }
    string? StateMessage { get; }
    TimeSpan LastExecutionTime { get; }
}

public sealed class BlockDebugInfo : IBlockDebugInfo
{
    public BlockState State { get; set; } = BlockState.NotCompiled;
    public string? StateMessage { get; set; }
    public TimeSpan LastExecutionTime { get; set; }
}

public enum BlockState
{
    OK,          // Running normally
    Warning,     // Something noteworthy
    Error,       // Compilation/runtime error
    NotCompiled  // Graph not yet compiled
}
```

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

## Best Practices

1. **Keep blocks focused**: One block = one logical operation
2. **Use composition**: Build complex from simple
3. **Expose meaningful ports**: Hide internal complexity
4. **Document parameters**: Use description strings
5. **Consider reusability**: Design for multiple use cases
6. **Match VL patterns**: Fit with visual programming style
7. **Constructor = Setup**: All description happens in the constructor
8. **Always Dispose**: Unregister from CudaContext to trigger rebuild
