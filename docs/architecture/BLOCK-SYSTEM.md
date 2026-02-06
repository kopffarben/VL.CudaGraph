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

Note: There is no `Setup()` method in the interface. Setup happens in the **constructor**, which receives `NodeContext` (injected by VL) and `CudaContext` as parameters.

> **NodeContext Convention**: VL injects `NodeContext` as the first constructor parameter automatically — VL users never see it. C# users see it in the signature. See `VL-INTEGRATION.md`.

---

## Block Lifecycle

```
Constructor(NodeContext nodeContext, CudaContext ctx):
    1. Store nodeContext (for identity, logging)
    2. Create BlockBuilder
    3. Define kernels, pins, connections (declarative)
    4. builder.Commit()
    5. ctx.Registry.Register(this, nodeContext)

VL Update() (every frame, optional):
    → Read DebugInfo for tooltip display
    → Push parameter value changes to CudaContext
    → NO GPU work

Dispose():
    → ctx.Registry.Unregister(this)
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

    public IReadOnlyList<IBlockPort> Inputs => _inputs;
    public IReadOnlyList<IBlockPort> Outputs => _outputs;
    public IReadOnlyList<IBlockParameter> Parameters => _parameters;
    public IBlockDebugInfo DebugInfo { get; set; }

    private List<IBlockPort> _inputs = new();
    private List<IBlockPort> _outputs = new();
    private List<IBlockParameter> _parameters = new();

    public VectorAddBlock(NodeContext nodeContext, CudaContext ctx)
    {
        _ctx = ctx;

        var builder = new BlockBuilder(ctx, this);

        // Add the kernel
        var kernel = builder.AddKernel("kernels/vector_add.ptx", "vector_add_f32");

        // Define inputs (buffer pins)
        var inA = builder.Input<float>("A", kernel.In(0), "First vector");
        var inB = builder.Input<float>("B", kernel.In(1), "Second vector");

        // Define output
        var outSum = builder.Output<float>("Sum", kernel.Out(2), "Result vector");

        // Scalar parameter (becomes a config pin)
        builder.InputScalar<int>("Count", kernel.In(3), defaultValue: 0);

        // Store for interface
        _inputs.AddRange(new[] { inA, inB });
        _outputs.Add(outSum);

        builder.Commit();
        ctx.RegisterBlock(this);
    }

    public void Dispose() => _ctx.UnregisterBlock(this.Id);
}
```

### Composite Block Example

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

### Kernel Operations

```csharp
// Source 1: Add a kernel from filesystem PTX
KernelHandle AddKernel(string ptxPath, string entryPoint, string? debugName = null);

// Source 1: Add with explicit grid config
KernelHandle AddKernel(string ptxPath, string entryPoint, GridConfig grid, string? debugName = null);

// Source 2: Add a kernel from NVRTC-compiled module (patchable kernel)
KernelHandle AddKernel(CUmodule nvrtcModule, string entryPoint, string? debugName = null);
```

### Captured Operations (Library Calls)

```csharp
// Source 3: Add a library operation via Stream Capture (cuBLAS, cuFFT, cuDNN)
CapturedHandle AddCaptured(string name, Action<CUstream> captureAction,
                           CapturedOpDescriptor descriptor);
```

CapturedNodes define their I/O via the `CapturedOpDescriptor` rather than `KernelPin` references. The BlockBuilder maps descriptor entries to standard BlockPorts:

```csharp
var op = builder.AddCaptured("MatMul", captureAction, new CapturedOpDescriptor
{
    Inputs  = new[] { ("A", typeof(float)), ("B", typeof(float)) },
    Outputs = new[] { ("C", typeof(float)) },
    Scalars = new[] { ("M", typeof(int)), ("N", typeof(int)) }
});

// Descriptor entries become standard BlockPorts:
// op.In("A")  → InputHandle<GpuBuffer<float>>  (same as KernelPin-based input)
// op.Out("C") → OutputHandle<GpuBuffer<float>> (same as KernelPin-based output)
// Scalars become BlockParameter<T> (trigger Recapture instead of Hot Update)
```

From the outside, CapturedNode ports are indistinguishable from KernelNode ports — the block system stays uniform. The difference is internal: parameter changes on CapturedNodes trigger Recapture instead of Hot Updates/Warm Updates.

See `KERNEL-SOURCES.md` for the full three-source architecture.

### Input Definitions

```csharp
// Buffer input
InputHandle<GpuBuffer<T>> Input<T>(string name, KernelPin pin, string? description = null);

// AppendBuffer input
InputHandle<AppendBuffer<T>> InputAppend<T>(string name, KernelPin pin, string? description = null);

// Scalar input (becomes parameter)
void InputScalar<T>(string name, KernelPin pin, T defaultValue = default, string? description = null);
```

### Output Definitions

```csharp
// Buffer output
OutputHandle<GpuBuffer<T>> Output<T>(string name, KernelPin pin, string? description = null);

// AppendBuffer output
OutputHandle<AppendBuffer<T>> OutputAppend<T>(string name, KernelPin pin, string? description = null);
```

### Internal Connections

```csharp
// Connect output to input within block
void Connect<T>(OutputHandle<T> source, InputHandle<T> target);
```

### Child Block Operations

```csharp
// Add child block (child receives same CudaContext)
T AddChild<T>() where T : ICudaBlock;
ICudaBlock AddChild(Type blockType);

// Connect between children
void ConnectChildren<T>(OutputHandle<T> source, InputHandle<T> target);
void ConnectChildren(ICudaBlock source, string sourcePort, ICudaBlock target, string targetPort);
```

### Exposing Ports

```csharp
// Expose child input as block input
InputHandle<T> ExposeInput<T>(string name, InputHandle<T> childInput);
InputHandle<T> ExposeInput<T>(InputHandle<T> childInput);  // Keep original name

// Expose child output as block output
OutputHandle<T> ExposeOutput<T>(string name, OutputHandle<T> childOutput);
OutputHandle<T> ExposeOutput<T>(OutputHandle<T> childOutput);  // Keep original name

// Expose child parameter
BlockParameter<T> ExposeParameter<T>(string name, ICudaBlock child, string childParamName);
```

---

## KernelHandle

Represents a kernel added to a block:

```csharp
public class KernelHandle
{
    public Guid Id { get; }
    public string PTXPath { get; }
    public string EntryPoint { get; }
    public string DebugName { get; }
    
    // Grid configuration
    public GridConfig Grid { get; set; }
    
    // Access kernel pins
    public KernelPin In(int index);   // Input parameter
    public KernelPin Out(int index);  // Output parameter
    
    // Metadata
    public KernelDescriptor Descriptor { get; }
}
```

### Grid Configuration

```csharp
public class GridConfig
{
    // Thread block dimensions
    public int[] BlockDim { get; set; } = new[] { 256 };
    
    // How to determine grid size
    public GridSizeMode Mode { get; set; } = GridSizeMode.Auto;
    
    // For Fixed mode
    public int[]? FixedGridDim { get; set; }
    
    // For Auto mode: which parameter determines size
    public int? AutoSizeFromParam { get; set; }
}

public enum GridSizeMode
{
    Fixed,  // Use FixedGridDim
    Auto    // Calculate from buffer size
}
```

---

## Block Ports

### IBlockPort

```csharp
public interface IBlockPort
{
    string Name { get; }
    PinType Type { get; }
    string? Description { get; }
}
```

### BlockPort<T>

```csharp
public class BlockPort<T> : IBlockPort
{
    public string Name { get; }
    public PinType Type { get; }
    public string? Description { get; }
    
    // Internal: which node/pin this maps to
    internal Guid NodeId { get; }
    internal int PinIndex { get; }
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
    object DefaultValue { get; }
    string? Description { get; }
}

public class BlockParameter<T> : IBlockParameter
{
    public string Name { get; }
    public T Value { get; set; }
    public T DefaultValue { get; }
    public string? Description { get; }
}
```

When a parameter value changes, the block notifies the CudaContext (parametersDirty = true). The CudaEngine picks this up in its next Update() and applies a Hot Update — no graph rebuild needed.

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

```csharp
public interface IBlockDebugInfo
{
    // Timing
    TimeSpan LastExecutionTime { get; }
    TimeSpan AverageExecutionTime { get; }
    
    // Resources
    IReadOnlyList<BufferInfo> Buffers { get; }
    IReadOnlyList<KernelDebugInfo> Kernels { get; }
    
    // State
    BlockState State { get; }
    string? StateMessage { get; }
    
    // Hierarchy
    IReadOnlyList<IBlockDebugInfo> Children { get; }
}

public enum BlockState
{
    OK,          // Running normally
    Warning,     // Something noteworthy
    Error,       // Compilation/runtime error
    NotCompiled  // Graph not yet compiled
}
```

---

## Composition Patterns

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
