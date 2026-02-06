# VL Integration

## Execution Model

> **See `EXECUTION-MODEL.md` for the full execution model.**

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
    
    [PinGroup("Inputs", PinGroupKind.Dynamic)]
    public IEnumerable<IBlockPort> Inputs => _inputs;
    
    [PinGroup("Outputs", PinGroupKind.Dynamic)]
    public IEnumerable<IBlockPort> Outputs => _outputs;
    
    private List<BlockPort> _inputs;
    private List<BlockPort> _outputs;
    
    public KernelBlock(CudaContext ctx, string ptxPath, string entryPoint)
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
    
    public void Dispose() => _ctx.UnregisterBlock(this);
}
```

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
var emitter = new SphereEmitterBlock(ctx);
var forces = new ForcesBlock(ctx);
var renderer = new ParticleRendererBlock(ctx);

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
    public ComplexBlock(CudaContext ctx)
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

VL shows debug information as tooltips on nodes and pins. The CudaEngine distributes debug info to blocks after each frame:

```
Node Tooltip:
┌─────────────────────────────────────────┐
│  ParticleEmitter                        │
│                                         │
│  ⏱ 0.42 ms (avg: 0.38 ms)              │
│  📦 Particles: 45,231 / 100,000        │
│  ✅ OK                                  │
└─────────────────────────────────────────┘

Pin Tooltip:
┌─────────────────────────────────────────┐
│  Particles (AppendBuffer<float4>)       │
│  Count: 45,231 / 100,000               │
│  Size: 724 KB                           │
└─────────────────────────────────────────┘

Error Tooltip:
┌─────────────────────────────────────────┐
│  Forces                                 │
│  ❌ Type mismatch on input              │
│     Expected: GpuBuffer<float3>         │
│     Got: GpuBuffer<float4>              │
└─────────────────────────────────────────┘
```

Access through:
```csharp
block.DebugInfo.LastExecutionTime;
block.DebugInfo.Buffers;
block.DebugInfo.State;
```

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
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Emitter  │───▶│ Forces   │───▶│Integrate │───▶│ Renderer │  │
│  │          │    │          │    │          │    │          │  │
│  │ Config ◀─│    │ Gravity◀─│    │ DeltaT ◀─│    │ Camera ◀─│  │
│  │ ctx   ◀──│────│ ctx   ◀──│────│ ctx   ◀──│────│ ctx   ◀──│  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       ▲               ▲               ▲               │         │
│       │               │               │               ▼         │
│    External        Parameters       Time           Output       │
│    Buffer          from VL          from VL        to Stride    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

External connections:
- **Context**: CudaEngine → all blocks (registration)
- **Config/Parameters**: VL values → CUDA scalars
- **Buffers**: Can flow in/out of the graph
- **Renderer output**: Goes to VL.Stride for visualization
