# VL UX Architecture

> **Status**: Architecture design complete. Not yet implemented.
> See `VL-INTEGRATION.md` for C# integration patterns, `BLOCK-SYSTEM.md` for BlockBuilder DSL.

## Overview

This document describes how VL.CudaGraph **looks and feels** in the vvvv gamma visual programming environment. It covers the Region-based design, CudaFunction reusability, data flow patterns, and the three user levels (Consumer, Composer, Author).

The key insight: **CudaGraph is a VL Region**, not a standalone node. GPU work is described inside the region, compiled into a CUDA Graph, and executed by the CudaEngine at the region boundary.

---

## CudaGraph Region

The CudaGraph Region is the primary container for GPU work. It follows the VL Region pattern (like VL.ImGui) with context threading.

### Structure

```
┌─ CudaGraph Region ──────────────────────────────────────┐
│                                                          │
│  ctx in ──▶ [Upload] ──▶ [Kernel] ──▶ [Download] ──▶ ctx out
│                                                          │
│  Spread<float> ──▶ (BCP) ──▶ GpuBuffer<float>          │
│                              ──▶ ... ──▶                │
│  GpuBuffer<float> ──▶ (BCP) ──▶ Spread<float>          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Context Threading

Every node inside the CudaGraph Region has:
- **First input**: context (ctx in) — controls execution order
- **First output**: context (ctx out) — signals completion

This is the same pattern as VL.ImGui. The context carries no data — it is purely an execution ordering mechanism. VL evaluates nodes top-to-bottom following context links.

```
┌──────────────┐
│  KernelNode  │
│              │
│  ctx in  ──▶ │──▶ ctx out       ← execution ordering
│  Data    ──▶ │──▶ Data out      ← GPU buffer data
│  Scale   ──▶ │                  ← scalar parameter
└──────────────┘
```

### Border Control Points (BCP)

Data enters and exits the Region through BCPs:

| Direction | CPU Side | GPU Side | CUDA Graph Node |
|-----------|----------|----------|-----------------|
| In (Upload) | `Spread<T>` | `GpuBuffer<T>` | Memcpy H→D |
| Out (Download) | `Spread<T>` | `GpuBuffer<T>` | Memcpy D→H |
| Scalar In | `float`, `int`, etc. | Kernel parameter | Hot Update (no graph node) |

**Upload** and **Download** are explicit nodes inside the Region — they compile to Memcpy nodes in the CUDA Graph. The user places them intentionally because GPU↔CPU transfers are expensive.

Scalar parameters (float, int, Vector3, etc.) flow directly to kernel pins without Upload nodes. They are set via `cuGraphExecKernelNodeSetParams` (Hot Update).

---

## DAG Branching — PinGroup on Context Input

### The Problem

VL inputs accept exactly **one link**. But a CUDA node may depend on multiple predecessors.

### The Solution

The context input is a **PinGroup** — one pin per dependency, user adds/removes with Ctrl+/Ctrl-.

```
┌─────────────────────────┐
│  KernelNode "C"         │
│                         │
│  Ctx In  [PinGroup]     │  ← Multiple ctx inputs
│    ├─ Ctx 1  ←── from MemsetA ctx out
│    └─ Ctx 2  ←── from MemsetB ctx out
│                         │
│  Buffer In: data        │  ← Normal data pins (single link)
│                         │
│  Ctx Out ───────→       │  ← Single ctx out
│  Buffer Out: result     │
└─────────────────────────┘
```

### Dependency Rules

1. **Buffer links create implicit dependencies.** If KernelB reads KernelA's output buffer, B automatically depends on A. No additional ctx link needed.
2. **Ctx PinGroup is for non-data dependencies only.** Use it when ordering matters but no buffer is shared (e.g., two independent memsets before a kernel).
3. **Fan-out is free.** One ctx-out can connect to multiple downstream ctx-in pins (VL outputs support multiple links).
4. **Default: one ctx-in pin.** The PinGroup starts with one pin. User adds more only when needed.

### Graph Compiler Behavior

```
For each node in the CudaGraph Region:
  For each ctx-in link:
    → Create CUDA Graph dependency edge (predecessor → this)
  For each buffer-in link:
    → Create CUDA Graph dependency edge (source node → this)
    (implicit — no ctx link needed)

  Zero ctx-in links + zero buffer-in links:
    → Root node (depends only on Region entry)
```

---

## CudaFunction — Reusable Sub-Graphs

A CudaFunction is a reusable piece of GPU logic, defined **outside** the CudaGraph Region and invoked **inside** via an Invoke node.

### Definition (Outside CudaGraph Region)

```
┌─ CudaFunction "MatMul" ──────────────────────────┐
│  Mode: Inline | SubGraph                          │
│                                                    │
│  ctx in ──▶ [cuBLAS GEMM] ──▶ ctx out            │
│  A     ──▶                ──▶ C                   │
│  B     ──▶                                        │
└────────────────────────────────────────────────────┘
```

### Invocation (Inside CudaGraph Region)

```
┌─ CudaGraph Region ─────────────────────────┐
│                                              │
│  ctx ──▶ [Upload] ──▶ [Invoke MatMul] ──▶  │
│                        A ──▶  ──▶ C         │
│                        B ──▶                │
└──────────────────────────────────────────────┘
```

### Inline vs SubGraph Mode

| Mode | CUDA Graph Result | Use Case |
|------|-------------------|----------|
| **Inline** (default) | Nodes flattened into parent graph | Most cases, zero overhead |
| **SubGraph** | Child Graph Node | Conditional bodies (IF/WHILE), explicit encapsulation |

IF/WHILE bodies are **always SubGraph** — this is a CUDA API requirement (conditional nodes require child graphs).

### Buffer Passing

Buffers are passed through **explicit pins** at every CudaFunction boundary. No implicit scope capture.

```
Outer CudaFunction receives buffer "Data" as pin
  → passes it to Inner Invoke as pin
    → Inner kernel uses it as parameter

On CUDA Graph level:
  Inline: all flattened, buffer pointer bound directly to kernel param
  SubGraph: buffer pointer passed as child graph parameter
```

### Workspace Buffers (Internal)

CudaFunctions can declare internal buffer needs (e.g., cuBLAS workspace, cuFFT plan buffer). These are:
- **Invisible to the user** — no pin exposed
- **Allocated by BufferPool** during graph compilation
- **Managed by Liveness Analysis** — reused when possible

```
┌──────────────────┐
│ Invoke FFT       │
│                  │
│  Signal ──▶      │──▶ Spectrum      ← user sees these
│  ctx in  ──▶     │──▶ ctx out
│                  │
│  [workspace: 16MB] ← invisible, BufferPool-managed
└──────────────────┘
```

Optional: Workspace size configurable at Definition level, overridable at Invoke level.

---

## Upload / Download

Upload and Download are **nodes inside the Region** that compile to CUDA Graph Memcpy nodes.

### Upload (CPU → GPU)

```
┌──────────────┐
│ Upload        │
│               │
│  Spread<T> ──▶│──▶ GpuBuffer<T>
│  ctx in    ──▶│──▶ ctx out
└──────────────┘
```

Compiles to a `cuGraphAddMemcpyNode` (Host → Device).

### Download (GPU → CPU)

```
┌──────────────┐
│ Download      │
│               │
│  GpuBuffer ──▶│──▶ Spread<T>
│  ctx in    ──▶│──▶ ctx out
└──────────────┘
```

Compiles to a `cuGraphAddMemcpyNode` (Device → Host). **Expensive** — causes GPU→CPU synchronization. Should be used sparingly.

### Scalar Parameters

Scalar values (float, int, etc.) from VL flow directly to kernel parameter pins. No Upload node needed — they are set via Hot Update (`cuGraphExecKernelNodeSetParams`).

---

## FrameDelay — Double Buffer + Pointer Swap

For feedback loops (output of frame N → input of frame N+1), a FrameDelay node provides GPU-side persistence without CPU roundtrip.

### Mechanism

- **Two GpuBuffers** (A and B), swapped each frame
- Frame N: kernel writes A, FrameDelay outputs B (previous frame)
- Frame N+1: kernel writes B, FrameDelay outputs A
- **No copy** — only pointer swap (Warm Update, cheap)

### In the CUDA Graph

FrameDelay is a **virtual node** — it generates no CUDA Graph node. The GraphCompiler resolves it by:
1. Allocating two buffers from BufferPool
2. Binding the "read" buffer to dependent kernel params
3. Swapping buffer pointers between graph launches (CudaEngine.Update)
4. The swap triggers a Warm Update (pointer changed → `cuGraphExecKernelNodeSetParams`)

### VL View

```
┌──────────────┐
│ FrameDelay   │
│              │
│  In  ──▶     │──▶ Out (previous frame)
│  Init ──▶    │    ← initial value (Memset on first frame)
│  ctx in ──▶  │──▶ ctx out
└──────────────┘
```

---

## GridSize Auto

When a kernel receives a GpuBuffer as input, the grid size can be derived automatically:

```
GridDimX = ceil(buffer.Length / BlockSize)
```

### Behavior

| Scenario | GridSize |
|----------|----------|
| Buffer input exists | Auto: `ceil(firstBuffer.Length / blockSize)` |
| No buffer input (scalars only) | Must be set explicitly — error if missing |
| Manual override | User sets GridSize pin explicitly |

### VL View

```
┌──────────────────┐
│ Kernel "Forces"  │
│                  │
│  Particles ──▶   │  ← Buffer.Length = 10000
│  Grid Size: Auto │  ← = ceil(10000 / 256) = 40
│  Block Size: 256 │
└──────────────────┘
```

Default is **Auto**. The Grid Size pin shows the computed value for debugging.

---

## IF / WHILE Sub-Regions

Conditional execution and loops map to CUDA Conditional Nodes. They are implemented as **sub-regions inside the CudaGraph Region**.

### IF Region

```
┌─ CudaGraph Region ──────────────────────────────────┐
│                                                      │
│  ctx ──▶ [SetCondition] ──▶ ┌─ IF ──────────────┐  │
│                              │  [Forces]          │  │
│                              │  [Integrate]       │  │
│                              └────────────────────┘  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

- Condition is set by a GPU kernel via `cudaGraphSetConditional` (PTX intrinsic)
- IF body compiles to a Child Graph Node with `CUgraphConditionalNodeType.If`
- The body is **always a SubGraph** (CUDA API requirement)

### WHILE Region

Same pattern, but loops until condition becomes false:

```
┌─ WHILE ────────────────────────┐
│  [Iterate]                      │
│  [CheckConvergence → condition] │
└─────────────────────────────────┘
```

Maps to `CUgraphConditionalNodeType.While`.

---

## Link-Type Enforcement

VL's generic type system enforces GPU/CPU separation automatically:

```
GpuBuffer<float> ←→ GpuBuffer<float>   ✅ connects
GpuBuffer<float> ←→ Spread<float>      ❌ type mismatch
GpuBuffer<float> ←→ GpuBuffer<int>     ❌ type mismatch
float             →  float kernel pin   ✅ scalar parameter
```

No special "GPU mode" or custom type checker needed. `GpuBuffer<T>` is a generic type — VL checks compatibility at link-creation time.

---

## Three User Levels — UX Summary

### Level 1: Consumer (Most VL Users)

Places pre-made blocks inside CudaGraph Region, connects pins, tweaks scalar parameters. Zero GPU knowledge required.

```
┌─ CudaGraph Region ──────────────────────────┐
│                                              │
│  [Emitter] ──▶ [Forces] ──▶ [Integrate]    │
│                 Gravity: 9.81                │
│                                              │
└──────────────────────────────────────────────┘
```

### Level 2: Composer (Advanced VL Users)

Opens CudaFunction definitions, patches custom GPU logic using element-wise GPU nodes. Same mental model as ShaderFX. Understands parallel thinking.

```
┌─ CudaFunction "CustomForce" ─────────────────┐
│                                                │
│  Position ──▶ [GPU.Sub] ──▶ [GPU.Normalize]  │
│  Center   ──▶            ──▶ [GPU.Mul] ──▶ Force
│  Strength ──────────────────▶              │
└────────────────────────────────────────────────┘
```

### Level 3: Author (C#/CUDA Developers)

Writes PTX kernels (Triton, nvcc, hand-written), loads via filesystem. Writes CUDA C++ via NVRTC escape-hatch. Creates CudaFunction definitions with Library Calls (cuBLAS, cuFFT). Full CUDA control.

---

## Complete Particle System Example

```
┌─ CudaGraph Region ──────────────────────────────────────────────────────────┐
│                                                                              │
│  BCP In: Spread<EmitterConfig> ──Upload──▶ GpuBuffer<EmitterConfig>         │
│  BCP In: float DeltaTime ───────────────▶ (scalar param)                    │
│                                                                              │
│  ctx ──▶ [Memset Counter] ──▶ [Emitter] ──▶ [Invoke Forces] ──▶            │
│                                   │              │                           │
│                                   │  Particles   │  Particles               │
│                                   ▼              ▼                           │
│                               [FrameDelay] ──▶ [Integrate] ──▶             │
│                                                     │                        │
│                                                     ▼                        │
│  BCP Out: GpuBuffer<Particle> ───────────────▶ Stride Renderer              │
│  BCP Out: uint ParticleCount  ──Download──▶ Spread (tooltip/debug)          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

Key observations:
- FrameDelay feeds previous frame's particles back to Forces (double-buffer, no copy)
- DeltaTime flows as scalar parameter (Hot Update, no Upload)
- ParticleCount Download is optional (debug only, not on critical path)
- Forces is an Invoke of a CudaFunction defined elsewhere (reusable)
- Buffer links between Emitter → Forces → Integrate create implicit dependencies
- Context links control remaining execution order
