# VL.StandardLibs Documentation

## Technical Reference for Coding Agents

This documentation provides comprehensive technical reference for the VL.StandardLibs repository — the standard library ecosystem of [vvvv gamma](https://visualprogramming.net/), a visual programming environment for .NET.

These docs are written for **coding agents** that need to understand, generate, or modify VL code and `.vl` files. They cover the file format specification, layout best practices, and detailed library API documentation.

---

## Documentation Index

### VL File Format

The `.vl` file format is XML-based and defines visual dataflow programs. These two documents cover everything needed to programmatically generate valid `.vl` files.

| Document | Description |
|----------|-------------|
| [VL.Fileformat.Description](VL.Fileformat/VL.Fileformat.Description.md) | Complete specification of the `.vl` XML format: document structure, namespaces, ID encoding, element types (Document, Patch, Canvas, Node, Pin, Pad, Link, Fragment, Slot), serialization rules, property encoding, NodeReference/CategoryReference patterns, and all element kinds |
| [VL.Fileformat.BestPractices](VL.Fileformat/VL.Fileformat.BestPractices.md) | Layout and patching conventions extracted from 769+ production `.vl` files: data flow direction, node spacing, alignment, sizing, multi-section layouts, fan-out patterns, link routing, and annotated real-world examples with pixel measurements |

### Command-Line Compiler

| Document | Description |
|----------|-------------|
| [VL.Compiler](VL.Compiler.md) | `vvvvc.exe` command-line compiler: export VL patches to standalone executables, validate `.vl` files, NuGet dependency resolution, cross-compilation (win/osx/linux), CI integration, build configuration via `.props` files |

### Standard Libraries

The VL standard libraries provide the complete node API for vvvv gamma — from primitives and math to 3D rendering, GUI, and networking.

| Document | Library | Description |
|----------|---------|-------------|
| [VL.CoreLib](VL.StandardLibs/VL.CoreLib.md) | VL.CoreLib | **Foundation library.** Primitive types, Spread collections (ring-buffer indexing), math, Color (RGBA/HSLA/HSVA), 2D/3D geometry, animation (Oscillator, Sequencer, ADSR), reactive programming (Observables, Channels), I/O (UDP, Serial, File), control flow (Switch, Changed, S+H, TogEdge), logging, threading. Every VL patch implicitly depends on this. |
| [VL.Skia](VL.StandardLibs/VL.Skia.md) | VL.Skia | **2D rendering.** GPU-accelerated via EGL/OpenGL ES. Layer-based scene graph (`ILayer = IRendering + IBehavior`), immutable `CallerInfo` render context, composable paint pipeline (Fill, Stroke, Shaders, Filters), coordinate spaces (Normalized, DIP, Pixels), mouse/keyboard/touch input with front-to-back propagation, off-screen rendering, deferred rendering via SKPicture. |
| [VL.Stride](VL.StandardLibs/VL.Stride.md) | VL.Stride | **3D rendering.** Built on the Stride game engine (Direct3D 11). Entity-Component System, cameras (perspective/orthographic, orbit), forward rendering pipeline, PBR materials (7 microfacet models), lights/shadows/skybox IBL, 15+ post-processing effects, SDSL/HLSL shader system with 4 node types (TextureFX, DrawFX, ComputeFX, ShaderFX) and hot-reload, 80+ TextureFX effects, Bullet physics, OpenXR VR. |
| [VL.ImGui](VL.StandardLibs/VL.ImGui.md) | VL.ImGui | **Immediate mode GUI.** Wraps Dear ImGui with 50+ widgets (sliders, drags, inputs, trees, tabs, tables, menus, popups), bidirectional VL Channel binding, 20+ composable style nodes, primitive drawing, item state queries. Two backends: Skia (2D vertex rendering) and Stride (GPU shader pipeline). Supports embedding native renderer content inside ImGui panels. |
| [VL.Serialization](VL.StandardLibs/VL.Serialization.md) | VL.Serialization.* | **Serialization.** Four backends: Core XML (built-in Persistent node), FSPickler (XML/JSON/Binary for debugging), MessagePack (high-performance binary + JSON mode), Raw (zero-copy memory blitting for GPU/audio buffers). |
| [VL.Video](VL.StandardLibs/VL.Video.md) | VL.Video | **Video playback and capture.** Windows-only via Media Foundation. VideoPlayer (file/URL with GPU D3D11 or CPU WIC rendering paths), VideoIn (USB camera capture via UVC 1.1), audio support, Skia and Stride integration. |
| [VL.IO](VL.StandardLibs/VL.IO.md) | VL.IO.* | **Input/Output.** Six subsystems: File I/O (stream-based resource providers), HTTP (server, downloader, WebRequest), Redis (channel-to-key binding with per-frame transactions), OSCQuery (parameter tree sync), Named Pipes (IPC), TPL Dataflow (concurrent pipelines with Rx integration). |

---

## Library Dependency Graph

```
VL.CoreLib (foundation — everything depends on this)
│
├── VL.Skia (2D rendering)
│     └── VL.ImGui.Skia (ImGui 2D backend)
│
├── VL.Stride (3D rendering)
│     ├── VL.ImGui.Stride (ImGui 3D backend)
│     └── VL.Stride.TextureFX (80+ GPU texture effects)
│
├── VL.ImGui (core widget library, backend-agnostic)
│
├── VL.Serialization.FSPickler
├── VL.Serialization.MessagePack
├── VL.Serialization.Raw
│
├── VL.Video (video playback/capture)
│
├── VL.IO.Redis
├── VL.IO.OSCQuery
├── VL.IO.Pipes
└── VL.TPL.Dataflow
```

---

## Key Concepts for Agents

### The Process Node Pattern

Every stateful VL node follows the **Create/Update** lifecycle:
- **Create** — runs once on instantiation (constructor)
- **Update** — runs every frame (tick method)

State is stored in Slots, accessed via Pads on the canvas.

### Spread — The Primary Collection

`Spread<T>` is VL's immutable collection with **ring-buffer indexing** via `ZMOD` (modular arithmetic). Index wrapping means `spread[5]` on a 3-element spread returns `spread[5 % 3]` — no out-of-bounds exceptions.

### Reactive Bridging

VL bridges its frame-based execution model to Rx Observables through `ToObservable (Switch Factory)` + `Changed` detection. This pattern appears throughout the reactive, channel, and I/O subsystems.

### Rendering Architectures

| Renderer | Scene Model | Graphics API | Key Interface |
|----------|-------------|-------------|---------------|
| **VL.Skia** | Linked-list layer chain | OpenGL ES (EGL) | `ILayer` (Render + Notify) |
| **VL.Stride** | Entity-Component System | Direct3D 11 | Entity + Components |
| **VL.ImGui** | Widget tree | Via Skia or Stride backend | `Widget` (UpdateCore) |

### Coordinate Conventions

- VL.Skia: 1 unit = 100 pixels (Normalized space: height -1 to +1)
- VL.Stride: 7 screen spaces (World, View, Projection, Normalized, DIP, DIPTopLeft, PixelTopLeft)
- VL.ImGui: "Hecto" scaling (1 VL unit = 100 ImGui pixels)

### Serialization Selection Guide

| Scenario | Use |
|----------|-----|
| Settings/config files | Core XML (Persistent node) |
| Debugging serialized data | FSPickler (JSON mode) |
| Network/real-time | MessagePack (binary) |
| GPU/audio buffers | Raw (zero-copy) |

### I/O Selection Guide

| Scenario | Use |
|----------|-----|
| File read/write | File I/O |
| REST APIs / downloads | HTTP |
| Shared state across instances | Redis |
| OSC-compatible software | OSCQuery |
| Fast IPC on same machine | Named Pipes |
| Parallel processing | TPL Dataflow |
