# VL.CudaGraph

GPU-accelerated compute pipelines for [vvvv gamma](http://vvvv.org) via NVIDIA's CUDA Graph API.

Write kernels in [Triton](https://triton-lang.org/) (Python), compile to PTX, and wire them together visually in vvvv. Everything stays on the GPU — no readback unless you ask for it.

## What It Does

VL.CudaGraph lets you build GPU compute graphs in vvvv's visual patching environment. Blocks describe GPU work (kernels, buffers, connections), and a single CudaEngine compiles and launches the entire graph each frame using CUDA's native Graph API.

**Key properties:**
- **Centralized execution** — one Engine, one CUDA Graph launch per frame
- **Passive blocks** — blocks describe structure, they never touch the GPU directly
- **Three-level dirty tracking** — only rebuild what changed (Hot/Warm/Cold)
- **Triton workflow** — author kernels in Python, ship as PTX + JSON metadata
- **Stride interop** — zero-copy sharing with VL.Stride's DX11 renderer

## Requirements

- **vvvv gamma** 6.x+
- **NVIDIA GPU** with Compute Capability 7.0+ (Volta or newer)
- **CUDA** 12.4+ (12.8+ recommended for improved conditional nodes)
- **Windows** (Linux support depends on VL.Stride availability)

## Getting Started

Install as [described here](https://thegraybook.vvvv.org/reference/hde/managing-nugets.html) via commandline:

```
nuget install VL.CudaGraph -pre
```

Usage examples and help patches are included and can be found via the [Help Browser](https://thegraybook.vvvv.org/reference/hde/findinghelp.html).

### Kernel Workflow

1. Write a Triton kernel in Python
2. Compile to PTX (`triton.compile()`)
3. Place `.ptx` + `.json` metadata files in your project
4. Use the corresponding block in vvvv — pins are generated from metadata

See `docs/architecture/PTX-LOADER.md` for the full Triton-to-PTX workflow.

## Architecture

```
Triton (Python)  →  PTX + JSON  →  VL.CudaGraph  →  CUDA Graph API  →  GPU
```

The system has three main actors:

| Component | Role |
|-----------|------|
| **Blocks** (passive) | Describe kernels, pins, connections — register with CudaContext |
| **CudaContext** (facade) | Manages block registry, connections, dirty state |
| **CudaEngine** (active) | Compiles and launches the CUDA Graph each frame |

Detailed documentation lives in `docs/architecture/`. Start with `OVERVIEW.md`.

## Project Structure

```
VL.CudaGraph/
  docs/
    architecture/          — Design documents (Overview, Execution Model, etc.)
    api/                   — C# API reference
    implementation/        — Roadmap and phase planning
  src/
    VL.Cuda.Core/          — Core library (blocks, engine, compiler, buffers)
    References/            — READ-ONLY git submodules (API reference only)
      VL.StandardLibs/     —   VL.Core, VL.Stride source
      managedCuda/         —   ManagedCuda source
```

> **Note:** The submodules in `src/References/` are for reading API source code only. They are not compiled as part of the solution — our code references ManagedCuda and VL.Core via NuGet.

## Status

🚧 **Early development** — architecture is designed, implementation is in progress. See `docs/implementation/PHASES.md` for the roadmap.

## Contributing

- Report issues on [the vvvv forum](https://forum.vvvv.org/c/vvvv-gamma/28)
- For custom development requests, please [get in touch](mailto:devvvvs@vvvv.org)
- When making a pull-request, please read the [guidelines on contributing to vvvv libraries](https://thegraybook.vvvv.org/reference/extending/contributing.html)

## Credits

- [ManagedCuda](https://github.com/kunzmi/managedCuda) — .NET bindings for the CUDA Driver API
- [Triton](https://triton-lang.org/) — Python-based kernel language by OpenAI
- [vvvv gamma](https://vvvv.org/) — visual live-programming environment for .NET

## License

TBD
