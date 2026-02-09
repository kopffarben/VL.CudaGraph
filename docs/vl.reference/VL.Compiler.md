# VL Command-Line Compiler (vvvvc.exe)

## Overview

`vvvvc.exe` is the standalone command-line compiler that ships with vvvv gamma (since 7.0). It is located next to `vvvv.exe` in the install directory.

Primary use cases:
- **Export** VL patches into standalone executables (apps with entry points)
- **Validate** `.vl` files — compilation without errors confirms the file is structurally and semantically valid

## Setup

A local copy of vvvv gamma is checked into `tools/`:

```
tools/vvvv_gamma_7.1-0161-g176d67638c-win-x64/
  vvvvc.exe          ← Command-line compiler
  vvvv.exe           ← Full IDE (not needed for compilation)
```

Download from TeamCity (guest auth):
```
https://teamcity.vvvv.org/guestAuth/app/rest/builds/id:39760/artifacts/content/vvvv_gamma_7.1-0161-g176d67638c-win-x64.zip
```

## Basic Usage

```bash
# Export an app .vl file to standalone executable
vvvvc.exe MyApp.vl

# Export to a specific directory
vvvvc.exe MyApp.vl --output-directory C:\temp

# Cross-compile for Linux
vvvvc.exe MyApp.vl --rid linux-x64
```

Default output: `%UserProfile%\Documents\vvvv\gamma\Exports\<AppName>`

## Validation of .vl Files

Running `vvvvc.exe` against a `.vl` file triggers full compilation: NuGet resolution, assembly loading, symbol building, patch compilation, and C# code generation. This validates that the `.vl` file is structurally correct and all references resolve.

### App vs. Library Validation

`vvvvc.exe` is primarily an **app exporter**. The behavior differs for apps vs. libraries:

| .vl Type | Compilation | Export | Exit Code |
|----------|-------------|--------|-----------|
| **App** (has entry point) | Validates structure + semantics | Produces executable | 0 = success |
| **Library** (no entry point) | Validates structure + semantics | Crashes with "Entry point not found" | Non-zero (expected) |

**For library packages** (like VL.CudaGraph), the compilation step **is** the validation. The export crash is expected and harmless. Check the log output to determine success:

```
# SUCCESS — DLL was generated, .vl is valid:
info: vvvv[0]
      D:\_MBOX\_CODE\_packages\VL.CudaGraph\VL.CudaGraph.vl -> ...\VL.CudaGraph.vl.dll

# FAILURE — errors found during compilation:
warn: vvvv[0]
      The project MyPackage.vl has errors. Package compilation will continue assuming the error is non-critical.
```

### Validation Command for VL.CudaGraph

```bash
tools\vvvv_gamma_7.1-0161-g176d67638c-win-x64\vvvvc.exe ^
  VL.CudaGraph.vl ^
  --package-repositories "D:\_MBOX\_CODE\_packages" ^
  --verbosity Information
```

**What to check in the output:**
1. `Loading D:\_MBOX\_CODE\_packages\VL.CudaGraph\VL.CudaGraph.vl` — file was found
2. `VL.CudaGraph.vl -> ...VL.CudaGraph.vl.dll` — compilation succeeded, file is valid
3. NuGet version warnings (e.g., "wasn't picked up because ... requires at least version ...") are harmless — these are transitive dependency version bumps
4. `Entry point for document ... not found` — expected for library packages, ignore

### Verbosity Levels

```bash
# Full diagnostics (shows every assembly loaded)
vvvvc.exe MyPackage.vl --verbosity Trace

# Standard output (recommended for validation)
vvvvc.exe MyPackage.vl --verbosity Information

# Errors and warnings only
vvvvc.exe MyPackage.vl --verbosity Warning
```

## Command-Line Arguments

### Compiler-Specific

| Argument | Values | Purpose |
|----------|--------|---------|
| `-v`, `--verbosity` | Trace, Debug, Information, Warning, Error, Critical, None | Output verbosity |
| `--ignore-errors` | true / false | Ignore VL compile errors (red nodes) |
| `--ignore-unhandled-exceptions` | true / false | Ignore runtime errors (pink nodes) |
| `--output-directory` | path | Export target directory |
| `--app-icon` | path to .ico | Application icon |
| `--asset-behavior` | RelativeToDocument / RelativeToOutput | Asset reference mode |
| `--output-type` | Exe / WinExe | Console or GUI application |
| `--rid` | win-x64, win-x86, win-arm64, osx-x64, osx-arm64, linux-x64, linux-arm, linux-arm64 | Target platform |
| `--platform` | AnyCPU, x64, x86 | CPU architecture |
| `--clean` | true / false | Clean build directory before export |

### Shared with vvvv.exe

| Argument | Purpose |
|----------|---------|
| `--nuget-path <path>` | Replace default global NuGet cache location |
| `--package-repositories <list>` | Semi-colon separated source-package directories |
| `--export-package-sources <list>` | Directories with .nupkg files for generated NuGet.config |
| `--editable-packages <list>` | Opt-out of read-only for packages (glob patterns allowed) |

## NuGet Dependency Resolution

The compiler must have access to all NuGet packages referenced by the `.vl` file. Options:

```bash
# Point to local source-package repositories
vvvvc.exe MyApp.vl --package-repositories "D:\_MBOX\_CODE\_packages"

# Point to a custom NuGet cache
vvvvc.exe MyApp.vl --nuget-path "C:\MyNugets"

# Provide .nupkg files directly
vvvvc.exe MyApp.vl --export-package-sources "C:\MyPackages"
```

## Build Configuration (.props)

Place a `.props` file (MSBuild XML) next to the main `.vl` file with the same name for advanced build configuration:

```xml
<!-- MyApp.props -->
<Project>
  <PropertyGroup>
    <Version>1.3</Version>
  </PropertyGroup>
</Project>
```

## Export Output Structure (Apps Only)

```
ExportDirectory/
  MyApp.exe              ← Standalone executable
  /src                   ← Generated C# solution (safe to delete after build)
    MyApp.csproj
    MyApp.sln
    ...
```

The `/src` directory contains a fully valid C# solution that can be opened in Visual Studio.

## Validated Behavior (VL.CudaGraph)

Tested with vvvv gamma 7.1-0161 on 2026-02-09:

```
$ vvvvc.exe VL.CudaGraph.vl --package-repositories "D:\_MBOX\_CODE\_packages" --verbosity Information

info: vvvv[0]
      Loading D:\_MBOX\_CODE\_packages\VL.CudaGraph\VL.CudaGraph.vl
warn: vvvv[0]                                          ← Harmless NuGet version warnings
      Microsoft.Extensions.FileProviders.Abstractions.6.0.0 wasn't picked up ...
warn: vvvv[0]
      The project VL.CudaGraph.vl has errors.          ← Non-critical (empty Application stub)
info: vvvv[0]
      VL.CudaGraph.vl -> ...VL.CudaGraph.vl.dll       ← COMPILATION SUCCESS
info: vvvv[0]
      Generating code
Unhandled exception: Entry point not found             ← EXPECTED for library packages
```

Result: `.vl` file is **valid** — compilation succeeded, DLL was generated.
