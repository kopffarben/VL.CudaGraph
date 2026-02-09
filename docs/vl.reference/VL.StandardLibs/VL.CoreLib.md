# VL.CoreLib

## The Foundation Library for vvvv gamma / VL

VL.CoreLib is the absolute foundation of the VL/vvvv visual programming environment. Every vvvv patch implicitly depends on it. It provides primitive types, collections, math, geometry, animation, reactive programming, channels, I/O, control flow, and more.

---

## Architecture

VL.CoreLib is a **two-layer** library:

**Layer 1 — VL Documents** (`.vl` files in root): XML documents defining the visual node API that users interact with in the vvvv editor. The main aggregator `VL.CoreLib.vl` re-exports 31 sub-documents.

**Layer 2 — C# Implementation** (`src/` directory): A .NET 8.0 class library (`VL.CoreLib.dll`, namespace `VL.Lib`) providing performant native implementations.

### Build Configuration

- **Target Framework:** net8.0
- **License:** LGPL-3.0-only
- **Unsafe Code:** Enabled

### Key NuGet Dependencies

| Package | Purpose |
|---------|---------|
| MathNet.Numerics | Linear algebra, statistics, interpolation |
| Newtonsoft.Json | JSON parsing, JSON-to-XML conversion |
| System.IO.Ports | Serial port communication |
| System.Text.Json | Modern JSON handling |
| Microsoft.Windows.CsWin32 | Windows API interop |

### Sub-Document Inventory (31 modules)

| Sub-Document | Primary Categories |
|---|---|
| `CoreLibBasics.vl` | Primitive, Math, 2D, 3D, Control, Animation basics |
| `VL.Collections.vl` | Collections (Spread, Dictionary, Sequence) |
| `VL.Animation.vl` | Animation (Sequencer, Track, Playhead) |
| `VL.Reactive.vl` | Reactive (Where, Select, Merge, Observables) |
| `VL.CoreLib.IO.vl` | IO (Socket, Net, File streams) |
| `VL.Xml.vl` | System.XML (XDocument, JSON readers/writers) |
| `VL.HTTP.vl` | IO.HTTP (HTTP clients, file parameters) |
| `VL.Threading.vl` | System.Threading.Advanced |
| `VL.Tokenizer.vl` | Text tokenization |
| `VL.Imaging.vl` | Imaging basics |
| `VL.CoreLib.Experimental.vl` | Experimental (Try/Catch regions) |
| `System.Reflection.vl` | Reflection wrappers |
| `System.Serialization.vl` | Object serialization |
| `System.Memory.vl` | Memory/Span operations |
| `VL.Regex.vl` | Regular expressions |
| `VL.Algorithms.vl` | Generic algorithms |
| `VL.Paths.vl` | Arc length, path operations |
| `VL.Bezier.Cubic.vl` | Cubic bezier splines |
| `B-Spline.vl` | B-spline evaluation |
| `Audio.vl` | Audio device interfaces |
| `Video.vl` | Video device interfaces |
| `Attributes.vl` | Editor attributes (Label, Widget, Browsable) |
| `PublicAPI.vl` | Public editor/runtime API |
| `SubChannelModule.vl` | Channel sub-modules |
| `GaussianSpread.vl` | Gaussian-distributed spread generation |
| `VL.Logging.vl` | Structured logging |
| `VL.Themes.vl` | UI theming |
| `VL.Random.vl` | Random number generation |
| `AppControl.vl` | Application lifecycle and transitions |
| `VL.Attractor.vl` | Point attractor physics |
| `VL.Typewriter.vl` | Text input with cursor management |

---

## 1. Primitive Types

VL maps these .NET types:

| VL Name | .NET Type | Source |
|---------|-----------|--------|
| `Boolean` | System.Boolean | mscorlib |
| `Byte` | System.Byte | mscorlib |
| `Char` | System.Char | mscorlib |
| `Integer32` | System.Int32 | mscorlib |
| `Integer64` | System.Int64 | mscorlib |
| `Float32` | System.Single | mscorlib |
| `Float64` | System.Double | mscorlib |
| `String` | System.String | mscorlib |
| `Vector2` | Stride.Core.Mathematics.Vector2 | Stride.Core.Mathematics.dll |
| `Vector3` | Stride.Core.Mathematics.Vector3 | Stride.Core.Mathematics.dll |
| `Vector4` | Stride.Core.Mathematics.Vector4 | Stride.Core.Mathematics.dll |

Integer conversions between all widths (Int8–Int64, UInt8–UInt64) are T4-generated in `IntegerConversions.cs`.

---

## 2. Collections

### Spread — The Primary Collection

`Spread<T>` is VL's fundamental immutable collection (defined in `VL.Core`, forwarded here).

**Key characteristics:**
- **Immutable:** All operations return new spreads
- **Ring-buffer indexing:** Index wraps via `ZMOD` (modular arithmetic) — index 5 in a 3-element spread returns element at `5 % 3 = 2`. Never throws `IndexOutOfRange`.
- **Builder pattern:** Mutable `SpreadBuilder<T>` for construction, then `ToSpread()`

**Core operations** (`CollectionNodes.cs`):

| Node | Description |
|------|-------------|
| `GetSlice` | Ring-buffer access (never throws) |
| `SetSlice` | Ring-buffer set |
| `GetList` | Sub-list with wrapping |
| `Pairwise` | Operation on consecutive element pairs |
| `Count` | Number of elements |
| `Add` / `Insert` / `RemoveAt` | Modification (returns new spread) |

### Dictionary (Immutable)

Wraps `System.Collections.Immutable.ImmutableDictionary<TKey, TValue>` with custom VL operations:
- `Create` / `CreateDefault` / `FromSequence` / `CreateRange`
- Key comparer and value comparer support

### Additional Collection Types

| Type | Description |
|------|-------------|
| `SpreadBuilder<T>` | Mutable builder for efficient Spread construction |
| `KeyValuePair<K,V>` | Dictionary entry type |
| `GaussianSpread` | Gaussian-distributed spread generation |
| `Resample` | Interpolating spreads to different sizes |
| `Trees` | Tree data structures |

---

## 3. Math Operations

### Arithmetic and Comparison

Standard binary operators (`+`, `-`, `*`, `/`), comparison (`>`, `<`, `>=`, `<=`, `=`, `!=`), boolean logic (`AND`, `OR`, `NOT`, `XOR`), unary operations (negate, absolute value), and type conversions.

### ZMOD — The VL Modulo

VL's modulo always returns non-negative results (unlike C#'s `%`). This is what enables ring-buffer indexing with negative indices.

### Range and Mapping

`Range.cs` provides value mapping between ranges (the Map node), clamping, and wrapping.

### Noise and Random

**Noise** (`Noise.cs`):
```csharp
[ThreadStatic] private static Random FRandomGenerator;  // Thread-safe

public static float Random(float from, float to = 1)
public static Vector2 Random(Vector2 from, Vector2 to)
public static Vector3 Random(Vector3 from, Vector3 to)
public static Color4 Random(Color4 from, Color4 to)
```

**SimplexNoise** (`SimplexNoise.cs`): Coherent Perlin-style noise functions.

**RandomGenerator** (VL Record): Wraps `System.Random` with optional seed (`Optional<Integer32>`).

### Tweening

`TweenerFloat32.cs`: Float32 value easing for smooth animation transitions.

### Linear Equation Solver

Matrix-based linear equation solving backed by MathNet.Numerics.

---

## 4. Color System

**Source:** `Color.cs` | **Category:** `Color`

Wraps `Stride.Core.Mathematics.Color4` (RGBA float4) with conversions between:
- **RGBA** (Red, Green, Blue, Alpha) — 0.0 to 1.0
- **HSLA** (Hue, Saturation, Lightness, Alpha)
- **HSVA** (Hue, Saturation, Value, Alpha)
- **Hex strings** (e.g., "#FF00FF")

Operations: Join/Split for all spaces, hex conversion, inversion, linear interpolation.

---

## 5. 2D Geometry and Collision

### Types

| Type | Source | Description |
|------|--------|-------------|
| `Vector2` | Stride.Core.Mathematics | 2D point/direction |
| `Circle` | VL.CoreLib.dll | Center + Radius |
| `RectangleF` | System.Drawing / Stride | Axis-aligned rectangle |

### Vector2 Operations (`Vector2.cs`)

- Split/Join components
- Near-equality comparison
- Conversion to Vector3/Vector4
- Ring-buffer component access via `GetSlice`/`SetSlice` with ZMOD
- `ref` parameters used for value types to avoid copying

### Collision Detection (`Collision.cs`)

**Category:** `2D.Collision` | **Tags:** `hittest, picking`

| Node | Description |
|------|-------------|
| `CircleContainsCircle` | Full containment test |
| `CircleContainsPoint` | Point-in-circle test |
| `CircleIntersectsCircle` | Overlap test |
| `CircleIntersectsRect` | Circle-rectangle intersection |
| `RectContainsPoint` | Point-in-rectangle test |
| `RectContainsRect` | Full containment test |

---

## 6. 3D Geometry and Transforms

All 3D types wrap Stride.Core.Mathematics types:

| Type | Stride Type | Purpose |
|------|-------------|---------|
| `Vector3` | Vector3 | 3D point/direction |
| `Vector4` | Vector4 | Homogeneous coordinates |
| `Matrix` | Matrix | 4×4 transformation |
| `Matrix3x3` | Matrix3x3 | 3×3 rotation/scale |
| `Quaternion` | Quaternion | Rotation |
| `Ray` | Ray | Origin + Direction |
| `AlignedBox` | BoundingBox | Axis-aligned bounding box |
| `OrientedBoundingBox` | OrientedBoundingBox | Rotated bounding box |
| `BoundingFrustum` | BoundingFrustum | View frustum |
| `ViewportF` | ViewportF | Viewport definition |

### Coordinate Conversions (`CoordinateSystemConversion.cs`)

Cartesian ↔ Polar, Spherical, Geographic (lat/lon), Cylindrical.

---

## 7. Animation and Timing

**File:** `VL.Animation.vl` | **Category:** `Animation`

### IClock Interface

The time abstraction used throughout VL:
- `.Seconds` → Float64 (current time in seconds)
- `.Time` → TimeInfo

The global `FrameClock` advances once per frame. Custom clocks enable deterministic replay, variable-speed playback, and networked synchronization.

### Oscillator (Damper/Spring)

A physics-based value smoother using analytical (closed-form) solutions — not numerical integration:

```csharp
public Oscillator Update(float goal, out float position, out float velocity,
                         float filterTime = 1f, float cycles = 0f, bool cyclic = false)
```

- **filterTime:** How long to reach the target (damping = `12 * (1/filterTime)`)
- **cycles:** Number of oscillation cycles (0 = critically damped)
- **cyclic:** Wrap cyclically (for angles/rotations)
- Three oscillation types: Overdamped (hyperbolic), Underdamped (oscillatory), Critically damped

### ADSR Envelope

Standard Attack-Decay-Sustain-Release envelope for audio-style amplitude control.

### Sequencer

Full VL process node for recording and playing back value sequences:
- **Record mode:** Captures time-stamped values via internal `Track` process
- **Playback mode:** `Playhead` process with Loop, Speed, Seek controls
- **Interpolation:** `BinarySearch (KeyValuePair Lerp)` for smooth value lookup

---

## 8. Reactive Programming

**File:** `VL.Reactive.vl` | **Category:** `Reactive`

### Frame-to-Observable Bridge

VL is frame-based (update loop runs every frame). The reactive module bridges to event-driven semantics via `ToObservable (Switch Factory)` — converts frame-based data changes into Rx observable streams.

### Core Reactive Nodes

| Node | Description |
|------|-------------|
| `Where` | Filters observable stream based on predicate |
| `Select` | Maps/transforms each element |
| `Merge` | Combines multiple observable streams |

### C# Observable Infrastructure (`ObservableNodes.cs`)

| Method | Description |
|--------|-------------|
| `Empty<T>()` / `Never<T>()` | Singleton empty/never observables |
| `PubRefCount<T>` | Share a single subscription |
| `OnErrorTerminate<T>` | Complete on error instead of crashing |
| `BackoffAndRetry<T>` | Retry with exponential backoff |
| `CatchAndReport<T>` | Catch exceptions and report to VL runtime |
| `Loop<TState, T>` | Background loop observable with state |

### Supporting Classes

| Class | Purpose |
|-------|---------|
| `HoldLatest` | Thread-safe latest-value buffer |
| `Sampler` | Sample observable values at frame rate |
| `SplitterNode` | Split observable into multiple streams |
| `ForEachObservable` | ForEach region for observables |
| `SafeScheduler` | Thread-safe Rx scheduler |
| `Async` | Task/async integration with observables |

---

## 9. Channels and Sub-Channels

**File:** `SubChannelModule.vl` | **Category:** `Reactive`

VL's mechanism for decoupled communication between patches. Built on top of the reactive infrastructure.

**Key concepts** (21+ help patches):
- Channel setup, binding, validation, throttling
- Public channels (accessible across documents)
- Channel presets (save/restore state)
- Channel transitions (smooth value changes)
- ForEach (Channel), bulk changes

---

## 10. I/O Primitives

**File:** `VL.CoreLib.IO.vl` | **Category:** `IO`

### Network Sockets

| Node | Description |
|------|-------------|
| `UdpClient (Reactive)` | UDP sending with optional multicast |
| `UdpServer (Reactive)` | UDP receiving with multicast support |
| `UdpSocket` | Low-level socket wrapper |
| `IPEndPoint` | Creates endpoint from address + port |
| `IPAddress` | Parses IP address strings |
| `Sender/Receiver (Datagram)` | Low-level send/receive |

### Input Devices

| Source | Types |
|--------|-------|
| `MouseNodes.cs` | Position, ClientArea, button name conversion |
| `KeyboardNodes.cs` | Key state queries |
| `TouchNodes.cs` | Touch input handling |
| `GestureNodes.cs` | Gesture recognition |
| `NotificationNodes.cs` | Base notification system |

### File System

| Source | Purpose |
|--------|---------|
| `StreamNodes.cs` | Stream read/write operations |
| `StreamObservableNodes.cs` | Reactive stream wrappers |
| `Chunk.cs` | Data chunking |
| `PathExtension.cs` | Path manipulation |
| `SpecialFolderEnum.cs` | Special folder enumeration |

### Serial Port

`SerialPort4.cs`: RS-232 serial communication via `System.IO.Ports`.

### Multi-PC Synchronization

`SyncClient.cs` / `SyncHelpers.cs`: Clock and frame-based player synchronization across networked machines.

---

## 11. Control Flow

### Switch Nodes (`SwitchNodes.cs`)

```csharp
public static T Switch<T>(bool condition, T @false, T @true) => condition ? @true : @false;
public static void SwitchOutput<T>(bool condition, T input, T @default, out T @true, out T @false);
public static void Swap<T>(bool condition, T input, T input2, out T output, out T output2);
```

### VL-Level Control Nodes

| Node | Description |
|------|-------------|
| `Changed` | Detects when input value changes (has `Changed On Create` option) |
| `S+H` (Sample and Hold) | Latches value when Sample pin triggers |
| `TogEdge` | Detects rising (Up Edge) and falling (Down Edge) transitions |
| `Switch (Boolean)` | Selects between two inputs |
| `OR` / `AND` / `NOT` | Boolean logic |

### Type Switches (`TypeSwitches.cs`)

Runtime type-based dispatching — selecting behavior based on concrete type.

---

## 12. Text Processing

| Module | Description |
|--------|-------------|
| `Text.cs` | String manipulation utilities |
| `Encodings.cs` | Text encoding support (UTF-8, ASCII, etc.) |
| `VL.Tokenizer.vl` | String tokenization/splitting |
| `VL.Regex.vl` | Regular expressions |
| `VL.Typewriter.vl` | Text input with cursor, selection, constraints |

---

## 13. XML and JSON

**File:** `VL.Xml.vl` | **Category:** `System.XML`

| Node | Description |
|------|-------------|
| `FileReader (JSON)` | Reads .json and converts to XDocument |
| `JoinXDocument` / `SplitXDocument` | Construct/deconstruct XDocument |
| `JoinXDeclaration` / `SplitXDeclaration` | XML declaration handling |
| XPath | XPath evaluation |
| XSLT | XSLT transformation |
| JSON-to-XML | Via `Newtonsoft.Json.JsonConvert` |

---

## 14. HTTP Networking

**File:** `VL.HTTP.vl` | **Category:** `IO.HTTP`

- `FileParameter` record for HTTP file upload (Filename, MediaType, Data)
- Experimental HTTP server (`HttpServer.cs`)
- HTTP request utilities (`WebRequest.cs`)
- Network interface enumeration (`NetworkInterfaceUtils.cs`)

---

## 15. Logging

**File:** `VL.Logging.vl` | **Category:** `System.Logging`

### Log Process Node

Logs into vvvv's Log Window:

| Pin | Type | Description |
|-----|------|-------------|
| `Message` | String | Log message (supports `{0}` argument placeholders) |
| `Log Level` | LogLevel | Trace/Debug/Information/Warning/Error/Critical |
| `Argument` | Spread<Object> | Pin group for message arguments (Ctrl+Click to add) |
| `Category` | String | Log category |
| `Exception` | Optional | Exception to log |

### Forwarded Types

`EventId`, `LogLevel`, `ILoggerFactory`, `ILoggerProvider` from `Microsoft.Extensions.Logging.Abstractions`.

---

## 16. Threading

**File:** `VL.Threading.vl` | **Category:** `System.Threading.Advanced`

- `SynchronizationContext` forward for UI thread marshaling
- `BackgroundAction.cs` — background thread execution
- `Async.cs` — bridges async/await with VL frame loop and Rx observables

---

## 17. Bezier Curves and Splines

### Cubic Bezier (`VL.Bezier.Cubic.vl`)

| Type | Description |
|------|-------------|
| `BezierSpline` | Record: `Knots` (Spread<BezierKnot>) |
| `BezierKnot` | Control point with tangent handles |
| `BezierSegment` | Cubic curve between two knots (4 control points) |
| `BezierSplineSpread` | Samples points along a spline |

### B-Spline (`B-Spline.vl`)

B-spline curve evaluation using basis functions and knot vectors.

### Arc Length (`VL.Paths.vl`)

Arc-length parameterization for uniform-speed traversal along curves.

---

## 18. Application Control

**File:** `AppControl.vl` | **Category:** `Application`

- `ITransitionModel (TimeBoxed)` — interface for time-bounded transitions
- `BezierTransitionModel (TimeBoxed)` — record with Warmup, Cooldown, Duration
- `ConstructBezier (V)` — builds BezierSegments for transition curves

---

## 19. Editor Attributes

**File:** `Attributes.vl` | **Category:** `System.EditorAttributes`

| VL Name | .NET Type | Purpose |
|---------|-----------|---------|
| `LabelAttribute` | VL.Core.EditorAttributes.LabelAttribute | Custom pin/node labels |
| `WidgetType` | VL.Core.EditorAttributes.WidgetType | Widget type enum |
| `WidgetTypeAttribute` | VL.Core.EditorAttributes.WidgetTypeAttribute | Assign widget types to pins |
| `BrowsableAttribute` | System.ComponentModel.BrowsableAttribute | Show/hide in editor |
| `TypeSelectorAttribute` | VL.Core.EditorAttributes.TypeSelectorAttribute | Type selection dropdown |
| `NonSerializedAttribute` | System.NonSerializedAttribute | Exclude from serialization |

---

## 20. The Process Node Pattern

The fundamental design pattern of VL for stateful nodes:

### Lifecycle

1. **Create** — Called once on instantiation. Sets initial state, allocates resources.
2. **Update** — Called every frame. Reads inputs, computes outputs, manages state.

### State

State is stored in `Slot` elements, accessed via `Pad` elements within the canvas.

### Pin Types

| Kind | Description |
|------|-------------|
| `InputPin` | Standard input — read each frame |
| `OutputPin` | Standard output — written each frame |
| `StateInputPin` | Receives the process's own state |
| `StateOutputPin` | Passes state downstream |
| `ApplyPin` | Boolean "execute" trigger |

---

## 21. Adaptive Nodes

VL mechanism for type-polymorphic nodes without explicit generics. The compiler selects the appropriate implementation based on connected types.

Example: The `+` node works on scalars, vectors, colors — VL automatically selects `Vector2.Add`, `Single.Add`, etc. based on what's connected.

---

## 22. Cross-Cutting Patterns

| Pattern | Description |
|---------|-------------|
| **Ring-buffer indexing** | `ZMOD` everywhere — never throws IndexOutOfRange |
| **Immutable-first** | Primary types (Spread, Dictionary) are immutable; builders for construction |
| **IClock abstraction** | Decouples time from system clock |
| **Reactive bridging** | `ToObservable (Switch Factory)` + `Changed` detection |
| **ref parameters** | Value types use `ref` in C# to avoid copying |
| **Process Node pattern** | Create/Update lifecycle for all stateful nodes |

---

## Help Documentation

Located in `VL.CoreLib/help/` with 150+ help patches:

| Directory | Topics |
|-----------|--------|
| `2D/` | Vector angles, distances, bounding rects, random walks |
| `Animation/` | Frame differences, counting, time ramps, smoothing |
| `API/` | Tooltips, element preview, node inspection |
| `Collections/` | Spread operations, dictionary, queue, GaussianSpread, Resample |
| `Color/` | All color space conversions, hex, inverse, interpolation |
| `Control/` | Toggling, edge detection, gating, send/receive, exceptions |
| `IO/` | File, JSON, Keyboard, Mouse, Path, Serialization, UDP, Web, XML (53+ patches) |
| `Math/` | Range mapping, quantization |
| `Primitive/` | Number/string conversion, coordinate systems, enums, random |
| `Reactive/` | Channels, observables, tasks, threads, presets (21+ patches) |
| `System/` | CLI args, date/time, console, external programs, logging |
| `Transform/` | Homography, billboard |

---

## Dependencies

| Dependency | Purpose |
|------------|---------|
| VL.Core | Runtime (NodeContext, AppHost, IClock, Spread<T>, RuntimeGraph) |
| VL.Serialization.Raw | Binary serialization for VL.Sync |
| Stride.Core.Mathematics | All math types (Vector2/3/4, Matrix, Quaternion, Color4) |
| System.Reactive | Observable streams |
| System.Collections.Immutable | Immutable collections |
| MathNet.Numerics | Numerical computing |
| Newtonsoft.Json | JSON parsing |
| System.IO.Ports | Serial communication |
| Microsoft.Extensions.Logging.Abstractions | Logging interfaces |
