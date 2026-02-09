# VL.Skia

## 2D Graphics Rendering Library for vvvv gamma / VL

VL.Skia wraps Google's Skia graphics engine (via SkiaSharp .NET bindings) to provide a complete GPU-accelerated 2D drawing pipeline. It is the primary 2D rendering system for vvvv.

- **Target Framework:** net8.0-windows
- **License:** LGPL-3.0-only
- **GPU Backend:** OpenGL ES via ANGLE (EGL)
- **Capabilities:** 2D drawing, text, images, SVG, PDF export, GIF, Lottie/Skottie animation, touch input, video

---

## Overview

| Feature | Description |
|---------|-------------|
| **Rendering** | GPU-accelerated via EGL/OpenGL ES with SkiaSharp |
| **Scene Graph** | Linked-list layer system (`ILayer = IRendering + IBehavior`) |
| **Coordinate Spaces** | Normalized, DIP, Pixels, ManualSize |
| **Drawing** | Circles, rectangles, lines, polygons, text, images, paths, SVGs |
| **Paint System** | Composable pipeline: Fill, Stroke, Shaders, Filters, Effects |
| **Input** | Mouse, keyboard, touch with front-to-back notification propagation |
| **Off-screen** | Render to SKImage without a window |
| **Deferred** | Record/replay via SKPicture for caching and threading |

---

## 1. Core Interfaces — The Layer System

All interfaces are defined in `VL.Core.Skia/Interfaces.cs`.

### ILayer

```csharp
public interface ILayer : IRendering, IBehavior
{
    new void Render(CallerInfo caller);
    new bool Notify(INotification notification, CallerInfo caller);
    new RectangleF? Bounds { get; }
}
```

The primary scene graph node. Every visual element implements `ILayer`, combining rendering and input handling. The `new` keyword re-declares members from both parent interfaces to unify them.

### IRendering

```csharp
public interface IRendering
{
    void Render(CallerInfo caller);
    RectangleF? Bounds { get; }
}
```

Render-only interface for nodes that draw without handling input.

### IBehavior

```csharp
public interface IBehavior
{
    bool Notify(INotification notification, CallerInfo caller);
}
```

Input-only interface. Returns `true` if the notification was consumed (stops propagation).

### Sizing Enum

```csharp
public enum Sizing
{
    Pixels,      // 1 unit = 100 actual pixels
    DIP,         // 1 unit = 100 device-independent pixels (respects display scaling)
    ManualSize,  // Manually set width/height; 0 computes from aspect ratio
}
```

---

## 2. CallerInfo — The Immutable Render Context

Defined in `VL.Core.Skia/CallerInfo.cs`. An immutable `record class` that flows downstream through the scene graph.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `GRContext` | GRContext | SkiaSharp GPU context (null for CPU-only rendering) |
| `Canvas` | SKCanvas | The drawing surface |
| `Transformation` | SKMatrix | Accumulated world-to-canvas transformation |
| `ViewportBounds` | SKRect | Viewport rectangle in pixel coordinates |
| `Scaling` | float | DPI scaling factor (e.g., 1.0 for 96dpi, 1.5 for 144dpi) |
| `RenderInfoHack` | Func<object, object> | Delegate to modify paint objects flowing through the tree |
| `IsTooltip` | bool | Whether the current render pass is for a tooltip |

### Immutable Modification

Being a record, CallerInfo is modified via `with` expressions:

```csharp
public CallerInfo WithTransformation(SKMatrix transformation)
public CallerInfo WithCanvas(SKCanvas canvas)
public CallerInfo WithGRContext(GRContext context)
public CallerInfo AsTooltip => this with { IsTooltip = true };
```

### Factory

```csharp
public static CallerInfo InRenderer(float width, float height, SKCanvas canvas, GRContext context, float scaling)
```

### Coordinate Space Computation

`CallerInfoExtensions.GetWithinSpaceTransformation()` builds the matrix from coordinate space to pixels:

| Space | Behavior |
|-------|----------|
| **Normalized** | Height spans -1 to +1, origin at center, width from aspect ratio |
| **DIP** | `scaling * 100` units per dimension, origin at center |
| **DIPTopLeft** | Same as DIP but origin at top-left |
| **PixelTopLeft** | 100 px per unit, origin at top-left |
| **ManualSize** | Custom width/height with aspect-ratio preservation |

---

## 3. Scene Graph Nodes

### LinkedLayerBase — The Chain Pattern

Most scene graph nodes inherit from `LinkedLayerBase`, forming a singly-linked chain:

```csharp
public class LinkedLayerBase : ILayer
{
    protected ILayer Input;
    public virtual void Render(CallerInfo caller) => Input?.Render(caller);
    public virtual bool Notify(INotification n, CallerInfo caller) => Input?.Notify(n, caller) ?? false;
    public virtual RectangleF? Bounds => Input?.Bounds;
}
```

Subclasses override `Render` and/or `Notify` to inject behavior, calling `base.Render(caller)` to continue the chain.

### Layer Decorators (`Layers.cs`)

| Decorator | Purpose |
|-----------|---------|
| `SetRender` | Replaces render behavior with a custom `Action<ILayer, CallerInfo>` delegate |
| `SetNotify` | Replaces notification behavior with a custom `Func<INotification, CallerInfo, bool>` delegate |
| `SetBehavior` | Attaches an `IBehavior`: first tries upstream, then delegates to the behavior if unhandled |
| `SetRendering` | Prepends an `IRendering` before the upstream render (drawn behind) |
| `HackRendering<TState, TPaint>` | Intercepts paint objects flowing through the tree via a generic hack function |
| `TweakCallerInfo` | Applies a `Func<CallerInfo, CallerInfo>` transformation for both render and notify paths |

### Group (`Group.cs`)

Combines two `ILayer` inputs:

```
Render order:  Input (behind) → Input2 (in front)
Notify order:  Input2 (front) → Input (behind)  // front-to-back for input priority
```

**Recursion guard:** `maxStackCount = 10` prevents infinite loops in feedback patches.

### SpectralGroup (`Group.cs`)

Takes `IEnumerable<ILayer>` and renders all in sequence. Uses `TryGetSpan` for allocation-free iteration. Does NOT extend LinkedLayerBase.

---

## 4. The Renderer

### SkiaRenderer (Window)

A WinForms `Form` that serves as the render window (`SkiaRenderer.cs`).

**Key behavior:**
- Sets up for opaque rendering with `ControlStyles.Opaque` (Skia handles buffering)
- Creates observable streams for mouse, keyboard, touch, bounds, and focus
- Initial window size: 600×400 DIP, centered on screen
- First render call is skipped to avoid uninitialized content flash
- Custom Win32 title bar with minimize/maximize/close buttons drawn on SKCanvas

**OnPaint flow:**
```
EglSkiaRenderer.Render(hwnd, width, height, scaling, vsync, callerInfo => {
    Input?.Render(callerInfo);
    PaintTitleBarButtons(canvas);
});
```

**Input flow:**
```
INotification → CommandList.TryExecute(n) → if unhandled → Input.Notify(n, callerInfo)
```

### SkiaRendererNode — The Process Node Wrapper

The `[ProcessNode]` that VL patches interact with (`SkiaRendererNode.cs`). Manages the window lifecycle.

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| Ctrl+1 | Show the patch of this node in the editor |
| F11 / Alt+Enter | Toggle fullscreen |
| Ctrl+2 | Take screenshot |
| F2 | Toggle performance meter |

**Bounds persistence:** When `SaveBounds` is true, window bounds are saved to the VL solution (throttled at 0.5s).

### User-Facing Renderer Node (VL.Skia.vl)

The composite Renderer visible to users:

**Input Pins:**

| Pin | Type | Default | Description |
|-----|------|---------|-------------|
| `Input` | ILayer | — | Layer tree to render |
| `Title` | String | "Skia" | Window title |
| `Show Cursor` | Boolean | true | Show/hide cursor |
| `V Sync` | Boolean | true | Vertical sync |
| `Commands` | ICommandList | — | Keyboard shortcuts |
| `Enabled` | Boolean | true | Enable rendering |
| `Bounds` | Rectangle | — | Window bounds |
| `Save Bounds` | Boolean | true | Persist window position |

**Output Pins:** `Render Time`, `Form`, `Client Bounds`

---

## 5. EGL/OpenGL ES Rendering Backend

Defined in `VL.Core.Skia.Windows/EglSkiaRenderer.cs`. The full GPU render pipeline:

1. Get `RenderContext` (EGL context + GRContext)
2. Create/recreate EGL platform window surface
3. Make context current for rendering
4. Create `SKSurface` backed by GPU render target (`GRBackendRenderTarget`, `RGBA8888`, `BottomLeft` origin)
5. Set VSync via `eglContext.SwapInterval`
6. Render with `SKAutoCanvasRestore` → `renderAction(CallerInfo)` → `surface.Flush()` → `SwapBuffers`

---

## 6. Off-Screen and Deferred Rendering

### OffScreenRenderer

Renders a layer tree to an `SKImage` without a window (`OffScreenRenderer.cs`):
- Creates GPU-backed `SKSurface` (TopLeft origin, with mipmaps)
- Forwards mouse and keyboard notifications to the layer tree
- Returns `surface.Snapshot()` with reference counting

### DeferredLayer

Records rendering into `SKPicture` for caching or cross-thread replay (`DeferredLayer.cs`):
- **Producer side (Update):** Records layer render into `SKPictureRecorder`, processes queued notifications
- **Consumer side (Render):** Replays the latest recording via `canvas.DrawPicture`
- Thread-safe via `ConcurrentQueue<INotification>` and `lock`

### PictureRecorder

Simpler alternative — records a layer tree as a replayable `SKPicture` (CPU only, GRContext is null).

---

## 7. Drawing Primitives

All defined as VL Process nodes in `VL.Skia.vl` under `Graphics.Skia` category.

| Node | Description | Key Inputs |
|------|-------------|------------|
| `Circle` | Draws a circle | Position, Radius (0.5), Anchor, Paint |
| `Rectangle` | Draws a rectangle | Position, Size, Anchor, Paint |
| `Line` | Draws a line | From, To, Paint |
| `Polygon` | Draws a polygon | Points, Paint |
| `Segment` | Draws an arc segment | — |
| `Points` | Draws point set | Points, PointMode, Paint |
| `Text` | Draws text at position | Text, Position, Paint |

Each takes a `Paint` (SKPaint) input and an `Enabled` flag, and outputs an `ILayer`.

---

## 8. Paint System

Paint properties are configured through a composable chain of modifier nodes. Each takes an `Input` SKPaint and returns a modified `Output` SKPaint.

### Primary Paint Nodes

| Node | Category | Description |
|------|----------|-------------|
| `Fill` | Graphics.Skia.Paint | Sets style to Fill + color + optional shader |
| `Stroke` | Graphics.Skia.Paint | Sets style to Stroke + color + width + cap + join + shader |

### Paint Modifier Nodes

| Node | Description |
|------|-------------|
| `SetStyle` | Fill, Stroke, or StrokeAndFill |
| `SetColor` | Apply a color |
| `SetShader` | Apply a shader (gradient, noise, image) |
| `SetStrokeWidth` | Set stroke width |
| `SetStrokeCap` | Butt, Round, Square |
| `SetStrokeJoin` | Miter, Round, Bevel |
| `ImageFilter` | Blur, shadow, color matrix |
| `ColorFilter` | Color transformation matrix |
| `MaskFilter` | Blur masks |
| `DropShadow` | Drop shadow effect |
| `FontAndParagraph` | Text-related paint settings |

### Typical Pipeline

```
SKPaint → Fill(color) → Stroke(color, width) → ImageFilter(blur) → [Drawing Node]
```

---

## 9. Transformation System

| Node | Description |
|------|-------------|
| `WithinCommonSpace` | Establishes coordinate space (Normalized/DIP/Pixels) |
| `Transform` | Applies arbitrary SKMatrix via `CallerInfo.PushTransformation` |
| `Camera` | Interactive 2D pan/zoom |
| `Translate` | Translation transform |
| `Rotate` | Rotation transform |
| `Scale` | Scale transform |

---

## 10. Input Handling — The Console Node

The `Console` node (`Console.cs`) intercepts notifications flowing through the layer tree and re-exposes them as observable streams:

```csharp
public class Console : LinkedLayerBase, IMouse, IKeyboard, ITouchDevice
```

**Parameters:**
- `block` — Don't forward notifications upstream
- `force` — Capture even if already processed
- `treatAsProcessed` — Mark notifications as consumed

**Outputs:** `IMouse`, `IKeyboard`, `ITouchDevice`, `IObservable<INotification>`

---

## 11. Image Handling

Defined in `Imaging.cs`.

### IImage to SKImage Conversion

| VL PixelFormat | SKColorType | Notes |
|----------------|-------------|-------|
| R8 | Gray8 | Direct |
| R8G8B8X8 | Rgb888x | Direct |
| R8G8B8A8 | Rgba8888 | Direct |
| B8G8R8A8 | Bgra8888 | Direct |
| R16G16B16A16F | RgbaF16 | Direct |
| R32G32B32A32F | RgbaF32 | Direct |
| R16 | Gray8 | Manual 16→8 bit conversion |
| R8G8B8 / B8G8R8 | Rgba8888/Bgra8888 | Manual 24→32 bit expansion (Parallel.For) |
| B8G8R8X8 | Bgra8888 | Sets alpha to 0xFF |

- **Volatile images:** Uses `SKImage.FromPixelCopy` (copies data)
- **Non-volatile images:** Uses `SKImage.FromPixels` with release callback (zero-copy)

### SKImage to IImage

Creates an `Image` wrapper implementing `IImage`, backed by `SKPixmap.GetPixels()` with pinned memory.

---

## 12. Video Support

Located in `VL.Skia/src/Video/`:

| File | Purpose |
|------|---------|
| `TexturePool.cs` | GPU texture pooling for efficient reuse |
| `SKImageToVideoStream.cs` | Convert SKImage to video stream |
| `VideoStreamToSKImage.cs` | Convert video stream frames to SKImage |
| `VideoSourceToSKImage.cs` | Convert video sources to SKImage |

---

## 13. Initialization and Resource Management

### Assembly Initialization (`Initialization.cs`)

1. **Reference counting:** Registers `IRefCounter<SKImage>` and `IRefCounter<SKPicture>` for GPU resource lifetime management
2. **GRContext node:** Registers via node factory API (category: `Graphics.Skia.Advanced`, input: Resource Cache Limit)
3. **Typeface serializer:** Serializes `SKTypeface` by family name, weight, width, and slant

### Reference Counting

```csharp
sealed class SKObjectRefCounter : IRefCounter<SKObject>
{
    public void Init(SKObject resource) => resource?.RefCounted();
    public void AddRef(SKObject resource) => resource?.AddRef();
    public void Release(SKObject resource) => resource?.Release();
}
```

Critical for GPU-backed `SKImage` and `SKPicture` objects that need deterministic lifetime management.

---

## 14. Performance Features

### Zero-Copy Point Drawing

`CanvasExtensions` uses `MemoryMarshal.Cast<Vector2, SKPoint>` to reinterpret spans without copying, and P/Invokes directly to `sk_canvas_draw_points` in native libSkiaSharp.

### Parallel Pixel Conversion

The Imaging module uses `unsafe` pointer arithmetic and `Parallel.For` for fast 24→32 bit pixel format expansion (4 pixels per iteration for alignment).

### DPI Awareness

The entire pipeline is DPI-aware:
- `Scaling` field in CallerInfo carries the DPI factor
- DIP coordinate space scales by `scaling * 100`
- Window bounds tracked in both pixel and DIP coordinates

---

## 15. Key Design Patterns

| Pattern | Description |
|---------|-------------|
| **Linked-List Scene Graph** | Nodes form a chain via `Input` field; override Render/Notify to inject behavior |
| **Immutable CallerInfo** | Record `with` expressions ensure branch isolation in the scene graph |
| **Dual Render/Notify Paths** | Render = back-to-front; Notify = front-to-back (top element gets input first) |
| **GPU-Accelerated Default** | EGL/OpenGL ES backend; SKSurface backed by GPU render target |
| **Paint Pipeline** | Composable modifier chain for styling |
| **Decorator Pattern** | SetRender/SetNotify/SetBehavior wrap layers without changing underlying implementation |
| **Process Node Pattern** | `[ProcessNode]` classes with constructor (Create) and Update method |
| **Recursion Protection** | `stackCount` with `maxStackCount = 10` in Group/SpectralGroup |

---

## 16. Help Files

Located in `VL.Skia/help/`:

### Drawing
- Circles, rectangles, lines, polygons
- Text (alignment, word wrap, ellipsis, custom fonts, typewriter, multi-column)
- Images, SVGs, paths

### Paint
- Fill and Stroke
- FontAndParagraph for text styling
- Shaders (gradients, noise, image)
- Image filters (blur, shadow, color matrix)
- Color filters, mask filters
- Path effects (dash, trim, corner)

### Transformations
- Coordinate spaces (Normalized, DIP, Pixel)
- Translate, rotate, scale
- Camera (pan, zoom), viewports

### Input/Interaction
- Mouse, keyboard, touch
- Console node, behaviors

### Advanced
- Off-screen rendering, deferred rendering, picture recording
- PDF export, video/animation (Lottie/Skottie)
- Boolean path operations, clipping, masking
- GRContext configuration

---

## Dependencies

| Dependency | Purpose |
|------------|---------|
| SkiaSharp.Skottie | SkiaSharp + Lottie animation support |
| SharpDX.Direct3D11 | D3D11 interop for texture sharing |
| VL.Core | Runtime (NodeContext, AppHost, services) |
| VL.CoreLib | Standard types (Vector2, Color4, Mouse, Keyboard) |
| VL.Core.Skia | Interfaces (ILayer, IRendering, IBehavior, CallerInfo) |
| VL.Core.Skia.Windows | EGL rendering, Win32 title bar, touch processing |
| VL.Core.Commands | Keyboard shortcut system |
| Stride.Core.Mathematics | Math types (Color4, Vector2, Int2, RectangleF) |
| System.Reactive | Observable streams for input events |
