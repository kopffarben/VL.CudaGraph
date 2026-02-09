# VL.ImGui

## Immediate Mode GUI for vvvv gamma / VL

VL.ImGui wraps [Dear ImGui](https://github.com/ocornut/imgui) (via ImGuiNET) to provide a complete GUI widget system with two rendering backends (Skia 2D and Stride 3D). It features bidirectional channel binding, composable styling, auto-generated retained/immediate mode nodes, and support for embedding native renderer content inside ImGui panels.

---

## Overview

| Feature | Description |
|---------|-------------|
| **Widget System** | 50+ widgets: sliders, drags, inputs, buttons, checkboxes, trees, tabs, tables, menus, popups |
| **Layout** | Row, Column, Fold, windowing, child windows, docking |
| **Data Binding** | Bidirectional VL Channel binding with attribute-driven defaults |
| **Styling** | 20+ composable style nodes with Optional properties |
| **Primitive Drawing** | Lines, rectangles, circles, ellipses, triangles, beziers, text |
| **Backends** | Skia (2D, vertex-based) and Stride (3D, GPU shader pipeline) |
| **Embedding** | Embed Skia layers and Stride render targets inside ImGui panels |
| **Docking** | ImGui docking support for multi-panel layouts |

---

## Package Structure

**`VL.ImGui`** — Core backend-agnostic library. All widget definitions, Context class, channel binding, styling, layout, queries, code generation system.

**`VL.ImGui.Skia`** — 2D rendering backend. Converts ImGui draw data into SkiaSharp vertex calls. Implements `ILayer` for Skia pipeline integration.

**`VL.ImGui.Stride`** — 3D rendering backend. Converts ImGui draw data into Stride GPU draw calls via custom SDSL shader. Extends `RendererBase` for Stride integration.

---

## 1. Core Architecture: Context and Widget

### Widget Base Class

Every UI element inherits from `Widget` (`Core/Widget.cs`):

```csharp
public abstract class Widget : IDisposable
{
    internal abstract void UpdateCore(Context context);   // Implement ImGui API calls here

    [Pin(Priority = 10)]
    public IStyle? Style { set; protected get; }          // Per-widget styling

    protected virtual bool HasItemState => true;          // For item query system
    protected WidgetLabel widgetLabel = new();             // Auto-unique ID management
}
```

**Key design:**
1. `UpdateCore(Context)` is the abstract method every widget implements with actual ImGui calls
2. `Update(Context?)` is the non-virtual entry point that validates context, clears item state, applies style, then calls `UpdateCore`
3. `WidgetLabel` auto-appends `##<uniqueId>` to every label to prevent ImGui ID collisions
4. Every widget has a `Style` pin for individual styling

### Context Class

The central orchestrator (`Core/Context.cs`), wrapping a native ImGui context pointer:

```csharp
public class Context : IDisposable
{
    [ThreadStatic] internal static Context? Current = null;  // Thread-local

    internal ImDrawListPtr DrawListPtr;
    internal DrawList DrawList;
    internal ItemState? CapturedItemState { get; set; }
    internal readonly Dictionary<string, ImFontPtr> Fonts;
}
```

**Frame lifecycle:**
1. `NewFrame()` — resets all widgets from previous frame, calls `ImGui.NewFrame()`
2. `MakeCurrent()` — saves/restores ImGui context (returns disposable `Frame`)
3. `Update(Widget?)` — renders a widget and queues it for next frame reset
4. `ApplyStyle(IStyle?)` — returns disposable `StyleFrame` (push on create, pop on dispose)

---

## 2. Node Generation System

### The GenerateNode Attribute

Widgets annotated with `[GenerateNode]` are auto-registered as VL nodes at startup:

```csharp
[GenerateNode]
public class GenerateNodeAttribute : Attribute
{
    public string? Name { get; set; }           // Override node name
    public string? Category { get; set; }       // VL category
    public bool GenerateRetained = true;        // Generate retained-mode node
    public bool GenerateImmediate = true;       // Generate immediate-mode node
    public bool IsStylable = true;              // Expose Style pin
}
```

**Two modes per widget:**
- **Retained mode:** Widget persists across frames, properties set via pins (default for most widgets)
- **Immediate mode:** Widget configured via `Action<Context>` callback

Container widgets (Window, TreeNode, TabBar, etc.) typically only generate retained mode (`GenerateImmediate = false`).

---

## 3. Channel Binding System

### ChannelWidget<T> — Bidirectional Data Binding

Every value-editing widget (sliders, checkboxes, text inputs, combos, color pickers) inherits from `ChannelWidget<T>`:

```csharp
internal abstract class ChannelWidget<T> : Widget, IHasLabel
{
    public IChannel<T>? Channel { protected get; set; }   // Bidirectional binding
    public string? Label { get; set; }
    public bool Bang { get; private set; }                 // True when value changed this frame
    public T? Value { get; protected set; }                // Current value
}
```

**Value flow:**
- **Reading:** Each frame, reads current value from the channel
- **Writing:** When the user edits the value, writes back to the channel and sets `Bang = true`
- **Attribute-driven defaults:** When a channel is connected, its `AttributesChannel` is subscribed. Min/Max ranges, labels, and other metadata are automatically applied via `ValueSelector` instances.

### ChannelFlange<T> — Container State Binding

For container state (visibility, collapsed, bounds) that isn't simple value editing:

```csharp
// WindowCore example:
public IChannel<bool>? Visible { private get; set; }
ChannelFlange<bool> VisibleFlange = new ChannelFlange<bool>(true);
```

Enables two-way synchronization: set visibility from VL, and ImGui reports changes back.

---

## 4. Complete Widget Catalog

### Buttons

| Widget | Type | Key Pins | Tags |
|--------|------|----------|------|
| `Button` | Unit | `Size` (Vector2) | bang |
| `ButtonSmall` | Unit | — | Compact, no padding |
| `InvisibleButton` | Unit | `Size`, `Flags` | Hit-test with no visual |

### Checkboxes and Toggles

| Widget | Type | Description |
|--------|------|-------------|
| `Checkbox` | bool | Standard toggle (tags: "toggle") |
| `Selectable` | bool | Highlights on hover, extends to fill |
| `RadioButtons` | int | One radio button per label string |

### Sliders

All provide `Min`, `Max`, `Format`, `Flags` pins:

| Widget | Type | Default Range |
|--------|------|--------------|
| `Slider (Float)` | float | 0–1 |
| `Slider (Float64)` | double | 0–1 |
| `Slider (Int)` | int | 0–100 |
| `Slider (Int2/3/4)` | Int2/3/4 | Per-component |
| `Slider (Vector2/3/4)` | Vector2/3/4 | Per-component |
| `SliderFloatVertical` | float | Vertical orientation + Size |

### Drag Widgets

All provide `Speed`, `Min`, `Max`, `Format`, `Flags`, `NotifyWhileTyping` pins:

| Widget | Type | Default Speed |
|--------|------|--------------|
| `Drag (Float/Float64)` | float/double | 0.01 |
| `Drag (Int/Int2/3/4)` | int variants | 1 |
| `Drag (Vector2/3/4)` | Vector2/3/4 | 0.01 |
| `Drag (Quaternion)` | Quaternion | 0.01 |
| `Drag (TimeSpan)` | TimeSpan | Special |
| `DragFloatRange` / `DragIntRange2` | Range pair | Two-handle |

**NotifyWhileTyping:** When false (default), during keyboard editing values are held until the user deactivates the widget (tab out or enter). During mouse dragging, values are pushed immediately.

### Text Input

| Widget | Type | Key Pins |
|--------|------|----------|
| `Input (String)` | string | `MaxLength` (100), `Flags` |
| `InputTextMultiline` | string | `MaxLength`, `Size`, `Flags` |
| `InputTextWithHint` | string | `MaxLength`, `Hint`, `Flags` |
| `Input (Int2/3/4)` | Int variants | `Flags` |
| `Input (Vector2/3/4)` | Vector variants | `Format`, `Flags` |

### Color Widgets

| Widget | Type | Key Pins |
|--------|------|----------|
| `ColorPicker` | Color4 | `Flags` (tags: "rgba, hsv, hsl") |
| `ColorButton` | Color4 | `Flags`, `Size` |

### List/Selection

| Widget | Type | Key Pins |
|--------|------|----------|
| `Combo (String)` | string | `Items`, `Flags` (tags: "dropdown, pulldown, enum") |
| `ListBox` | string | `Items`, `Size` |

### Display/Text

`Text`, `TextWrapped`, `TextBullet`, `TextLabel`, `Bullet`, `Separator`, `SeparatorText`

### Plotting and Progress

| Widget | Key Pins |
|--------|----------|
| `PlotLines` | `Values`, `Offset`, `ScaleMin`, `ScaleMax`, `Size` |
| `PlotHistogram` | Same as PlotLines |
| `ProgressBar` | `Fraction` (0–1), `OverlayText`, `Size` |

### Debugging

`DemoWindow`, `MetricsWindow`, `AboutWindow`, `IDStackToolWindow`, `UserGuide`

---

## 5. Layout System

### Layout Containers

| Container | Description |
|-----------|-------------|
| `Column` | Vertical stack (natural ImGui flow) with BeginGroup/EndGroup |
| `Row` | Horizontal layout; divides width equally, uses SameLine between children |
| `Fold` | Simply renders children sequentially without grouping |
| `Dummy` | Adds empty space of specified size |

### Layout Commands

| Command | Purpose |
|---------|---------|
| `SameLine` | Next widget on same line |
| `NewLine` | Force new line |
| `Spacing` | Vertical spacing |
| `Indent` / `Unindent` | Horizontal indentation |
| `Group` | Wrap in BeginGroup/EndGroup |
| `SetCursorPosition` | Absolute positioning within window |
| `SetItemWidth` | Width for subsequent items |

---

## 6. Windowing and Containers

### WindowCore

The primary floating window:

| Pin | Type | Direction | Description |
|-----|------|-----------|-------------|
| `Name` | string | Input | Title (auto-appended with unique ID from NodeContext) |
| `Content` | Widget | Input | Child widget tree |
| `Visible` | IChannel<bool> | Bidirectional | Show/hide |
| `Collapsed` | IChannel<bool> | Bidirectional | Collapse state |
| `Bounds` | IChannel<RectangleF> | Bidirectional | Position and size |
| `Closing` | Optional<IChannel<Unit>> | Input | Close button (shows when provided) |
| `Flags` | ImGuiWindowFlags | Input | All ImGui window flags |
| `ContentIsVisible` | bool | Output | Whether content is rendering |

### ChildWindow

Scrollable, clippable sub-region. Size convention: 0 = remaining space, positive = fixed, negative = remaining minus abs value.

### TreeNode

Collapsible tree with bidirectional `Collapsed` channel. Uses `CaptureItemState()` to preserve item state across subtree for query widgets.

### CollapsingHeader

Full-width header bar with optional close button. Supports `Visible` and `Closing` channels.

### TabBar + TabItem

Tab container with per-tab `Visible`, `Active` (bidirectional), `Closing` channels.

### Menu System

| Widget | Description |
|--------|-------------|
| `MainMenuBar` | Screen-top menu bar |
| `MenuBar` | In-window menu bar |
| `Menu` | Individual dropdown with Content, Label, Enabled |

### Popup

Floating popup with bidirectional `Position` and `Visible` channels.

### TableCore

ImGui table with configurable column count, content, label, size, flags.

### Disabled

Wraps content in `BeginDisabled`/`EndDisabled` when `Apply` is true.

---

## 7. Styling and Theming

### Architecture

Composable via chaining through `Input` pin:

```csharp
public interface IStyle
{
    void Set(Context context);    // Push onto ImGui style stack
    void Reset(Context context);  // Pop from style stack
}
```

All properties use `Optional<T>` — only explicitly set values are pushed. This makes styles composable without unintended overrides.

### Style Nodes (category: `ImGui.Styling`)

| Style Node | Key Properties |
|------------|---------------|
| `SetAlphaStyle` | Alpha, DisabledAlpha |
| `SetButtonStyle` | Background, Hovered, Active (Color4), TextAlign |
| `SetCheckboxStyle` | CheckMark color |
| `SetFrameStyle` | Background, BackgroundHovered, BackgroundActive, Padding, Rounding |
| `SetGrabStyle` | Grab, GrabHovered colors, MinSize |
| `SetHeaderStyle` | Header, HeaderHovered, HeaderActive colors |
| `SetScrollStyle` | Scrollbar colors, size, rounding |
| `SetTextStyle` | Text, TextDisabled, TextSelectedBg colors |
| `SetWindowStyle` | Background, TitleBg, Padding, Rounding, MinSize |
| `SetPopupStyle` | Background, Rounding |
| `SetTabStyle` | Tab colors and rounding |
| `SetTableStyle` | Table colors and dimensions |
| `SetSeparatorStyle` | Separator colors |
| `SetSliderStyle` | SliderGrab, SliderGrabActive |
| `SetSelectableStyle` | Selectable colors |
| `SetBorderStyle` | Border colors and size |
| `SetSpacingStyle` | ItemSpacing, ItemInnerSpacing |
| `SetIndentStyle` | IndentSpacing |
| `SetPlotStyle` | Plot line/histogram colors |
| `SetDrawList` | Active draw list (Foreground/Background/Window/AtCursor) |

### Application Points

1. **Global:** Via `Style` pin on `ToSkiaLayer` / `ImGuiRenderer`
2. **Per-widget:** Via `Style` pin on any Widget
3. **Per-container:** Style on Window/TreeNode applies to all children

---

## 8. Primitive Drawing

All primitives inherit from `PrimitiveWidget` (HasItemState = false, so they don't affect item queries):

| Primitive | Parameters |
|-----------|-----------|
| `Line` | Start, end, color, thickness |
| `Rect` | Min, max, color, rounding, thickness, filled |
| `RectMulticolor` | Min, max, 4 corner colors |
| `Circle` | Center, radius, color, segments, thickness |
| `Ellipse` | Center, radius, rotation, color, segments |
| `Triangle` | 3 vertices, color, thickness |
| `Quad` | 4 vertices, color, thickness |
| `NGon` | Center, radius, color, segment count |
| `BezierCubic` | 4 control points, color, thickness |
| `BezierQuadratic` | 3 control points, color, thickness |
| `TextPrimitive` | Position, color, text |

### Draw List Modes

| Mode | Description |
|------|-------------|
| `Foreground` | Overlay on top of everything |
| `Background` | Behind all windows |
| `Window` | Current window's draw list (origin = window top-left) |
| `AtCursor` | Current window's draw list (origin = cursor position) |

---

## 9. State Management and Item Queries

### The Item State Problem

ImGui only tracks state for the "last item." Container widgets (TreeNode, CollapsingHeader) use `context.CaptureItemState()` to preserve their item state across the subtree rendering. The captured state is restored after children render, so query widgets placed after the container read the container's state.

### Query Widgets (HasItemState = false)

**Item state:** `IsItemActivated`, `IsItemActive`, `IsItemClicked`, `IsItemDeactivated`, `IsItemDeactivatedAfterEdit`, `IsItemEdited`, `IsItemFocused`, `IsItemHovered`, `IsItemToggledOpen`, `IsItemVisible`

**Global:** `IsAnyItemActive`, `IsAnyItemFocused`, `IsAnyItemHovered`, `IsPopupOpen`

**Geometry:** `GetCursorPos`, `GetContentRegionAvail`, `GetItemRectMin/Max/Size`, `CalcTextSize`

**Style:** `GetStyle`, `GetFontSize`, `GetFrameHeight`

**Mouse:** `GetMousePos`, `GetMouseDragDelta`, `IsMouseClicked`, `IsMouseDragging`, `IsMouseHoveringRect`

**Keyboard:** `IsKeyDown`, `IsKeyPressed`, `IsKeyReleased`

**Table:** `TableGetColumnCount`, `TableGetColumnFlags`, `TableGetSortSpecs`

---

## 10. Skia Backend

### ToSkiaLayer — Entry Point

Implements `ILayer` (Skia's rendering protocol):

**Update method (called each frame):**
1. Makes context current
2. Rebuilds font atlas if fonts changed (Alpha8 SKImage with pixel shader)
3. Sets DisplaySize, toggles DockingEnable
4. Applies global style (before and during frame)
5. `NewFrame()` → render widget tree → `ImGui.Render()`

**Render method (called by Skia pipeline):**
1. Extracts vertex data into pooled arrays (`ArrayPool`)
2. Swaps R/B channels (ImGui=RGBA, Skia=BGRA)
3. For each draw command: sets scissor, draws vertices via native P/Invoke to `libSkiaSharp`
4. User callbacks render embedded Skia layers

**Input:** Translates vvvv's `INotification` (mouse, keyboard, touch) into ImGui's IO system.

### SkiaWidget — Embedding Skia Content

Embed arbitrary Skia `ILayer` content inside ImGui panels:
- Creates InvisibleButton for hit-testing
- Registers layer with context, adds draw callback
- `EventFilter` controls input passthrough (BlockAll/ItemHasFocus/WindowHasFocus/AllowAll)

---

## 11. Stride Backend

### ImGuiRenderer — Entry Point

Extends Stride's `RendererBase`:

**DrawCore (called by Stride render pipeline):**
1. Makes context current, handles input subscription
2. Sets display size from render target
3. Applies global style, `NewFrame()` → widget tree → `ImGui.Render()`
4. GPU rendering via `StrideDeviceContext.RenderDrawLists`

### StrideDeviceContext

Manages complete GPU pipeline:
- Custom `ImGuiEffect` SDSL shader
- Vertex layout: Position<Vector2>, TextureCoordinate<Vector2>, Color(R8G8B8A8)
- Pipeline state: NonPremultiplied blend, no culling, scissor test, triangle list
- Font atlas as GPU Texture2D
- Color space handling (sRGB/linear)

### Embedding in Stride

- `SkiaWidget` — embed Skia 2D layers (via `SkiaRendererWithOffset`)
- `RenderWidget` — embed Stride render targets inside ImGui panels
- `TextureWidget` — display Stride textures as images

---

## 12. Backend Comparison

| Aspect | VL.ImGui.Skia | VL.ImGui.Stride |
|--------|--------------|----------------|
| **Entry point** | `ToSkiaLayer` (ILayer) | `ImGuiRenderer` (RendererBase) |
| **Font atlas** | Alpha8 SKImage | GPU Texture2D |
| **Rendering** | Native P/Invoke vertex drawing | Custom SDSL shader pipeline |
| **Color** | Manual RGBA→BGRA swizzle | Shader-based sRGB/linear |
| **Input** | INotification from Skia events | Stride InputManager |
| **Embedding** | Skia ILayer | Skia layers + Stride render targets |
| **Multi-window** | Not supported | Supported via ImGuiWindows |
| **Use case** | 2D apps, overlays, prototyping | 3D apps, game UIs, viewports |

Both backends share the identical core widget library. Switching requires only changing the rendering entry point.

---

## 13. Font Management

- `Fonts` pin accepts `Spread<FontConfig?>`
- Font atlas rebuilt when fonts change
- DPI scaling applied to atlas (`DIPHelpers.DIPFactor()`)
- UI scaling applied separately
- `GlyphRange` class for Unicode range configuration (Latin, CJK, etc.)

---

## 14. Coordinate System

VL.ImGui uses "hecto" scaling: 1 VL unit = 100 ImGui pixels. Conversion methods `FromHectoToImGui()` and `ToVLHecto()` handle the transformation. This ensures consistent coordinates regardless of DPI.

---

## 15. Commands

### Layout Commands
`SameLine`, `NewLine`, `Spacing`, `Indent`, `Unindent`, `Group`, `SetCursorPosition`, `SetItemWidth`

### Scroll Commands
`SetScrollX/Y`, `SetScrollHereX/Y`, `SetScrollFromPosX/Y`

### Window Commands
`SetNextWindowPosition/Size/ContentSize/SizeConstraints/Scroll/BgAlpha/Focus`

### Item Commands
`SetNextItemOpen/Width/AllowOverlap`, `SetItemDefaultFocus`, `SetKeyboardFocusHere`

### Table Commands
`TableNextColumn/Row`, `TableSetColumnIndex`, `TableHeader/HeadersRow`, `TableSetupColumn`, `TableSetBgColor`

### Popup Commands
`OpenPopup`, `CloseCurrentPopup`

### INI Commands
`LoadIniSettingsFromDisk/Memory`, `SaveIniSettingsToDisk/Memory`

Note: Automatic ini persistence is disabled by default (`IniFilename = null`).

---

## Dependencies

| Dependency | Purpose |
|------------|---------|
| ImGuiNET | C# binding of Dear ImGui |
| Stride.Core.Mathematics | Vector/color types |
| VL.Core | NodeContext, AppHost, IChannel, services |
| VL.CoreLib | Standard types |
| SkiaSharp (Skia backend) | 2D rendering |
| VL.Core.Skia (Skia backend) | ILayer interface |
| Stride (Stride backend) | 3D rendering pipeline |
