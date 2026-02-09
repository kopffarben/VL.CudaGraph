# VL File Best Practices

## Layout, Patching, and Node Organization Guide for Coding Agents

This document describes best practices for generating well-structured `.vl` files, based on analysis of 769+ production `.vl` files in the VL.StandardLibs repository. Following these conventions ensures generated patches look professional and match the patterns vvvv users expect.

---

## Table of Contents

1. [Data Flow Direction](#1-data-flow-direction)
2. [Coordinate System and Grid](#2-coordinate-system-and-grid)
3. [Node Positioning and Spacing](#3-node-positioning-and-spacing)
4. [Pad (IOBox) Placement](#4-pad-iobox-placement)
5. [Link Routing](#5-link-routing)
6. [Comment and Title Placement](#6-comment-and-title-placement)
7. [Canvas Organization](#7-canvas-organization)
8. [Process Definition Patterns](#8-process-definition-patterns)
9. [Multi-Section Layouts](#9-multi-section-layouts)
10. [Region Layout](#10-region-layout)
11. [Typical Element Sizes](#11-typical-element-sizes)
12. [Package File Organization](#12-package-file-organization)
13. [Naming Conventions](#13-naming-conventions)
14. [Complete Layout Recipe](#14-complete-layout-recipe)

---

## 1. Data Flow Direction

**Data always flows top-to-bottom in VL.** This is the single most important layout rule.

- **Inputs** are at the **top** of a node
- **Outputs** are at the **bottom** of a node
- **Input Pads** (IOBoxes providing values) are placed **above** the nodes they feed
- **Output Pads** (IOBoxes displaying results) are placed **below** the nodes they receive from
- **Links** go from smaller Y (output) to larger Y (input) — top to bottom
- **Feedback links** (`IsFeedback="true"`) are the only exception — they flow bottom-to-top

```
  [Input Pad]        y = 180
       |
       v
  [Processing Node]  y = 300
       |
       v
  [Output Pad]       y = 400
```

### Verified from real files (170+ links across 14 help files):
- **76%** of all links flow top-to-bottom (source Y < sink Y)
- **11%** flow bottom-to-top (feedback/state connections)
- **13%** connect at same Y level (horizontal state flow)
- Mean Y-delta: +62 px (positive = downward)
- Median Y-delta: +41 px
- Link `Ids="sourceId,sinkId"` — source (output, top) is listed first

---

## 2. Coordinate System and Grid

### Coordinate System
- Origin `(0,0)` is at the **top-left** of the canvas
- **X increases to the right**
- **Y increases downward**
- All coordinates are in **pixels** (floating-point values)

### Grid Snapping

**There is NO grid snapping in VL.** Statistical analysis of all coordinates across 14 help files shows no significant alignment to any grid (5, 10, 13, 15, 20, 25, 50, or 100). Coordinates are arbitrary integers resulting from free-form manual placement.

| Grid Size | Alignment % | Expected Random % |
|-----------|------------|-------------------|
| 5 | ~16% | 20% |
| 10 | ~8% | 10% |
| 25 | ~2% | 4% |

No grid pattern exceeds random chance.

### Recommended Practice for Code Generation
When generating files programmatically, use **clean coordinate values** for readability:
- Use round numbers (multiples of 5 or 10) for clarity, even though VL doesn't require it
- Keep consistent spacing between elements
- Align related elements on the same X or Y axis
- Content area typically spans X:100-700, Y:120-600 in help files

---

## 3. Node Positioning and Spacing

### Vertical Spacing Between Connected Elements

| Connection Type | Recommended | Median | Observed Range |
|----------------|------------|--------|----------------|
| Input Pad → Node | **50 px** | 47 px | 15-80 px |
| Node → Node (chain) | **40 px** | 41 px | 19-50 px |
| Node → Output Pad | **65 px** | 66 px | 48-80 px |
| Comment → First element | **100 px** | ~80 px | 40-150 px |

*Statistics from 170+ links across 14 analyzed help files.*

### Horizontal Alignment

Connected elements should be **roughly X-aligned**:
- An input pad feeding a node should have a similar X coordinate
- A chain of nodes should maintain consistent X positioning
- Small X offsets (5-20 px) are acceptable for visual clarity

### Example: Simple Node Chain (Top to Bottom)

```
Input Pad A    Bounds="200,160,80,20"     y=160
Input Pad B    Bounds="200,220,80,20"     y=220
Add Node       Bounds="200,300,65,19"     y=300
Output Pad     Bounds="200,370,80,20"     y=370
```

### Staircase Pattern for Multiple Inputs

When a node has multiple inputs, input pads are often arranged in a **staircase pattern** — each successive input pad is offset both horizontally and vertically:

```
Pad "Red"      Bounds="112,187,35,15"
Pad "Green"    Bounds="139,216,35,15"    offset: +27x, +29y
Pad "Blue"     Bounds="165,243,35,15"    offset: +26x, +27y
Pad "Alpha"    Bounds="192,270,35,15"    offset: +27x, +27y
     ↓ ↓ ↓ ↓
Node "RGBA Join"  Bounds="110,305,73,26"
```

This creates a diagonal arrangement that clearly shows which pad connects to which input, keeping links uncrossed.

### X-Alignment Precision

Input pads align to their connected node's X-position within **1-4 pixels** consistently. This effectively creates vertical alignment between pad and node:

```
Pad position:    x=75     Pad position:    x=591
Node position:   x=74     Node position:   x=589
X-delta:         1        X-delta:         2
```

Output pads are consistently offset by **+2 pixels** to the right of their source node:

```
CosineWave:    x=106  ->  Output slider: x=108  (delta +2)
SineWave:      x=212  ->  Output slider: x=214  (delta +2)
SawtoothWave:  x=303  ->  Output slider: x=305  (delta +2)
```

---

## 4. Pad (IOBox) Placement

### Input Pads
- Place **above** the node they connect to
- Roughly aligned with the corresponding input pin's X position
- Typical Y-gap: **50-80 px** above the target node

### Output Pads
- Place **below** the node they receive from
- Aligned with the corresponding output pin's X position
- Typical Y-gap: **60-70 px** below the source node

### State Pads (connected to Slots)
- Use `SlotId` attribute to reference a Slot
- Typically placed near the node that uses the state
- Use 2-value Bounds: `Bounds="X,Y"` (position only)

### Multiple Output Pads
When a node has multiple outputs, spread them horizontally:

```
          [Node]              y=298
         /      \
   [Output A]  [Output B]    y=367
   x=150       x=255
```

Typical X-gap between side-by-side output pads: **80-120 px**

---

## 5. Link Routing

### Basic Rules
- Links flow **top-to-bottom** (source Y < sink Y)
- Links are **straight lines** — VL draws them automatically
- **Avoid crossing links** by positioning pads appropriately
- Use the staircase pattern (Section 3) to prevent input link crossings

### Multi-Hop Links (through ControlPoints)
Links can pass through intermediate ControlPoints for routing:
```xml
<Link Ids="sourceId,controlPointId,sinkId" />
```
This creates a link that bends through the ControlPoint position.

### Feedback Links
For loops (where output feeds back to input):
```xml
<Link Id="..." Ids="outputId,inputId" IsFeedback="true" />
```
Feedback links are drawn differently (flowing bottom-to-top).

### Hidden Links
For "reference" connections that shouldn't be drawn:
```xml
<Link Id="..." Ids="sourceId,sinkId" IsHidden="true" />
```

---

## 6. Comment and Title Placement

### Help File Comment Structure
Help files follow a consistent comment hierarchy:

| Element | Y Position | Font Size | Type |
|---------|-----------|-----------|------|
| **Title** (main instruction) | ~100-120 | 14 or 20 | Comment |
| **Description** (explanation) | ~140-190 | 9 | Comment |
| **Inline annotations** | Same Y as element | 9 | Comment with `< ` prefix |

### Title Comment

```xml
<Pad Id="..." Bounds="119,100,400,25" ShowValueBox="true" isIOBox="true"
     Value="HowTo Do Something Useful">
  <p:TypeAnnotation>
    <Choice Kind="TypeFlag" Name="String" />
  </p:TypeAnnotation>
  <p:ValueBoxSettings>
    <p:fontsize p:Type="Int32">14</p:fontsize>
    <p:stringtype p:Assembly="VL.Core" p:Type="VL.Core.StringType">Comment</p:stringtype>
  </p:ValueBoxSettings>
</Pad>
```

### Description Comment

```xml
<Pad Id="..." Bounds="119,145,350,40" ShowValueBox="true" isIOBox="true"
     Value="This example shows how to...">
  <p:TypeAnnotation>
    <Choice Kind="TypeFlag" Name="String" />
  </p:TypeAnnotation>
  <p:ValueBoxSettings>
    <p:fontsize p:Type="Int32">9</p:fontsize>
    <p:stringtype p:Assembly="VL.Core" p:Type="VL.Core.StringType">Comment</p:stringtype>
  </p:ValueBoxSettings>
</Pad>
```

### Inline Annotation (next to a node)
Place to the right of or near the element being annotated, at the same Y level.
Use `< ` prefix in the Value for "arrow" style annotations:
```xml
Value="&lt; this is important"
```

---

## 7. Canvas Organization

### Top-Level Canvas (Document Root)
```xml
<Canvas Id="..." DefaultCategory="Main" BordersChecked="false" CanvasType="FullCategory" />
```
- `CanvasType="FullCategory"` — always for the root canvas
- `DefaultCategory` — sets the base namespace (e.g., `"Main"`, `"Graphics.Skia"`)
- `BordersChecked="false"` — commonly set on root canvas

### Inner Canvas (Inside Application/Process)
```xml
<Canvas Id="..." CanvasType="Group" />
```
- `CanvasType="Group"` — for all inner canvases

### Sub-Category Canvas (in package files)
```xml
<Canvas Id="..." Name="SubCategory" Position="342,460" />
```
- Named canvases create category hierarchy levels
- Use `Position` to place them in the parent canvas
- Nesting creates dotted category paths: `Root.SubCategory.DeepSub`

### Category Hierarchy Example
```xml
<Canvas DefaultCategory="Graphics.Skia" CanvasType="FullCategory">
  <Canvas Name="Drawing" Position="100,100">         <!-- Graphics.Skia.Drawing -->
    <Canvas Name="Primitives" Position="200,200">     <!-- Graphics.Skia.Drawing.Primitives -->
      <Node Name="Circle" ... />
    </Canvas>
  </Canvas>
  <Canvas Name="Internal" Position="400,100">         <!-- Graphics.Skia.Internal -->
    <Node Name="Helper" ... />
  </Canvas>
</Canvas>
```

---

## 8. Process Definition Patterns

### Standard Process (Create + Update)
Every Application and most Process type definitions follow this pattern:

```xml
<Node Name="Application" Bounds="100,100" Id="...">
  <p:NodeReference>
    <Choice Kind="ContainerDefinition" Name="Process" />
    <FullNameCategoryReference ID="Primitive" />
  </p:NodeReference>
  <Patch Id="...">
    <Canvas Id="..." CanvasType="Group">
      <!-- Visual content here -->
    </Canvas>
    <Patch Id="createId" Name="Create" />
    <Patch Id="updateId" Name="Update" />
    <ProcessDefinition Id="...">
      <Fragment Id="..." Patch="createId" Enabled="true" />
      <Fragment Id="..." Patch="updateId" Enabled="true" />
    </ProcessDefinition>
  </Patch>
</Node>
```

### Process with Additional Fragments
Some processes have extra operations:
- `Name="Dispose"` — cleanup logic
- `Name="Notify"` — notification handler
- `Name="Render"` — rendering logic (Skia/Stride)
- `Name="Split"` — decomposition operation

### Application Node Position
The Application node is almost always placed at:
```
Bounds="100,100"
```
This is the standard convention for the top-level Application node.

### ProcessDefinition Placement
The `<ProcessDefinition>` element is placed as a child of the node's inner `<Patch>`, after the named operation patches.

### Forward Type Definitions
For forwarding .NET types:
```xml
<ProcessDefinition Id="..." IsHidden="true" />
```
`IsHidden="true"` prevents the forward type from appearing as a constructable process in the node browser.

---

## 9. Multi-Section Layouts

When demonstrating multiple related concepts or alternatives in a single patch:

### Horizontal Section Layout
Sections are arranged **left-to-right** with significant horizontal gaps:

```
Section 1          Section 2          Section 3
x: 100-400        x: 550-850        x: 1000-1300
                   (gap: ~150-400)
```

| Measurement | Recommended | Observed Range |
|-------------|------------|----------------|
| Gap between sections | **350-400 px** | 200-460 px |
| Gap between parallel node columns | **150 px** | 140-165 px |

### Each Section Has Its Own:
- Title comment (at same Y level across all sections)
- Input pads
- Processing nodes
- Output pads

### Critical Rule: Y-Alignment Across Sections

Verified from production files — parallel sections align precisely:

| Element | Section 1 Y | Section 2 Y | Delta |
|---------|------------|------------|-------|
| Title | 120 | 121 | 1 |
| Description | 155 | 156 | 1 |
| Processing node | 298 | 298 | **0** |
| Output pads | 367 | 367 | **0** |

**Always align corresponding elements across sections to the same Y coordinate.**

### Example Layout
```
y=100:  [Title A]                   [Title B]
y=155:  [Description A]            [Description B]
y=200:  [Input Pads A]             [Input Pads B]
y=300:  [Node A]                   [Node B]
y=370:  [Output A]                 [Output B]

        x: 100-350                 x: 550-800
```

### Fan-Out Pattern

When one output feeds multiple parallel nodes (e.g., LFO -> 5 wave shapers, or 7 primitives -> Group):

1. All destination nodes share **identical Y** (same horizontal band)
2. Destination nodes are spread horizontally at **~100-165 px intervals**
3. Each destination's output pad is X-aligned within +2 px of its node
4. The gap between the right edge of one node and left edge of the next: **~40-55 px**

```
                    [Source Node]              y=569
                   /  |   |   |   \
    [Node A] [Node B] [Node C] [Node D] [Node E]   y=638
      |        |        |        |        |
    [Out A]  [Out B]  [Out C]  [Out D]  [Out E]    y=705

    x=106    x=212   x=303    x=419    x=530
```

---

## 10. Region Layout

### Region Bounds
Regions use 4-value Bounds with explicit width and height:
```xml
<Node Bounds="X,Y,Width,Height" ...>
```

### Typical Region Sizes

| Region Type | Typical Size | Minimum |
|-------------|-------------|---------|
| If region | 300-530 x 200-530 | 200 x 150 |
| ForEach region | 190-400 x 120-300 | 150 x 100 |
| Cache region | 200-350 x 150-250 | 150 x 100 |

### ControlPoint Placement
- **Top ControlPoints** (`Alignment="Top"`): Placed at the top border of the region, same Y as the region's top
- **Bottom ControlPoints** (`Alignment="Bottom"`): Placed at the bottom border, Y = region.Y + region.Height

```
ControlPoint (Top)     Bounds="150,200"  Alignment="Top"
┌─────────────────────┐ Region Bounds="100,200,400,300"
│                     │
│  [Content Nodes]    │
│                     │
└─────────────────────┘
ControlPoint (Bottom)  Bounds="150,500"  Alignment="Bottom"
```

### Content Inside Regions
- Place content nodes inside the region's Bounds area
- Leave ~20-30 px padding from region borders
- Content nodes use the same top-to-bottom flow

---

## 11. Typical Element Sizes

### Node Sizes (Width x Height)

| Node Type | Width | Height | Example Bounds |
|-----------|-------|--------|----------------|
| Simple operation (+, -, *) | 22-25 | 19 | `"200,300,25,19"` |
| Standard CoreLib node | 45-85 | 19 | `"200,300,65,19"` |
| Long-named node | 85-165 | 19 | `"200,300,145,19"` |
| Join/Split/stateful ops | 41-73 | 26 | `"200,300,52,26"` |
| Skia primitive (Circle, Rect) | 105-145 | **13** | `"200,300,105,13"` |
| Skia Renderer | 105 | **13** | `"200,300,105,13"` |
| Skia Group | 473 | **20** | `"200,300,473,20"` |
| Stride Renderer | 105-165 | 19 | `"200,300,165,19"` |

**Key**: Standard node height is **19 px** (~80% of all nodes). Nodes with state pins (Join/Split) use **26 px** (~10%). Region nodes use 76-120 px height.

Node width statistics: min=25 ("+"), max=349 (Group), mean=82, median=80. Most common widths: 85, 105, 45, 65.

### Pad Sizes (Width x Height)

| Pad Type | Width | Height | Example |
|----------|-------|--------|---------|
| Float32 | 33-35 | 15-19 | `"200,160,35,15"` |
| Int32 | 18-35 | 15 | `"200,160,35,15"` |
| Boolean (toggle) | 35-45 | 35-43 | `"200,160,35,35"` |
| Boolean (bang) | 35-45 | 35-43 | `"200,160,35,35"` |
| String (value) | 64-273 | 15-20 | `"200,160,170,20"` |
| String (comment, font=14) | 139-632 | 25-39 | `"100,100,400,25"` |
| String (comment, font=9) | 95-472 | 19-312 | `"100,140,350,40"` |
| Color (RGBA) | 52-143 | 15-20 | `"200,160,136,15"` |
| Vector2 | 35-45 | 27-38 | `"200,160,38,38"` |
| Enum | 60-93 | 19 | `"200,160,66,19"` |
| Spread display | 35-315 | 47-233 | `"200,160,144,126"` |

### Nodes vs Pads: Position-Only vs Position+Size
- **Top-level definition nodes** (Application, type definitions): Use **2-value Bounds** `"X,Y"` (position only)
- **Processing nodes** (operation calls): Use **4-value Bounds** `"X,Y,W,H"` (position + size)
- **Pads**: Use **4-value Bounds** `"X,Y,W,H"` (position + size)
- **ControlPoints**: Use **2-value Bounds** `"X,Y"` (position only)

---

## 12. Package File Organization

### Aggregator Documents
Package root files (like `VL.CoreLib.vl`) serve as facades:
- Minimal content (just Application node + Canvas)
- Many `DocumentDependency IsForward="true"` entries
- Re-export types from sub-documents

```xml
<Document ...>
  <DocumentDependency Id="..." Location="./VL.Animation.vl" IsForward="true" />
  <DocumentDependency Id="..." Location="./VL.Reactive.vl" IsForward="true" />
  <DocumentDependency Id="..." Location="./VL.Collections.vl" IsForward="true" />
  <Patch Id="...">
    <Canvas Id="..." CanvasType="FullCategory" />
    <Node Name="Application" Bounds="100,100" ...>
      <!-- Empty Application boilerplate -->
    </Node>
  </Patch>
</Document>
```

### Type Definition Files
Files containing type definitions organize them via Canvas hierarchy:

```
Canvas (FullCategory, DefaultCategory="Domain")
├── Canvas (Name="SubCategory1", Position="100,100")
│   ├── Node (ForwardDefinition "TypeA")
│   └── Node (ForwardDefinition "TypeB")
├── Canvas (Name="SubCategory2", Position="300,100")
│   └── Node (ContainerDefinition "ProcessC")
└── Node (OperationDefinition "HelperOp")
```

### Forward Definition Placement
Forward definitions in package files:
- Placed inside their category Canvas
- Use 2-value Bounds (position only): `Bounds="367,253"`
- Always include `<p:ForwardAllNodesOfTypeDefinition p:Type="Boolean">true</p:ForwardAllNodesOfTypeDefinition>`
- `<ProcessDefinition Id="..." IsHidden="true" />` inside their Patch

### Dependency Ordering
Dependencies can appear before or after `<Patch>` in the `<Document>`:
- No strict ordering requirement
- Common pattern: NugetDependencies first, then Patch, then PlatformDependencies
- `IsForward="true"` on dependencies that should be re-exported to consumers
- `IsFriend="true"` for internal/friend access (less common)

---

## 13. Naming Conventions

### Node Names
| Context | Convention | Examples |
|---------|-----------|----------|
| Process | PascalCase noun | `OrbitCamera`, `CollapsingHeader`, `Sequencer` |
| Operation | PascalCase verb/action | `CircleContainsPoint`, `CreateDefault` |
| Overload | Parenthetical suffix | `Create (KeyComparer)`, `FromSequence (ElementSelector)` |
| Forward type | Match .NET name | `LabelAttribute`, `VertexDeclaration` |
| Internal | `(Internal)` suffix or `IsHidden="true"` | `Helper (Internal)` |

### Pin Names
- PascalCase with spaces: `"Near Plane"`, `"Key Comparer"`, `"Remote End Point"`
- State pins: `"Input (this)"` / `"Output (this)"`
- Boolean defaults: `DefaultValue="True"` or `DefaultValue="False"`

### Category Names
- Root categories use dotted paths: `"Graphics.Skia"`, `"IO"`, `"Animation"`
- Sub-categories use plain names: `"Internal"`, `"Advanced"`, `"API"`
- `"Primitive"` is a special root category for type definitions

### Patch Names
Standard names:
| Name | Purpose |
|------|---------|
| `Create` | Constructor / initialization |
| `Update` | Per-frame execution |
| `Dispose` | Cleanup on destruction |
| `Then` | If-region true branch |
| `Else` | If-region false branch |
| `Split` | Decomposition operation |
| `Render` | Visual rendering (Skia/Stride) |
| `Notify` | Event notification handler |

---

## 14. Complete Layout Recipe

### Help File Template

Follow this recipe when generating a help/example `.vl` file:

#### Step 1: Document Structure
```
Document
├── NugetDependency (VL.CoreLib)
└── Patch
    ├── Canvas (DefaultCategory="Main", CanvasType="FullCategory", BordersChecked="false")
    └── Node "Application" (Bounds="100,100")
        └── Patch
            ├── Canvas (CanvasType="Group")
            │   └── [Visual content — see steps below]
            ├── Patch "Create"
            ├── Patch "Update"
            └── ProcessDefinition with Fragments
```

#### Step 2: Content Canvas Layout (inside the Group Canvas)

Use these Y-position guidelines:

```
y = 100    [Title Comment]         font=14, stringtype=Comment
y = 145    [Description Comment]   font=9, stringtype=Comment
y = 200    [Input Pad 1]           isIOBox=true
y = 230    [Input Pad 2]           isIOBox=true (offset +30y from previous)
y = 300    [Processing Node]       Main operation
y = 350    [Processing Node 2]     Chained operation (if needed)
y = 420    [Output Pad]            Display result
y = 800    [Renderer Node]         (if visual output, e.g., Skia Renderer)
```

#### Step 3: Connecting Elements
1. Create Links from input Pads to processing Node input Pins
2. Create Links between chained Nodes
3. Create Links from Node output Pins to output Pads
4. Always: `Ids="sourceId,sinkId"` (output first)

#### Step 4: Spacing Checklist
- [ ] Input pads are 50-80 px above their target node
- [ ] Sequential nodes are 40-50 px apart vertically
- [ ] Output pads are 60-70 px below their source node
- [ ] Connected elements are roughly X-aligned
- [ ] Multi-input pads use staircase pattern (avoid crossing links)
- [ ] Multiple sections separated by 350-400 px horizontally
- [ ] Title and description at top of canvas (y=100-190)
- [ ] Renderer node at bottom (y=800+)

### Package File Template

Follow this recipe for library definition files:

```
Document
├── NugetDependency (VL.CoreLib, version)
├── PlatformDependency (referenced .dll files)
├── DocumentDependency (referenced .vl files, IsForward="true")
└── Patch
    ├── Canvas (DefaultCategory="MyLib", CanvasType="FullCategory")
    │   ├── Canvas (Name="SubCategory", Position="X,Y")
    │   │   ├── Node (ForwardDefinition/RecordDefinition/etc.)
    │   │   └── ...
    │   └── Canvas (Name="AnotherCategory", Position="X,Y")
    │       └── ...
    └── Node "Application" (Bounds="100,100")
        └── Patch (empty boilerplate with Create/Update/ProcessDefinition)
```

---

## Quick Reference: Layout Dimensions

| Metric | Value |
|--------|-------|
| Application node position | `Bounds="100,100"` |
| Title comment Y | ~100-120 |
| Description comment Y | ~140-190 |
| First input pad Y | ~200-250 |
| Main processing area Y | ~280-400 |
| Output display area Y | ~370-500 |
| Renderer Y | ~800-900 |
| Vertical pad-to-node gap | 50-80 px |
| Vertical node-to-node gap | 40-50 px |
| Vertical node-to-output gap | 60-70 px |
| Horizontal section gap | 350-400 px |
| Standard node height | 19 px |
| Standard pad height (value) | 15-20 px |
| Boolean pad size | 35x35 px |
| X-alignment precision (pad to node) | 1-4 px |
| Output pad X-offset from node | +2 px |
| Fan-out horizontal spacing | 100-165 px |
| Fan-out node-to-node edge gap | 40-55 px |

---

## Appendix A: Annotated Real-World Example

This is the spatial layout of `HowTo Detect If a Value is Changed.vl`, annotated with all measurements:

```
SECTION 1 (x: 115-290)               SECTION 2 (x: 581-754)
========================               ========================

y=120  "Use a Changed node!"          y=121  "Use a SequenceChanged node!"
       (font 14, Comment)                    (font 14, Comment)

y=155  "If the value is just           y=156  "If the value is a sequence."
       a single item." (font 9)               (font 9)

y=250  [LFO] x=134,45x19              y=212  [LFO] x=625,45x19
                                              |  (30 px gap)
                                       y=242  [RandomSpread] x=612,85x19
       | (48 px gap)                          |  (56 px gap)
y=298  [Changed] x=174,57x19          y=298  [SequenceChanged] x=612,99x19
       /         \                            /         \
       | (69 px)  | (69 px)                  | (69 px)  | (69 px)
y=367  [Changed]  [Unchanged]          y=367  [Changed]  [Unchanged]
       x=150      x=255                       x=604      x=719
       35x35      35x35                       35x35      35x35

<--- 466 px horizontal gap --->
```

### Key Observations from This Example:
1. **Titles at same Y** (120 vs 121 = 1px difference)
2. **Descriptions at same Y** (155 vs 156 = 1px difference)
3. **Main processing nodes at exact same Y** (both at 298)
4. **Output pads at exact same Y** (both at 367)
5. **Equal vertical gap from node to outputs** (both 69 px)
6. **Output pads spread left/right of node** (Changed pad ~24px left, Unchanged pad ~81px right)

---

## Appendix B: Vertical Pipeline Template (Skia)

For Skia visual patches with renderer:

```
y=120    [Title]           font=20, Comment
y=164    [Description]     font=9, Comment
y=250    [Position pads]   Vector2, row 1 inputs
y=300    [Size pads]       Vector2/Float32, row 2 inputs
y=345    [Extra params]    Float32, row 3 inputs
y=399    [Primitive nodes] ALL at same Y (horizontal band, height=13)
y=478    [Group node]      Collects all primitives (height=20)
y=528    [Color pad]       RGBA input for renderer
y=558    [Renderer]        Skia Renderer (height=13)
```

Gaps:
- Position pad to primitive: **~148 px** (long links, multiple input rows)
- Primitive to Group: **~79 px**
- Group to Renderer: **~80 px**
- Color pad to Renderer: **~30 px**
