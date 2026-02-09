# VL File Format Description (.vl)

## Comprehensive Guide for Coding Agents

This document describes the `.vl` file format used by vvvv gamma (VL), a visual programming environment. A `.vl` file is an XML document that defines a visual dataflow program. This guide provides all information needed to programmatically generate valid `.vl` files.

---

## Table of Contents

1. [Overview](#1-overview)
2. [XML Namespaces and Root Element](#2-xml-namespaces-and-root-element)
3. [ID System](#3-id-system)
4. [Element Hierarchy](#4-element-hierarchy)
5. [Document Element](#5-document-element)
6. [Dependencies](#6-dependencies)
7. [Patch Element](#7-patch-element)
8. [Canvas Element](#8-canvas-element)
9. [Node Element](#9-node-element)
10. [NodeReference System (Choices)](#10-nodereference-system-choices)
11. [Pin Element](#11-pin-element)
12. [Pad Element (IOBox)](#12-pad-element-iobox)
13. [Link Element](#13-link-element)
14. [ProcessDefinition and Fragments](#14-processdefinition-and-fragments)
15. [Slot Element](#15-slot-element)
16. [Type Definitions](#16-type-definitions)
17. [Region Nodes](#17-region-nodes)
18. [TypeAnnotation and TypeReference](#18-typeannotation-and-typereference)
19. [Property Serialization Format](#19-property-serialization-format)
20. [Complete Minimal Example](#20-complete-minimal-example)
21. [Complete Working Example](#21-complete-working-example)
22. [Common Patterns and Recipes](#22-common-patterns-and-recipes)
23. [Validation Rules and Gotchas](#23-validation-rules-and-gotchas)

---

## 1. Overview

A `.vl` file is an XML document that encodes a visual program. The key concepts are:

- **Document**: The root of a `.vl` file. Contains dependencies and a single top-level Patch.
- **Patch**: A container for visual elements (Nodes, Pads, Links, Canvases, etc.).
- **Node**: Represents either a node call (operation/process) or a type/operation definition.
- **Pin**: An input or output on a Node (defined at the definition site).
- **Pad**: A visual data element on the canvas (IOBox for displaying/editing values).
- **Link**: Connects two data endpoints (Pins or Pads) by referencing their IDs.
- **Canvas**: A visual grouping container; does not introduce a new scope.
- **ProcessDefinition**: Defines the lifecycle operations (Create, Update) of a process node.
- **Fragment**: Links a ProcessDefinition to its operation Patches.
- **Slot**: A state field within a type definition.
- **Dependencies**: References to NuGet packages, other `.vl` files, or .NET assemblies.

### Class Hierarchy (Serialization)

```
Element (base)
├── Compound (has Children)
│   ├── Document
│   ├── Patch
│   ├── Canvas
│   └── NodeOrPatch
│       └── Node
│           └── ProcessDefinition
├── DataHub (has TypeAnnotation, Comment)
│   ├── Pin
│   ├── Pad
│   └── ControlPoint
├── Link
├── Fragment
├── Slot
├── Overlay
└── Dependency (abstract)
    ├── NugetDependency
    ├── DocumentDependency
    ├── PlatformDependency
    ├── ProjectDependency
    └── NodeFactoryDependency
```

---

## 2. XML Namespaces and Root Element

Every `.vl` file starts with:

```xml
<?xml version="1.0" encoding="utf-8"?>
<Document xmlns:p="property" Id="..." LanguageVersion="..." Version="0.128">
  ...
</Document>
```

Or with reflection namespace (only needed when using `r:` attributes like `r:IsNull="true"`):

```xml
<?xml version="1.0" encoding="utf-8"?>
<Document xmlns:p="property" xmlns:r="reflection" Id="..." LanguageVersion="..." Version="0.128">
  ...
</Document>
```

### Namespaces

| Prefix | URI | Required | Purpose |
|--------|-----|----------|---------|
| `p` | `property` | **Yes** | Used for complex properties serialized as child elements (e.g., `<p:NodeReference>`) |
| `r` | `reflection` | **No** | Used for reflection-related attributes (e.g., `r:IsNull="true"`). Only declare if needed. |

### Root Element Attributes

| Attribute | Required | Description |
|-----------|----------|-------------|
| `Id` | Yes | Unique base62-encoded GUID for the document |
| `LanguageVersion` | Yes | Version string like `"2024.6.0-0147-g7216424c8e"` |
| `Version` | Yes | Always `"0.128"` (legacy compatibility field) |
| `Authors` | No | Comma-separated author list (can also be `<p:Authors>` child) |
| `LicenseUrl` | No | URL to license |
| `ProjectUrl` | No | URL to project |

**Important**: The `xmlns:p="property"` namespace declaration MUST be present on the `Document` element. The `xmlns:r="reflection"` namespace is only needed when the file uses `r:` prefixed attributes (e.g., `r:IsNull="true"` for explicit null values).

---

## 3. ID System

### SerializedId

Every element in a `.vl` file has a unique `Id` attribute. IDs are **base62-encoded GUIDs** — 22-character strings using `[0-9A-Za-z]`.

**Rules:**
- Every `Id` MUST be unique within the document.
- IDs persist across saves (they are "serialized IDs").
- When generating new elements, always create a new GUID and encode it to base62.
- The format is: `GUIDEncoders.GuidTobase62(Guid.NewGuid())` which produces a 22-character string.

**Examples of valid IDs:**
```
C2vqbtoStWoOI1eKIKZBBM
BFeHf229CNdQZYNuoWvSNw
GEsBUL7W1aFO1qSdiPjEaL
```

### Base62 Encoding

The base62 encoding uses the character set `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`. A .NET GUID (16 bytes) is converted to a base62 string of 22 characters.

### ID References in Links

Links reference the IDs of the pins/pads they connect. The `Ids` attribute on a `<Link>` element contains a comma-separated list of exactly 2 IDs: `source,sink` (output,input).

### ID References in Fragments

Fragment elements reference Patch IDs via the `Patch` attribute to associate operation patches with a ProcessDefinition.

---

## 4. Element Hierarchy

A typical `.vl` file follows this nesting structure:

```
Document
├── NugetDependency (0..n)
├── DocumentDependency (0..n)
├── PlatformDependency (0..n)
└── Patch (exactly 1, the top-level document patch)
    ├── Canvas (the main canvas with DefaultCategory)
    │   ├── Canvas (sub-canvases for categories)
    │   │   └── Node (type definitions, operation definitions, etc.)
    │   ├── Node (type definitions)
    │   └── ...
    └── Node (Name="Application", the implicit entry point)
        └── Patch (the Application's inner patch)
            ├── Canvas (CanvasType="Group")
            │   ├── Node (operation calls, regions)
            │   │   ├── p:NodeReference
            │   │   └── Pin (0..n)
            │   ├── Pad (IOBoxes)
            │   └── ...
            ├── Patch (Name="Create")
            ├── Patch (Name="Update")
            ├── ProcessDefinition
            │   ├── Fragment (references Create patch)
            │   └── Fragment (references Update patch)
            └── Link (0..n, connecting pins/pads)
```

**Key rule**: Dependencies are direct children of `Document`, NOT children of `Patch`. They appear before or after the `Patch` element.

---

## 5. Document Element

The `<Document>` element is the XML root.

### Serialized Properties (XML)

| Property | XML Form | Description |
|----------|----------|-------------|
| `Id` | Attribute | Unique document ID (base62 GUID) |
| `LanguageVersion` | Attribute | VL language version string |
| `Version` | Attribute | Always `"0.128"` |
| `FilePath` | `<p:FilePath>` | Only written for non-local serialization |
| `Summary` | `<p:Summary>` | Document description |
| `Authors` | `<p:Authors>` | Comma-separated author list |
| `Credits` | `<p:Credits>` | Third-party credits |
| `LicenseUrl` | `<p:LicenseUrl>` | URL to license |
| `ProjectUrl` | `<p:ProjectUrl>` | URL to project |

### Children

Direct children of `<Document>`:
- `<NugetDependency>` — NuGet package references
- `<DocumentDependency>` — References to other `.vl` files
- `<PlatformDependency>` — References to .NET assemblies
- `<Patch>` — Exactly one top-level patch

---

## 6. Dependencies

Dependencies are direct children of `<Document>`. There are several types:

### NugetDependency

References a NuGet package (most common for VL libraries).

```xml
<NugetDependency Id="Pdwi7eJLF93NDbynZh5tNB" Location="VL.CoreLib" Version="2024.6.0-0147-g7216424c8e" />
```

| Attribute | Required | Description |
|-----------|----------|-------------|
| `Id` | Yes | Unique ID |
| `Location` | Yes | NuGet package ID (e.g., `"VL.CoreLib"`) |
| `Version` | No | NuGet version string |

**Important**: Almost every document should have a dependency on `VL.CoreLib` as it provides the core types and nodes.

### DocumentDependency

References another `.vl` file directly.

```xml
<DocumentDependency Id="..." Location="./MyOtherFile.vl" />
```

| Attribute | Required | Description |
|-----------|----------|-------------|
| `Id` | Yes | Unique ID |
| `Location` | Yes | Relative or absolute path to the `.vl` file |

### PlatformDependency

References a .NET assembly directly.

```xml
<PlatformDependency Id="PJdphG5gv9wQPL1PuHvVN3" Location="VL.Core.dll" />
```

| Attribute | Required | Description |
|-----------|----------|-------------|
| `Id` | Yes | Unique ID |
| `Location` | Yes | Assembly file name or path |

### Common Dependency Attributes (all types)

| Attribute | Default | Description |
|-----------|---------|-------------|
| `IsForward` | `false` | If `true`, the dependency's types are forwarded to consumers |
| `IsFriend` | `false` | If `true`, grants friend access |

---

## 7. Patch Element

A `<Patch>` is a compound element that contains the visual elements of a dataflow graph.

```xml
<Patch Id="BFeHf229CNdQZYNuoWvSNw">
  <!-- children: Canvas, Node, Link, Pad, ProcessDefinition, etc. -->
</Patch>
```

### Attributes

| Attribute | Required | Description |
|-----------|----------|-------------|
| `Id` | Yes | Unique ID |
| `Name` | No | Name of the patch (e.g., `"Create"`, `"Update"`, `"Dispose"`, `"Then"`, `"Split"`) |
| `IsGeneric` | No | `"true"` if this is a generic patch |
| `SortPinsByX` | No | If `true`, pin ordering is determined by X position |
| `ManuallySortedPins` | No | If `"true"`, pins use manual order instead of auto-sorting |
| `ParticipatingElements` | No | Comma-separated IDs of elements that participate in this named patch |

### Named Patches (Operations)

Named patches define operations like `Create` and `Update`:

```xml
<Patch Id="TMFMkcZDcYUNKxGjByaXt0" Name="Create" />
<Patch Id="JXnEepVfp1ZPHJ6XEXlYmq" Name="Update" />
```

These are referenced by `Fragment` elements inside a `ProcessDefinition`.

---

## 8. Canvas Element

A `<Canvas>` provides visual grouping but does NOT introduce a new logical scope. All children are treated as if they belong to the parent patch.

```xml
<Canvas Id="OHBhKxype2ALATNlngbqoB" DefaultCategory="Main" BordersChecked="false" CanvasType="FullCategory" />
```

### Attributes

| Attribute | Required | Default | Description |
|-----------|----------|---------|-------------|
| `Id` | Yes | — | Unique ID |
| `Name` | No | — | Canvas name (for sub-canvases acting as categories) |
| `Position` | No | `"0,0"` | Position as `"X,Y"` |
| `DefaultCategory` | No | — | Full category name (e.g., `"Main"`, `"System"`) |
| `BordersChecked` | No | `"true"` | Whether border control points are checked |
| `CanvasType` | No | `"Category"` | One of: `Category`, `FullCategory`, `Group` |

### CanvasType Values

| Value | Description |
|-------|-------------|
| `Category` | Standard category canvas (default) |
| `FullCategory` | Full category canvas — used for the top-level document canvas |
| `Group` | Simple group — used inside type definitions for visual organization |

---

## 9. Node Element

A `<Node>` is the most versatile element. It can represent:
- **Operation calls** (calling a function/process)
- **Type definitions** (Class, Record, Interface, Forward, Process)
- **Operation definitions** (defining a new operation)
- **Regions** (If, ForEach, Cache, etc.)

### Basic Structure

```xml
<Node Name="MyNode" Bounds="100,200" Id="abc123def456ghijk7890l">
  <p:NodeReference>
    <Choice Kind="..." Name="..." />
    ...
  </p:NodeReference>
  <Pin ... />
  <Pin ... />
</Node>
```

### Attributes

| Attribute | Required | Description |
|-----------|----------|-------------|
| `Id` | Yes | Unique ID |
| `Name` | No* | Name of the node/definition (for definitions: the type/operation name) |
| `Bounds` | No | Position and size as `"X,Y"` or `"X,Y,W,H"` |
| `Category` | No | Category full name (e.g., `"Primitive"`) |
| `AutoConnect` | No | Default `false`. If `true`, auto-connect pins |
| `Aspects` | No | Symbol smell aspects (rarely used) |
| `StructureOfTypeDefinition` | No | For type defs: `Class`, `Record`, etc. |
| `HideCategory` | No | Boolean, hides category in node browser |
| `ForwardAllNodesOfTypeDefinition` | No | Boolean, forwards all nodes for Forward types |
| `HelpFocus` | No | Help priority: `High`, `Low`, etc. |
| `Summary` | No | Brief documentation text describing the node |
| `Remarks` | No | Extended documentation remarks |
| `Tags` | No | Comma-separated search keywords (e.g., `"hittest,picking"`) |

*`Name` is required for type/operation definitions.

### Bounds Format

Bounds can be specified as:
- `"X,Y"` — Position only (size auto-computed)
- `"X,Y,W,H"` — Position and explicit size

Values are floating-point numbers.

### Children of Node

| Child | Description |
|-------|-------------|
| `<p:NodeReference>` | Required. Defines what this node IS (see section 10) |
| `<Pin>` | Pin definitions (for definitions) or pin value overrides (for calls) |
| `<Patch>` | Inner patches (for definitions: Create, Update, operation body) |
| `<Canvas>` | Inner canvas (for region/definition content) |
| `<ProcessDefinition>` | Process lifecycle definition |
| `<Slot>` | State fields (for type definitions) |
| `<p:TypeAnnotation>` | Type annotation (for Forward types, type constraints) |
| `<p:HelpFocus>` | Help priority element |
| `<p:ForwardAllNodesOfTypeDefinition>` | Boolean property element |

---

## 10. NodeReference System (Choices)

The `<p:NodeReference>` property is the core mechanism that defines what a Node IS and what it references. It contains a list of "choices" that together identify the target symbol.

### Structure

```xml
<p:NodeReference LastCategoryFullName="Control" LastDependency="CoreLibBasics.vl">
  <Choice Kind="NodeFlag" Name="Node" Fixed="true" />
  <Choice Kind="ProcessAppFlag" Name="OnOpen" />
</p:NodeReference>
```

### NodeReference Attributes

| Attribute | Description |
|-----------|-------------|
| `LastCategoryFullName` | Full category path where the node was last found |
| `LastDependency` | The dependency (file/package) where the node was last resolved. Use `"Builtin"` for built-in VL constructs. |
| `LastSymbolSource` | Legacy alias for `LastDependency` (used in older files, both accepted) |
| `OverloadStrategy` | Overload resolution strategy (e.g., `"AllPinsThatAreNotCommon"`) |

### Choice Element

Each `<Choice>` has:

| Attribute | Required | Description |
|-----------|----------|-------------|
| `Kind` | Yes | ElementKind flag (see table below) |
| `Name` | Yes | Name of the symbol |
| `Fixed` | No | If `"true"`, this choice is locked |

### CategoryReference Element

Used to specify the category where a node lives:

```xml
<CategoryReference Kind="Category" Name="Primitive" />
```

Or with full namespace:

```xml
<FullNameCategoryReference ID="Primitive" />
```

### Common NodeReference Patterns

#### Operation Call (Process Node)
```xml
<p:NodeReference LastCategoryFullName="Control" LastDependency="CoreLibBasics.vl">
  <Choice Kind="NodeFlag" Name="Node" Fixed="true" />
  <Choice Kind="ProcessAppFlag" Name="MonoFlop" />
</p:NodeReference>
```

The first `Choice` with `Kind="NodeFlag"` and `Fixed="true"` is the "shape" indicator. The second choice specifies the actual node type:
- `Kind="ProcessAppFlag"` — Process node call (has state)
- `Kind="OperationCallFlag"` — Stateless operation call

#### Type Definition: Process (Container)
```xml
<p:NodeReference>
  <Choice Kind="ContainerDefinition" Name="Process" />
  <CategoryReference Kind="Category" Name="Primitive" />
</p:NodeReference>
```

#### Type Definition: Class
```xml
<p:NodeReference>
  <Choice Kind="ClassDefinition" Name="Class" />
  <CategoryReference Kind="Category" Name="Primitive" />
</p:NodeReference>
```

#### Type Definition: Record
```xml
<p:NodeReference>
  <Choice Kind="RecordDefinition" Name="Record" />
  <CategoryReference Kind="Category" Name="Primitive" />
</p:NodeReference>
```

#### Type Definition: Interface
```xml
<p:NodeReference>
  <Choice Kind="InterfaceDefinition" Name="Interface" />
  <CategoryReference Kind="Category" Name="Primitive" />
</p:NodeReference>
```

#### Type Definition: Forward (Wrapping .NET types)
```xml
<p:NodeReference>
  <Choice Kind="ForwardDefinition" Name="Forward" />
  <CategoryReference Kind="Category" Name="Primitive" />
</p:NodeReference>
```

#### Type Definition: Immutable Forward
```xml
<p:NodeReference>
  <Choice Kind="ForwardRecordDefinition" Name="Immutable Forward" />
  <CategoryReference Kind="Category" Name="Primitive" />
</p:NodeReference>
```

#### Operation Definition
```xml
<p:NodeReference>
  <Choice Kind="OperationDefinition" Name="Operation" />
  <CategoryReference Kind="Category" Name="Primitive" />
</p:NodeReference>
```

#### Region: If
```xml
<p:NodeReference LastCategoryFullName="Primitive" LastDependency="Builtin">
  <Choice Kind="StatefulRegion" Name="Region (Stateful)" Fixed="true" />
  <CategoryReference Kind="Category" Name="Primitive" />
  <Choice Kind="ApplicationStatefulRegion" Name="If" />
</p:NodeReference>
```

#### Region: ForEach
```xml
<p:NodeReference LastCategoryFullName="Primitive" LastDependency="Builtin">
  <Choice Kind="StatefulRegion" Name="Region (Stateful)" Fixed="true" />
  <CategoryReference Kind="Category" Name="Primitive" />
  <Choice Kind="ApplicationStatefulRegion" Name="ForEach" />
</p:NodeReference>
```

#### Region: Cache
```xml
<p:NodeReference LastCategoryFullName="Primitive" LastDependency="Builtin">
  <Choice Kind="StatefulRegion" Name="Region (Stateful)" Fixed="true" />
  <CategoryReference Kind="Category" Name="Primitive" />
  <Choice Kind="ProcessStatefulRegion" Name="Cache" />
</p:NodeReference>
```

### Important ElementKind Values for Choice.Kind

#### Node Call Kinds (used with NodeFlag)
| Kind | Description |
|------|-------------|
| `NodeFlag` | Base flag for all node calls (use with `Fixed="true"` as first choice) |
| `ProcessAppFlag` | Process node call (stateful) |
| `OperationCallFlag` | Operation call (stateless) |
| `OperationNode` | Direct .NET static method call (common in library files) |
| `ProcessNode` | Process node reference (e.g., `Kind="ProcessNode" Name="Changed"`) |

#### Definition Kinds
| Kind | Description |
|------|-------------|
| `ContainerDefinition` | Process type definition |
| `ClassDefinition` | Class definition |
| `RecordDefinition` | Record (immutable) definition |
| `InterfaceDefinition` | Interface definition |
| `ForwardDefinition` | Forward type (wrapping .NET mutable type) |
| `ForwardRecordDefinition` | Forward immutable type |
| `OperationDefinition` | Operation definition |
| `ProcessDefinition` | Process node definition |

#### Region Kinds
| Kind | Description |
|------|-------------|
| `StatefulRegion` | Base flag for all stateful regions (MUST be first Choice, with `Fixed="true"`) |
| `ApplicationStatefulRegion` | If, ForEach, ForLoop, Using, Try, ManageProcess, etc. |
| `ProcessStatefulRegion` | Cache region |
| `ApplicationRegion` | Delegate region |

#### Type Reference Kinds
| Kind | Description |
|------|-------------|
| `TypeFlag` | Base flag for type references |
| `ImmutableTypeFlag` | Immutable type |
| `MutableTypeFlag` | Mutable type |

#### Category Kinds (for CategoryReference)
| Kind | Description |
|------|-------------|
| `Category` | Category reference (by name) |
| `ClassType` | Referencing a class type as category |
| `RecordType` | Referencing a record type as category |
| `MutableInterfaceType` | Referencing an interface type as category |
| `InterfaceType` | Interface type reference |
| `AssemblyCategory` | .NET assembly namespace as category |
| `ArrayType` | Array type reference (e.g., `Name="MutableArray"`) |

#### Additional CategoryReference Attributes
| Attribute | Description |
|-----------|-------------|
| `NeedsToBeDirectParent` | If `true`, the category must be a direct parent |
| `IsGlobal` | If `true`, global category lookup |

---

## 11. Pin Element

A `<Pin>` represents an input or output parameter of a Node.

```xml
<Pin Id="H5v0bIB5B2iQbItncZ6Iom" Name="Simulate" Kind="InputPin" />
<Pin Id="Ooyrg7UiZbFM487c8ul2zt" Name="Output" Kind="OutputPin" />
```

### Attributes

| Attribute | Required | Default | Description |
|-----------|----------|---------|-------------|
| `Id` | Yes | — | Unique ID (used by Links to connect) |
| `Name` | Yes | — | Pin name |
| `Kind` | Yes | — | Pin kind (see below) |
| `Bounds` | No | — | Position/size as `"X,Y,W,H"` |
| `Visibility` | No | `"Visible"` | `Visible`, `Optional`, `OnCreateDefault`, `Hidden` |
| `IsHidden` | No | `false` | Hides the pin |
| `Exposition` | No | — | Pin exposition mode |
| `IsPinGroup` | No | `false` | If `true`, this is a pin group |
| `PinGroupName` | No | — | Name of the pin group |
| `PinGroupDefaultCount` | No | `0` | Default number of pins in group |
| `PinGroupEditMode` | No | — | Edit mode for pin group |

### Pin Kind Values

| Kind | Description |
|------|-------------|
| `InputPin` | Standard input pin |
| `OutputPin` | Standard output pin |
| `StateInputPin` | State input (for process definitions) |
| `StateOutputPin` | State output (for process definitions) |
| `ApplyPin` | Apply/trigger pin (execute trigger for process nodes) |

### Pin with Default Value

Default values can be specified either as an attribute (flat) or as a child element:

```xml
<!-- Flat attribute form (most common) -->
<Pin Id="..." Name="Value" Kind="InputPin" DefaultValue="42" />

<!-- Child element form -->
<Pin Id="..." Name="Value" Kind="InputPin">
  <p:DefaultValue>42</p:DefaultValue>
</Pin>
```

### Pin with TypeAnnotation

```xml
<Pin Id="..." Name="Input" Kind="InputPin">
  <p:TypeAnnotation>
    <Choice Kind="TypeFlag" Name="Float32" />
  </p:TypeAnnotation>
</Pin>
```

---

## 12. Pad Element (IOBox)

A `<Pad>` is a visual element on the canvas that can display and edit values. It is commonly known as an "IOBox".

```xml
<Pad Id="F1OWhZcr5kjQdAcHrH6BXa" Comment="Simulate" Bounds="207,227,35,35"
     ShowValueBox="true" isIOBox="true" Value="False">
  <p:TypeAnnotation LastCategoryFullName="Primitive" LastDependency="CoreLibBasics.vl">
    <Choice Kind="TypeFlag" Name="Boolean" />
  </p:TypeAnnotation>
</Pad>
```

### Attributes

| Attribute | Required | Default | Description |
|-----------|----------|---------|-------------|
| `Id` | Yes | — | Unique ID (used by Links) |
| `Bounds` | No | — | Position and size: `"X,Y,W,H"` |
| `Comment` | No | — | Display label / pin name for the pad |
| `ShowValueBox` | No | `false` | Whether the value editor is shown |
| `isIOBox` | No | `false` | Whether this is an IOBox (note: lowercase `i`!) |
| `SlotId` | No | — | ID of the Slot element this Pad is connected to (for state fields) |
| `Value` | No | — | The serialized value as string |

**Important**: Note the lowercase `i` in `isIOBox` — this is how it appears in the XML.

### Children

| Child | Description |
|-------|-------------|
| `<p:TypeAnnotation>` | Type of the value |
| `<p:ValueBoxSettings>` | Settings for the value editor widget |

### ValueBoxSettings

```xml
<p:ValueBoxSettings>
  <p:fontsize p:Type="Int32">9</p:fontsize>
  <p:stringtype p:Assembly="VL.Core" p:Type="VL.Core.StringType">Comment</p:stringtype>
  <p:buttonmode p:Assembly="VL.UI.Forms" p:Type="VL.HDE.PatchEditor.Editors.ButtonModeEnum">Bang</p:buttonmode>
</p:ValueBoxSettings>
```

Each setting inside `<p:ValueBoxSettings>` is a property element in the `p` namespace. The `p:Type` attribute specifies the .NET type, and `p:Assembly` the assembly if different from the default.

### Common Pad Types

#### String IOBox
```xml
<Pad Id="..." Bounds="100,200,200,20" ShowValueBox="true" isIOBox="true" Value="Hello World">
  <p:TypeAnnotation>
    <Choice Kind="TypeFlag" Name="String" />
  </p:TypeAnnotation>
</Pad>
```

#### Comment (String with Comment style)
```xml
<Pad Id="..." Bounds="100,200,300,60" ShowValueBox="true" isIOBox="true" Value="This is a comment">
  <p:TypeAnnotation>
    <Choice Kind="TypeFlag" Name="String" />
  </p:TypeAnnotation>
  <p:ValueBoxSettings>
    <p:fontsize p:Type="Int32">9</p:fontsize>
    <p:stringtype p:Assembly="VL.Core" p:Type="VL.Core.StringType">Comment</p:stringtype>
  </p:ValueBoxSettings>
</Pad>
```

#### Boolean IOBox (Toggle)
```xml
<Pad Id="..." Bounds="100,200,35,35" ShowValueBox="true" isIOBox="true" Value="False">
  <p:TypeAnnotation>
    <Choice Kind="TypeFlag" Name="Boolean" />
  </p:TypeAnnotation>
</Pad>
```

#### Boolean IOBox (Bang)
```xml
<Pad Id="..." Bounds="100,200,35,35" ShowValueBox="true" isIOBox="true" Value="False">
  <p:TypeAnnotation>
    <Choice Kind="TypeFlag" Name="Boolean" />
  </p:TypeAnnotation>
  <p:ValueBoxSettings>
    <p:buttonmode p:Assembly="VL.UI.Forms" p:Type="VL.HDE.PatchEditor.Editors.ButtonModeEnum">Bang</p:buttonmode>
  </p:ValueBoxSettings>
</Pad>
```

#### Float32 IOBox
```xml
<Pad Id="..." Bounds="100,200,80,20" ShowValueBox="true" isIOBox="true" Value="0.5">
  <p:TypeAnnotation>
    <Choice Kind="TypeFlag" Name="Float32" />
  </p:TypeAnnotation>
</Pad>
```

#### Integer IOBox
```xml
<Pad Id="..." Bounds="100,200,80,20" ShowValueBox="true" isIOBox="true" Value="42">
  <p:TypeAnnotation>
    <Choice Kind="TypeFlag" Name="Int32" />
  </p:TypeAnnotation>
</Pad>
```

---

## 13. Link Element

A `<Link>` connects two data endpoints by referencing their IDs.

```xml
<Link Id="Rhj440tINc9QVWqcrgSQGL" Ids="Ooyrg7UiZbFM487c8ul2zt,QQSKYIvpJi4La4O3CqBzTm" />
```

### Attributes

| Attribute | Required | Default | Description |
|-----------|----------|---------|-------------|
| `Id` | Yes | — | Unique ID |
| `Ids` | Yes | — | Comma-separated: `"SourceId,SinkId"` (output-to-input) |
| `IsHidden` | No | `false` | If `true`, the link is a "reference" (not visually drawn as a line) |
| `IsFeedback` | No | `false` | If `true`, this is a feedback link (loop) |

### Link Direction

**Source → Sink** means **Output → Input**:
- The first ID in `Ids` is the **source** (an OutputPin or a Pad)
- The second ID in `Ids` is the **sink** (an InputPin or a Pad)

### What Can Be Linked

Links connect IDs of:
- `Pin` elements (InputPin/OutputPin)
- `Pad` elements (IOBoxes)
- `ControlPoint` elements (on regions)

The IDs in the `Ids` attribute are the SerializedIds (the `Id` attributes) of these elements.

---

## 14. ProcessDefinition and Fragments

A `<ProcessDefinition>` defines the lifecycle of a process-type node. It sits inside a Node's Patch and links to the operation patches via Fragments.

### Structure

```xml
<Node Name="Application" ...>
  <p:NodeReference>
    <Choice Kind="ContainerDefinition" Name="Process" />
    <CategoryReference Kind="Category" Name="Primitive" />
  </p:NodeReference>
  <Patch Id="patchId">
    <Canvas Id="..." CanvasType="Group" />

    <!-- Named operation patches -->
    <Patch Id="createPatchId" Name="Create" />
    <Patch Id="updatePatchId" Name="Update" />

    <!-- ProcessDefinition links them together -->
    <ProcessDefinition Id="procDefId">
      <Fragment Id="frag1Id" Patch="createPatchId" Enabled="true" />
      <Fragment Id="frag2Id" Patch="updatePatchId" Enabled="true" />
    </ProcessDefinition>

    <!-- Links between nodes in the canvas go here -->
    <Link ... />
  </Patch>
</Node>
```

### ProcessDefinition Attributes

| Attribute | Required | Default | Description |
|-----------|----------|---------|-------------|
| `Id` | Yes | — | Unique ID |
| `IsHidden` | No | `false` | If `true`, not visible in node browser |
| `HasStateOut` | No | `false` | If `true`, has an explicit state output |

### Fragment Attributes

| Attribute | Required | Default | Description |
|-----------|----------|---------|-------------|
| `Id` | Yes | — | Unique ID |
| `Patch` | Yes | — | ID of the operation Patch this fragment references |
| `Enabled` | No | — | `"true"` if this fragment is active. Can be omitted (treated as implicitly enabled). |
| `IsDefault` | No | `false` | If `true`, this is the default fragment |

**Important**: The `Patch` attribute of a Fragment references the `Id` of a sibling `<Patch>` element (e.g., the Create or Update patch).

---

## 15. Slot Element

A `<Slot>` defines a state field within a type definition (Process, Class, Record).

```xml
<Slot Id="..." Name="MyField">
  <p:TypeAnnotation>
    <Choice Kind="TypeFlag" Name="Float32" />
  </p:TypeAnnotation>
  <p:Value>0.5</p:Value>
</Slot>
```

### Attributes

| Attribute | Required | Description |
|-----------|----------|-------------|
| `Id` | Yes | Unique ID |
| `Name` | Yes | Field name |

### Children

| Child | Description |
|-------|-------------|
| `<p:TypeAnnotation>` | Type of the field |
| `<p:Value>` | Default value (serialized) |
| `<p:Summary>` | Documentation summary |
| `<p:Remarks>` | Documentation remarks |

---

## 16. Type Definitions

Type definitions are `<Node>` elements with specific `NodeReference` patterns. They contain an inner `<Patch>` with the type's operations.

### Process Type Definition (most common)

```xml
<Node Name="MyProcess" Bounds="100,100" Id="...">
  <p:NodeReference>
    <Choice Kind="ContainerDefinition" Name="Process" />
    <CategoryReference Kind="Category" Name="Primitive" />
  </p:NodeReference>
  <Patch Id="...">
    <Canvas Id="..." CanvasType="Group">
      <!-- Visual content of the type (nodes, pads, links) -->
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

### Class Type Definition

```xml
<Node Name="MyClass" Bounds="100,100" Id="...">
  <p:NodeReference>
    <Choice Kind="ClassDefinition" Name="Class" />
    <CategoryReference Kind="Category" Name="Primitive" />
  </p:NodeReference>
  <Patch Id="...">
    <Canvas Id="..." CanvasType="Group" />
    <!-- Operation patches, ProcessDefinition, Slots, etc. -->
  </Patch>
</Node>
```

### Record Type Definition

```xml
<Node Name="MyRecord" Bounds="100,100" Id="...">
  <p:NodeReference>
    <Choice Kind="RecordDefinition" Name="Record" />
    <CategoryReference Kind="Category" Name="Primitive" />
  </p:NodeReference>
  <Patch Id="...">
    <Canvas Id="..." CanvasType="Group" />
    <!-- Slots (fields), ProcessDefinition, etc. -->
  </Patch>
</Node>
```

### Forward Type Definition (Wrapping .NET Types)

```xml
<Node Name="MyDotNetType" Bounds="100,100" Id="...">
  <p:NodeReference>
    <Choice Kind="ForwardDefinition" Name="Forward" />
    <CategoryReference Kind="Category" Name="Primitive" />
  </p:NodeReference>
  <p:TypeAnnotation LastCategoryFullName="Some.Namespace" LastDependency="SomeAssembly.dll">
    <Choice Kind="TypeFlag" Name="MyDotNetType" />
  </p:TypeAnnotation>
  <p:ForwardAllNodesOfTypeDefinition p:Type="Boolean">true</p:ForwardAllNodesOfTypeDefinition>
  <Patch Id="...">
    <Canvas Id="..." CanvasType="Group" />
    <ProcessDefinition Id="..." IsHidden="true" />
  </Patch>
</Node>
```

---

## 17. Region Nodes

Regions are `<Node>` elements with specific `NodeReference` patterns and a larger bounding box.

### If Region

```xml
<Node Bounds="100,200,400,300" Id="...">
  <p:NodeReference LastCategoryFullName="Primitive" LastDependency="Builtin">
    <Choice Kind="StatefulRegion" Name="Region (Stateful)" Fixed="true" />
    <CategoryReference Kind="Category" Name="Primitive" />
    <Choice Kind="ApplicationStatefulRegion" Name="If" />
  </p:NodeReference>
  <Patch Id="...">
    <Canvas Id="..." CanvasType="Group">
      <!-- Content inside the If region -->
    </Canvas>
    <Patch Id="thenPatchId" Name="Then" />
    <Fragment Id="..." Patch="thenPatchId" Enabled="true" />
  </Patch>
  <ControlPoint Id="..." Bounds="150,200" Alignment="Top" />
  <ControlPoint Id="..." Bounds="150,500" Alignment="Bottom" />
</Node>
```

### ForEach Region

```xml
<Node Bounds="100,200,400,300" Id="...">
  <p:NodeReference LastCategoryFullName="Primitive" LastDependency="Builtin">
    <Choice Kind="StatefulRegion" Name="Region (Stateful)" Fixed="true" />
    <CategoryReference Kind="Category" Name="Primitive" />
    <Choice Kind="ApplicationStatefulRegion" Name="ForEach" />
  </p:NodeReference>
  <Patch Id="...">
    <Canvas Id="..." CanvasType="Group">
      <!-- Loop body content -->
    </Canvas>
    <Patch Id="createId" Name="Create" />
    <Patch Id="updateId" Name="Update" />
    <Patch Id="disposeId" Name="Dispose" />
    <Fragment Id="..." Patch="createId" />
    <Fragment Id="..." Patch="updateId" Enabled="true" />
    <Fragment Id="..." Patch="disposeId" />
  </Patch>
  <ControlPoint Id="..." Bounds="150,200" Alignment="Top" />
  <ControlPoint Id="..." Bounds="150,500" Alignment="Bottom" />
</Node>
```

### Cache Region (ProcessStatefulRegion)

```xml
<Node Bounds="100,200,300,200" Id="...">
  <p:NodeReference LastCategoryFullName="Primitive" LastDependency="Builtin">
    <Choice Kind="StatefulRegion" Name="Region (Stateful)" Fixed="true" />
    <CategoryReference Kind="Category" Name="Primitive" />
    <Choice Kind="ProcessStatefulRegion" Name="Cache" />
  </p:NodeReference>
  <Patch Id="...">
    <Canvas Id="..." CanvasType="Group">
      <!-- Cached content -->
    </Canvas>
    <Patch Id="createId" Name="Create" />
    <Patch Id="updateId" Name="Update" />
    <Fragment Id="..." Patch="createId" />
    <Fragment Id="..." Patch="updateId" Enabled="true" />
  </Patch>
  <ControlPoint Id="..." Bounds="150,200" Alignment="Top" />
  <ControlPoint Id="..." Bounds="150,400" Alignment="Bottom" />
</Node>
```

### Region NodeReference Pattern

**Important**: Regions use `Kind="StatefulRegion"` as the FIRST Choice, NOT `Kind="NodeFlag"`. The order is:
1. `<Choice Kind="StatefulRegion" Name="Region (Stateful)" Fixed="true" />`
2. `<CategoryReference Kind="Category" Name="Primitive" />`
3. `<Choice Kind="ApplicationStatefulRegion|ProcessStatefulRegion" Name="..." />`

### Region ControlPoints

ControlPoints define the data entry/exit points on region borders:
- `Alignment="Top"` — input control point at the top border
- `Alignment="Bottom"` — output control point at the bottom border
- ControlPoints use 2-value Bounds: `"X,Y"` (position only)
- ControlPoint IDs can be used in Links to connect data into/out of regions

### Region Patch Names

| Region | Patch Names |
|--------|------------|
| If | `Then` (and optionally `Else`) |
| ForEach | `Create`, `Update`, `Dispose` |
| Cache | `Create`, `Update` |
| Try | `Try`, `Catch` |

### Region Size

Regions should have explicit width and height in their `Bounds` attribute (4 values: `"X,Y,W,H"`), unlike regular nodes which typically only need position (`"X,Y"`). Minimum recommended region size is about `200,150`.

---

## 18. TypeAnnotation and TypeReference

TypeAnnotations define the type of a Pad, Pin, Slot, or Forward node. They use the Choice-based reference system.

### Simple Type Reference

```xml
<p:TypeAnnotation>
  <Choice Kind="TypeFlag" Name="Float32" />
</p:TypeAnnotation>
```

### Type Reference with Category Info

```xml
<p:TypeAnnotation LastCategoryFullName="Primitive" LastDependency="CoreLibBasics.vl">
  <Choice Kind="TypeFlag" Name="Boolean" />
</p:TypeAnnotation>
```

### Common Type Names

| Name | .NET Type | Description |
|------|-----------|-------------|
| `Boolean` | `bool` | True/False |
| `Byte` | `byte` | 8-bit unsigned |
| `Int32` | `int` | 32-bit integer |
| `Int64` | `long` | 64-bit integer |
| `Float32` | `float` | 32-bit float |
| `Float64` | `double` | 64-bit float |
| `String` | `string` | Text |
| `Vector2` | `Vector2` | 2D vector |
| `Vector3` | `Vector3` | 3D vector |
| `Vector4` | `Vector4` | 4D vector |

### Generic Type Reference (with TypeArguments)

For generic types like `Spread<RGBA>`:

```xml
<p:TypeAnnotation LastCategoryFullName="Collections" LastDependency="VL.Collections.vl">
  <Choice Kind="TypeFlag" Name="Spread" />
  <p:TypeArguments>
    <TypeReference LastCategoryFullName="Color" LastDependency="CoreLibBasics.vl">
      <Choice Kind="TypeFlag" Name="RGBA" />
    </TypeReference>
  </p:TypeArguments>
</p:TypeAnnotation>
```

For generic .NET types, use backtick notation for the type name: `ImmutableDictionary`2` for two type parameters.

### Custom/External Type Reference

For types from .NET assemblies:

```xml
<p:TypeAnnotation LastCategoryFullName="Some.Namespace" LastDependency="SomeAssembly.dll">
  <Choice Kind="TypeFlag" Name="MyTypeName" />
</p:TypeAnnotation>
```

### Interface Type References (for `p:Interfaces`)

```xml
<p:Interfaces>
  <TypeReference LastCategoryFullName="VL.Skia" LastDependency="VL.Skia.dll">
    <Choice Kind="MutableInterfaceType" Name="IBehavior" />
  </TypeReference>
</p:Interfaces>
```

### PinReference (for overload disambiguation in NodeReference)

When a node reference needs pin-level type disambiguation:

```xml
<p:NodeReference ...>
  <Choice Kind="OperationNode" Name="MyOperation" />
  <PinReference Kind="InputPin" Name="Input">
    <p:DataTypeReference p:Type="TypeReference" LastCategoryFullName="Collections" LastDependency="VL.Collections.vl">
      <Choice Kind="TypeFlag" Name="Spread" />
    </p:DataTypeReference>
  </PinReference>
</p:NodeReference>
```

### LastDependency vs LastSymbolSource

Both `LastDependency` and `LastSymbolSource` serve the same purpose — identifying where a type/node was last resolved:
- **`LastDependency`**: Used in newer files (2024+). Preferred for new file generation.
- **`LastSymbolSource`**: Used in older files (pre-2024). Both are accepted by the parser.
- **`"Builtin"`**: Special sentinel value for built-in VL language constructs (e.g., `LastDependency="Builtin"`).

### Null Values

To explicitly set a null value, use the `r:` namespace (requires `xmlns:r="reflection"` on Document):

```xml
<p:Value r:IsNull="true" />
```

---

## 19. Property Serialization Format

The VL serializer uses two strategies for writing properties:

### 1. Flat Properties (XML Attributes)

Simple values are written as XML attributes on the element:

```xml
<Node Name="MyNode" Bounds="100,200" AutoConnect="false" />
```

The serializer calls `WritePropertyFlat(name, value)` for these. Flat properties include:
- `Id`, `Name`, `Bounds`, `Kind`, `Fixed`, `Position`
- Boolean flags, enum values, numeric values

### 2. Complex Properties (Namespaced Child Elements)

Complex objects are written as child elements in the `p:` (property) namespace:

```xml
<Node ...>
  <p:NodeReference>...</p:NodeReference>
  <p:TypeAnnotation>...</p:TypeAnnotation>
  <p:HelpFocus p:Assembly="VL.Lang" p:Type="VL.Model.HelpPriority">High</p:HelpFocus>
</Node>
```

The `p:` prefix corresponds to `xmlns:p="property"`.

### 3. Type Disambiguation

When a property value is a subtype, the `p:Type` and optionally `p:Assembly` attributes are added:

```xml
<p:HelpFocus p:Assembly="VL.Lang" p:Type="VL.Model.HelpPriority">High</p:HelpFocus>
```

- `p:Type` — The .NET type name (short name if same assembly, full name if different)
- `p:Assembly` — The assembly name (only if different from the declaring assembly)

### 4. Children (Range Serialization)

Child elements of Compound types are serialized as direct XML child elements using their class name as the element tag:

```xml
<Patch Id="...">
  <Canvas Id="..." />           <!-- Canvas child -->
  <Node Id="..." ... />         <!-- Node child -->
  <Link Id="..." ... />         <!-- Link child -->
  <Pad Id="..." ... />          <!-- Pad child -->
  <ProcessDefinition Id="..." /><!-- ProcessDefinition child -->
</Patch>
```

The XML element name matches the C# class name exactly (case-sensitive).

---

## 20. Complete Minimal Example

The absolute minimum valid `.vl` file:

```xml
<?xml version="1.0" encoding="utf-8"?>
<Document xmlns:p="property" xmlns:r="reflection" Id="A1b2C3d4E5f6G7h8I9j0Kl" LanguageVersion="2024.6.0" Version="0.128">
  <NugetDependency Id="B2c3D4e5F6g7H8i9J0k1Lm" Location="VL.CoreLib" Version="2024.6.0" />
  <Patch Id="C3d4E5f6G7h8I9j0K1l2Mn">
    <Canvas Id="D4e5F6g7H8i9J0k1L2m3No" DefaultCategory="Main" BordersChecked="false" CanvasType="FullCategory" />
    <Node Name="Application" Bounds="100,100" Id="E5f6G7h8I9j0K1l2M3n4Op">
      <p:NodeReference>
        <Choice Kind="ContainerDefinition" Name="Process" />
        <CategoryReference Kind="Category" Name="Primitive" />
      </p:NodeReference>
      <Patch Id="F6g7H8i9J0k1L2m3N4o5Pq">
        <Canvas Id="G7h8I9j0K1l2M3n4O5p6Qr" CanvasType="Group" />
        <Patch Id="H8i9J0k1L2m3N4o5P6q7Rs" Name="Create" />
        <Patch Id="I9j0K1l2M3n4O5p6Q7r8St" Name="Update" />
        <ProcessDefinition Id="J0k1L2m3N4o5P6q7R8s9Tu">
          <Fragment Id="K1l2M3n4O5p6Q7r8S9t0Uv" Patch="H8i9J0k1L2m3N4o5P6q7Rs" Enabled="true" />
          <Fragment Id="L2m3N4o5P6q7R8s9T0u1Vw" Patch="I9j0K1l2M3n4O5p6Q7r8St" Enabled="true" />
        </ProcessDefinition>
      </Patch>
    </Node>
  </Patch>
</Document>
```

---

## 21. Complete Working Example

A more complete example with nodes, pads, and links:

```xml
<?xml version="1.0" encoding="utf-8"?>
<Document xmlns:p="property" xmlns:r="reflection" Id="X1y2Z3a4B5c6D7e8F9g0Hi" LanguageVersion="2024.6.0" Version="0.128">
  <NugetDependency Id="Y2z3A4b5C6d7E8f9G0h1Ij" Location="VL.CoreLib" Version="2024.6.0" />
  <Patch Id="Z3a4B5c6D7e8F9g0H1i2Jk">
    <Canvas Id="A4b5C6d7E8f9G0h1I2j3Kl" DefaultCategory="Main" BordersChecked="false" CanvasType="FullCategory" />
    <Node Name="Application" Bounds="100,100" Id="B5c6D7e8F9g0H1i2J3k4Lm">
      <p:NodeReference>
        <Choice Kind="ContainerDefinition" Name="Process" />
        <CategoryReference Kind="Category" Name="Primitive" />
      </p:NodeReference>
      <Patch Id="C6d7E8f9G0h1I2j3K4l5Mn">
        <Canvas Id="D7e8F9g0H1i2J3k4L5m6No" CanvasType="Group">
          <!-- A + (Add) node -->
          <Node Bounds="300,200,65,19" Id="E8f9G0h1I2j3K4l5M6n7Op">
            <p:NodeReference LastCategoryFullName="Primitive.Math" LastDependency="CoreLibBasics.vl">
              <Choice Kind="NodeFlag" Name="Node" Fixed="true" />
              <Choice Kind="OperationCallFlag" Name="+" />
            </p:NodeReference>
            <Pin Id="F9g0H1i2J3k4L5m6N7o8Pq" Name="Input" Kind="InputPin" />
            <Pin Id="G0h1I2j3K4l5M6n7O8p9Qr" Name="Input 2" Kind="InputPin" />
            <Pin Id="H1i2J3k4L5m6N7o8P9q0Rs" Name="Output" Kind="OutputPin" />
          </Node>
          <!-- Float32 IOBox for first input -->
          <Pad Id="I2j3K4l5M6n7O8p9Q0r1St" Comment="Value A" Bounds="200,160,80,20" ShowValueBox="true" isIOBox="true" Value="3.14">
            <p:TypeAnnotation LastCategoryFullName="Primitive" LastDependency="CoreLibBasics.vl">
              <Choice Kind="TypeFlag" Name="Float32" />
            </p:TypeAnnotation>
          </Pad>
          <!-- Float32 IOBox for second input -->
          <Pad Id="J3k4L5m6N7o8P9q0R1s2Tu" Comment="Value B" Bounds="200,220,80,20" ShowValueBox="true" isIOBox="true" Value="2.71">
            <p:TypeAnnotation LastCategoryFullName="Primitive" LastDependency="CoreLibBasics.vl">
              <Choice Kind="TypeFlag" Name="Float32" />
            </p:TypeAnnotation>
          </Pad>
          <!-- Float32 IOBox for output display -->
          <Pad Id="K4l5M6n7O8p9Q0r1S2t3Uv" Comment="" Bounds="400,200,80,20" ShowValueBox="true" isIOBox="true">
            <p:TypeAnnotation LastCategoryFullName="Primitive" LastDependency="CoreLibBasics.vl">
              <Choice Kind="TypeFlag" Name="Float32" />
            </p:TypeAnnotation>
          </Pad>
        </Canvas>
        <Patch Id="L5m6N7o8P9q0R1s2T3u4Vw" Name="Create" />
        <Patch Id="M6n7O8p9Q0r1S2t3U4v5Wx" Name="Update" />
        <ProcessDefinition Id="N7o8P9q0R1s2T3u4V5w6Xy">
          <Fragment Id="O8p9Q0r1S2t3U4v5W6x7Yz" Patch="L5m6N7o8P9q0R1s2T3u4Vw" Enabled="true" />
          <Fragment Id="P9q0R1s2T3u4V5w6X7y8Za" Patch="M6n7O8p9Q0r1S2t3U4v5Wx" Enabled="true" />
        </ProcessDefinition>
        <!-- Links: connect Pad A -> Input, Pad B -> Input 2, Output -> display Pad -->
        <Link Id="Q0r1S2t3U4v5W6x7Y8z9Ab" Ids="I2j3K4l5M6n7O8p9Q0r1St,F9g0H1i2J3k4L5m6N7o8Pq" />
        <Link Id="R1s2T3u4V5w6X7y8Z9a0Bc" Ids="J3k4L5m6N7o8P9q0R1s2Tu,G0h1I2j3K4l5M6n7O8p9Qr" />
        <Link Id="S2t3U4v5W6x7Y8z9A0b1Cd" Ids="H1i2J3k4L5m6N7o8P9q0Rs,K4l5M6n7O8p9Q0r1S2t3Uv" />
      </Patch>
    </Node>
  </Patch>
</Document>
```

This example creates a simple program that adds two float values (3.14 + 2.71) and displays the result.

---

## 22. Common Patterns and Recipes

### Creating a New Empty Document

1. Generate a unique document ID
2. Add `NugetDependency` for `VL.CoreLib`
3. Create the top-level `Patch` with a `Canvas` (DefaultCategory="Main", CanvasType="FullCategory")
4. Create the `Application` Node with ContainerDefinition reference
5. Inside Application: Create inner Patch with Canvas, Create/Update patches, ProcessDefinition with Fragments

### Adding a Node Call to the Canvas

1. Generate a unique Node ID
2. Set `Bounds` to the desired position
3. Create `<p:NodeReference>` with appropriate Choice elements
4. Add any Pin elements needed for value overrides
5. Add Link elements to connect to other elements

### Connecting Two Elements

1. Identify the source element's ID (output pin or pad)
2. Identify the sink element's ID (input pin or pad)
3. Create a `<Link>` with `Ids="sourceId,sinkId"`

### Creating a Type Definition with Operations

1. Create a Node with appropriate definition kind (ContainerDefinition, ClassDefinition, etc.)
2. Inside the Node's Patch, create:
   - A Canvas (CanvasType="Group")
   - Named Patches for each operation (Create, Update, or custom)
   - A ProcessDefinition linking to these patches via Fragments
   - Any Slot elements for state fields

### Adding a Sub-Category

Create a nested Canvas within the top-level Canvas:

```xml
<Canvas Id="..." Name="MyCategory" Position="100,200">
  <!-- Definitions in this category -->
</Canvas>
```

---

## 23. Validation Rules and Gotchas

### Critical Rules

1. **All IDs must be unique** within the document. Duplicate IDs will cause load failures.

2. **IDs are base62-encoded GUIDs** — exactly 22 characters from `[0-9A-Za-z]`. Do not use standard GUID format.

3. **The `xmlns:p="property"` namespace** must be declared on the Document element. Without it, all `<p:...>` elements will be invalid XML.

4. **Fragment Patch references** must point to existing Patch IDs within the same parent. A Fragment with `Patch="XYZ"` requires a `<Patch Id="XYZ" ...>` sibling.

5. **Link Ids** must reference existing Pin or Pad IDs. The format is `"sourceId,sinkId"` with exactly one comma.

6. **CanvasType="FullCategory"** should only be used for the top-level document Canvas. Inner canvases use `"Group"` or `"Category"`.

7. **Every Document should have a `VL.CoreLib` dependency** unless it's a very specialized file that only uses platform dependencies.

8. **The Application node is the entry point.** Every document typically has a Node named `"Application"` with `ContainerDefinition` reference.

### Common Mistakes

- **Wrong namespace**: Using `property` instead of `p` prefix, or forgetting the namespace declaration.
- **Missing `Version="0.128"`**: This legacy field must always be present.
- **Incorrect Bounds format**: Must be `"X,Y"` or `"X,Y,W,H"` with comma separators, no spaces.
- **Case sensitivity**: Element names are case-sensitive. `<Patch>` is correct, `<patch>` is not.
- **`isIOBox` casing**: Note the lowercase `i` — this is `isIOBox`, not `IsIOBox`.
- **Dependencies inside Patch**: Dependencies must be direct children of `<Document>`, not inside the `<Patch>`.
- **Missing ProcessDefinition**: Process/Class type definitions need a ProcessDefinition with at least the Update fragment.
- **Link direction**: First ID = source (output), Second ID = sink (input). Reversing them creates an invalid connection.

### Encoding Notes

- **String encoding**: Special characters in string values use a custom encoding (VL.Core.Serialization.Encode/Decode). For most practical purposes, standard XML escaping works. The `<` character is escaped as `&lt;`, but within Value attributes, VL uses `&lt;` to start rich-text markers like `&lt; ` for comments.
- **InvariantCulture**: All numeric values use InvariantCulture formatting (period `.` as decimal separator, no thousands separator).
- **Newlines in comments**: XML comments use `\r\n` and are entitized per XML spec.

### Version String

The `LanguageVersion` attribute should be a valid NuGet version string. For new documents, use the current VL version. If unknown, a version like `"2024.6.0"` is a reasonable default.

---

## Appendix: Element Serialization Summary

| XML Element | Base Class | Key Properties (Attributes) | Key Properties (p: Children) |
|-------------|-----------|---------------------------|------------------------------|
| `Document` | Compound | `Id`, `LanguageVersion`, `Version` | `FilePath`, `Summary`, `Authors` |
| `Patch` | Compound | `Id`, `Name`, `IsGeneric`, `SortPinsByX`, `ManuallySortedPins`, `ParticipatingElements` | — |
| `Canvas` | Compound | `Id`, `Name`, `Position`, `DefaultCategory`, `BordersChecked`, `CanvasType` | — |
| `Node` | Compound | `Id`, `Name`, `Bounds`, `Category`, `AutoConnect`, `Aspects`, `StructureOfTypeDefinition`, `HideCategory`, `Summary`, `Remarks`, `Tags` | `NodeReference`, `TypeAnnotation`, `HelpFocus`, `ForwardAllNodesOfTypeDefinition`, `Interfaces` |
| `ProcessDefinition` | Node | (inherits Node) + `IsHidden`, `HasStateOut` | (inherits Node) |
| `Pin` | DataHub | `Id`, `Name`, `Kind`, `Bounds`, `DefaultValue`, `Visibility`, `IsHidden`, `Exposition`, `IsPinGroup`, `PinGroupName`, `PinGroupDefaultCount` | `TypeAnnotation`, `Comment`, `DefaultValue`, `Summary`, `Remarks` |
| `Pad` | DataHub | `Id`, `Bounds`, `Comment`, `ShowValueBox`, `isIOBox`, `SlotId`, `Value` | `TypeAnnotation`, `ValueBoxSettings` |
| `Link` | Element | `Id`, `Ids`, `IsHidden`, `IsFeedback` | — |
| `Fragment` | Element | `Id`, `Patch`, `Enabled`, `IsDefault` | — |
| `Slot` | Element | `Id`, `Name` | `TypeAnnotation`, `Value`, `Summary`, `Remarks` |
| `NugetDependency` | Dependency | `Id`, `Location`, `Version`, `IsForward`, `IsFriend` | — |
| `DocumentDependency` | Dependency | `Id`, `Location`, `IsForward`, `IsFriend` | — |
| `PlatformDependency` | Dependency | `Id`, `Location`, `IsForward`, `IsFriend` | — |
| `Choice` | — | `Kind`, `Name`, `Fixed` | — |
| `CategoryReference` | Choice | `Kind`, `Name`, `Fixed`, `NeedsToBeDirectParent`, `IsGlobal` | `OuterCategoryReference` |
| `FullNameCategoryReference` | — | `ID` | — |
| `ControlPoint` | DataHub | `Id`, `Bounds`, `Name`, `Alignment` | `TypeAnnotation`, `Comment` |
| `Overlay` | Element | `Id`, `Bounds`, `Name`, `InViewSpace` | — |
