# VL.Serialization

## Serialization Libraries for vvvv gamma / VL

VL provides four layers of serialization support, each targeting different use cases — from simple XML persistence to high-performance zero-copy binary blitting.

---

## Overview

| Library | Formats | Human-Readable | Speed | Use Case |
|---------|---------|---------------|-------|----------|
| **Core (System.Serialization)** | XML | Yes | Low-Medium | File persistence, settings |
| **VL.Serialization.FSPickler** | XML, JSON, Binary | XML/JSON: Yes | Medium | Debugging, inspection |
| **VL.Serialization.MessagePack** | MessagePack binary, JSON | JSON mode: Yes | High | Network, real-time data |
| **VL.Serialization.Raw** | Raw bytes | No | Highest | GPU/audio buffers, sensors |

---

## 1. Core Serialization (System.Serialization)

**Source:** `VL.CoreLib/System.Serialization.vl`
**Category:** `System.Serialization`

The built-in serialization framework provides the abstract contract and a ready-to-use persistence node.

### Key Nodes

| Node | Description |
|------|-------------|
| `Serialize` | Serializes any VL type to XML string |
| `Deserialize` | Deserializes XML string to typed value |
| `Serialize (Log Errors)` | Serialize with error output pins |
| `Deserialize (Log Errors)` | Deserialize with error output pins |
| `Persistent` | All-in-one save/load process node |
| `RegisterSerializer` | Register custom `ISerializer` implementations |

### The Persistent Node

The highest-level convenience node for file-based persistence:

**Inputs:**
- `Input` — Data to persist (generic type)
- `File Path` — Target file path
- `Write` — Trigger save to file
- `Read` — Trigger load from file
- `Read On Open` — Auto-load when the node starts
- `Include Defaults` — Include default values in serialization
- `Throw On Error` — Error handling mode

**Outputs:**
- `Output` — Current data value
- `Error Messages` — Any serialization errors

**Behavior:** Uses XML as default format via `XMLReader`/`XMLWriter`. On Write: serializes data, creates directory if needed, writes XML. On Read: checks file exists, reads XML, deserializes.

### Core Abstractions

- **`ISerializer<T>`** — Generic interface for custom serializers (wraps `VL.Core.ISerializer<T>`)
- **`SerializationContext`** — Stateful context providing `Serialize` and `Deserialize` operations

---

## 2. VL.Serialization.FSPickler

**Source:** `VL.Serialization.FSPickler/VL.Serialization.FSPickler.vl`
**Category:** `Serialization.FSPickler`
**Based on:** [MBrace.FsPickler](https://github.com/mbraceproject/FsPickler) (F# serialization framework)

### Key Nodes

| Node | Input | Output |
|------|-------|--------|
| `Serialize (Xml)` | `T`, `Indent` | `string` |
| `Deserialize (Xml)` | `string`, `Indent` | `T` |
| `Serialize (Json)` | `T`, `Indent`, `Omit Header` | `string` |
| `Deserialize (Json)` | `string`, `Indent`, `Omit Header` | `T` |
| `Serialize (Binary)` | `T`, `Force Little Endian` | `byte[]` |
| `Deserialize (Binary)` | `IEnumerable<byte>`, `Force Little Endian` | `T` |

Plus `Slicewise` variants for processing collections element-by-element.

### Implementation Details

- **TypeNameConverter** bridges VL's `TypeRegistry` with FsPickler's type name resolution
- **PicklerResolver** integrates with VL's hotswap system, creating wrap picklers that serialize process state via `GetStateObject`/`FromStateObject`
- Supports complex object graphs and nearly any .NET type
- **Warning:** "Avoid using for long-term persistence!" — format can break across type version changes
- Deserialization requires XML/JSON attributes in **alphabetical order**

### Characteristics

- Three human-readable formats (XML, JSON) plus compact binary
- Hotswap-aware (clears cache on re-instantiation)
- Best for debugging and inspecting serialized data
- Not recommended for long-term storage

---

## 3. VL.Serialization.MessagePack

**Source:** `VL.Serialization.MessagePack/VL.Serialization.MessagePack.vl`
**Category:** `Serialization.MessagePack`
**Based on:** [MessagePack-CSharp](https://github.com/neuecc/MessagePack-CSharp)

### Key Nodes

| Node | Input | Output |
|------|-------|--------|
| `Serialize` | `T` | `byte[]` |
| `Deserialize` | `IEnumerable<byte>` | `T` |
| `SerializeJson` | `T`, `Prettify` | `string` |
| `DeserializeJson` | `string` | `T` |

### VL-Specific Type Support

The `VLResolver` chains multiple resolvers for rich type support:

| Type | Formatter | Strategy |
|------|-----------|----------|
| **IVLObject** (patched types) | `IVLObjectFormatter` | Serializes properties as MessagePack map |
| **Spread\<T\>** | `SpreadFormatter` | Serialized as arrays |
| **Optional\<T\>** | `OptionalFormatter` | Handles valued/none states |
| **IDynamicEnum** | `DynamicEnumFormatter` | Serialized as string value |
| **Stride types** (Vector2, Matrix, Color4) | `StrideFormatter` | Custom binary layout |
| **SkiaSharp types** | `SkiaFormatter` | Custom binary layout |
| Standard .NET types | `StandardResolver` | Built-in MessagePack support |

### IVLObject Serialization Pattern

VL-patched types (processes, records) are serialized by:
1. Writing all properties marked `ShouldBeSerialized` as a map
2. Using `IVLPropertyInfo.NameForTextualCode` as keys
3. Deserializing via `AppHost.CreateInstance()` + `IVLObject.With(values)`
4. Expression tree compilation for high-performance property access

### Characteristics

- Fastest text-friendly serializer (compact binary + JSON mode)
- Richest VL-specific type support
- Ideal for network communication and real-time data
- JSON mode useful for debugging MessagePack data

---

## 4. VL.Serialization.Raw

**Source:** `VL.Serialization.Raw/VL.Serialization.Raw.vl`
**Category:** `Serialization.Raw`

The simplest and fastest serializer — works only with types that can be directly blitted to/from memory.

### Key Nodes

| Node | Input | Output |
|------|-------|--------|
| `Serialize` | `T` | `ReadOnlyMemory<byte>` |
| `Deserialize` | `ReadOnlyMemory<byte>` | `T` |

### Supported Types

| Type Category | Examples | Method |
|---------------|----------|--------|
| Blitable structs | `int`, `float`, `Vector2` | Single-element array cast |
| Arrays of blitable types | `float[]`, `byte[]` | Memory cast via `ReadOnlyMemoryExtensions.Cast` |
| `Spread<T>` | `Spread<float>` | `AsMemory()` + cast |
| `ImmutableArray<T>` | `ImmutableArray<int>` | Memory cast |
| `ReadOnlyMemory<T>` | `ReadOnlyMemory<float>` | Direct cast (zero-copy) |
| `ArraySegment<T>` | `ArraySegment<byte>` | Span-based copy |
| `string` | `string` | UTF-8 encoding/decoding |

### How It Works

Uses `BlitableUtils` to check `RuntimeHelpers.IsReferenceOrContainsReferences<T>()` — if a type contains no reference fields, it can be safely memory-mapped. Expression trees are compiled for high-performance serialization with aggressive `ConcurrentDictionary` caching.

### Characteristics

- Zero serialization overhead for supported types
- No type metadata in output (caller must know the type)
- Cannot handle complex object graphs or reference types
- Ideal for GPU data transfer, audio/video pipelines, high-frequency sensor data

---

## When to Use Which Format

| Scenario | Recommended |
|----------|-------------|
| Save application settings to disk | **Core (Persistent node)** |
| Configuration files, project state | **Core (Persistent node)** |
| Debugging serialized data | **FSPickler** (XML/JSON) |
| Inspecting data structure | **FSPickler** (JSON) or **MessagePack** (JSON mode) |
| Network communication | **MessagePack** (binary) |
| Real-time data exchange | **MessagePack** (binary) |
| Inter-process communication | **MessagePack** (binary) |
| GPU texture/buffer data | **Raw** |
| Audio buffer streaming | **Raw** |
| High-frequency sensor data | **Raw** |
| Large numeric arrays | **Raw** |

---

## Dependencies

| Library | Key Dependencies |
|---------|-----------------|
| Core | VL.CoreLib, VL.Xml |
| FSPickler | MBrace.FsPickler, VL.Core |
| MessagePack | MessagePack-CSharp, VL.Core |
| Raw | VL.Core (minimal) |
