# VL.IO

## Input/Output Libraries for vvvv gamma / VL

VL provides six I/O subsystems for different communication patterns — from local file access to networked data synchronization.

---

## Overview

| Library | Purpose | Scope | Latency |
|---------|---------|-------|---------|
| **File I/O** | Disk read/write | Local machine | High (disk) |
| **HTTP** | Web requests, downloads | Network/Internet | Medium |
| **Redis** | Shared state, pub/sub | LAN/Network | Low (memory) |
| **OSCQuery** | OSC parameter sync | LAN | Low |
| **Named Pipes** | Inter-process communication | Local machine | Very Low |
| **TPL Dataflow** | Parallel pipelines | In-process | N/A |

---

## 1. File I/O

**Source:** `VL.CoreLib/src/IO/`
**Category:** `IO`

### Architecture

File I/O is built around a **stream-based resource provider model**. The central abstraction is `IResourceProvider<Stream>`, which lazily creates and disposes file streams.

### Key Nodes

| Node | Description |
|------|-------------|
| `File` | Creates `IResourceProvider<Stream>` from path, mode, access, share |
| `ByteReader` / `CharReader` | Read chunks from streams |
| `ByteWriter` / `CharWriter` | Write data to streams |
| `XDocumentReader` / `XDocumentWriter` | XML document I/O |
| `FileWatcher` | Watch filesystem for changes (Changed, Created, Deleted, Renamed) |
| `FileMover` | Async copy/move with cancellation |

### File Node Pins

**Inputs:** `filePath` (Path), `fileMode` (default: OpenOrCreate), `fileAccess` (default: Read), `fileShare` (default: Read)
**Output:** `IResourceProvider<Stream>`

Auto-creates directories when file mode implies creation.

### Async Reactive Variants

All readers/writers have async versions returning `IObservable<Chunk<T>>`:
- `AsyncByteReader`, `AsyncCharReader`, `AsyncXDocumentReader`
- `AsyncByteWriter`, `AsyncCharWriter`, `AsyncXDocumentWriter`

### Path Operations

| Node | Description |
|------|-------------|
| `ToPath` / `ToFilePath` / `ToDirectoryPath` | String-to-Path conversion |
| `Filename(directory, filename, extension)` | Path construction |
| `MakePath(path, string)` | Combine with normalization |
| `SystemFolder(SpecialFolder)` | Access well-known folders |
| `CreateDirectory` / `Move` / `Rename` / `Copy` / `Delete` | File operations |

### Typical Usage
```
Path → File node → IResourceProvider<Stream> → ByteReader → process data
                                              → ByteWriter → write data
```

---

## 2. HTTP

**Source:** `VL.CoreLib/src/IO/Net/`, `VL.CoreLib/src/_Experimental/IO/Net/`
**Category:** `IO.Net`

### Key Nodes

| Node | Description |
|------|-------------|
| `HTTPServer` | Static file server (experimental, HttpListener-based) |
| `Downloader` | Download URL to file with progress reporting |
| `WebRequest` utilities | Create/configure HTTP requests |
| `UdpSocket` | UDP socket management |
| `DatagramSender` | Reactive UDP sending via `IObservable<Datagram>` |

### HTTPServer (Experimental)

**Inputs:** `path` (directory to serve), `port` (default: 8080)
- Serves static files with MIME type detection (~50 types)
- Requires admin URL ACL: `netsh http add urlacl url=http://+:80/ user=username`

### Downloader

`HttpClientExtensions.DownloadFile(url, path, progress)`:
- Streaming download with `IProgress<float>` callback (0-100%)
- 10-second read timeout per chunk
- Retry logic

### WebRequest Extensions

- `AddHeaders` — custom request headers
- `SetProxy` — proxy configuration with credentials
- `GetResponse` / `GetResponseStream` — wrapped as `IResourceProvider<T>`

---

## 3. Redis

**Source:** `VL.IO.Redis/`
**Category:** `IO.Redis`
**Based on:** [StackExchange.Redis](https://github.com/StackExchange/StackExchange.Redis)

The most architecturally sophisticated I/O subsystem — synchronizes VL's global channel system with Redis keys.

### RedisClient — Central Process Node

**Input Pins:**

| Pin | Type | Default | Description |
|-----|------|---------|-------------|
| `configuration` | String | `"localhost:6379"` | Redis connection string |
| `nickname` | Optional\<string\> | — | Client display name |
| `database` | Int32 | -1 (default DB) | Redis database index |
| `initialization` | Initialization | None | None / Local / Redis |
| `bindingType` | BindingDirection | InOut | In / Out / InOut |
| `collisionHandling` | CollisionHandling | None | None / LocalWins / RedisWins |
| `serializationFormat` | SerializationFormat | MessagePack | MessagePack / Json / Raw |
| `expiry` | Optional\<TimeSpan\> | — | Key expiration time |
| `connectAsync` | Boolean | False | Non-blocking connection |

**Outputs:** `IsConnected`, `ClientName`

### Channel-to-Redis Binding

The core innovation: VL global channels are automatically synchronized with Redis keys.

- **Per-frame transaction batching** — all binding writes are batched into a single Redis transaction
- **Client-side tracking** — uses Redis `CLIENT TRACKING` for cache invalidation notifications
- **Collision handling** — LocalWins (skip reads), RedisWins (skip writes), None (both)
- **Initialization** — Local (write VL→Redis), Redis (read Redis→VL), None (no init)

### Pub/Sub Nodes

| Node | Description |
|------|-------------|
| `Publish<T>` | Publish messages to Redis channel (fire-and-forget) |
| `Subscribe<T>` | Subscribe to Redis channel → `IObservable<T?>` |
| `SubscribePattern<T>` | Glob pattern subscription (e.g., `sensor.*`) |

### Data Management

| Node | Description |
|------|-------------|
| `BindToRedis` | Programmatically bind VL channel to Redis key |
| `DeleteKey` | Delete a Redis key |
| `FlushDB` | Clear entire database |
| `Scan` / `ScanAsync` | Enumerate keys with glob patterns |
| `Get<T>` / `Set<T>` | Direct imperative get/set (bypass binding) |

### Serialization Formats

- **MessagePack** (default) — compact binary, best performance
- **Json** — human-readable, interoperable
- **Raw** — direct .NET serialization

### Typical Usage
```
RedisClient(configuration="localhost:6379")
  → BindToRedis(key="myKey", input=GlobalChannel)
  → automatic per-frame sync via transactions

RedisClient → Publish<T>(channel="events", input=observable)
RedisClient → Subscribe<T>(channel="events") → IObservable<T>
```

---

## 4. OSCQuery

**Source:** `VL.IO.OSCQuery/`
**Category:** `IO.OSCQuery`

Bridges VL's channel system with the OSCQuery protocol for parameter tree discovery and synchronization with OSC-compatible applications (DAWs, media servers).

### Key Features

- Parameter tree discovery via HTTP
- Value synchronization via OSC transport
- Integration with VL's Channel Browser UI
- Programmatic channel binding
- Custom type converters (e.g., `Color4` as `#RRGGBBAA` hex)

### Help Files

| File | Description |
|------|-------------|
| `Explanation Overview of OSCQuery.vl` | Protocol overview |
| `HowTo Bind Channels via Channel Browser.vl` | GUI-based binding |
| `HowTo Programmatically bind Channels.vl` | Code-based binding |
| `Reference OSCQueryClient.vl` | Client reference |

---

## 5. Named Pipes

**Source:** `VL.IO.Pipes/`
**Category:** `IO.Pipes`

Inter-process communication via Windows named pipes. Implemented primarily in VL visual patches.

### Key Features

- Bidirectional byte-stream communication between processes on the same machine
- Wraps .NET's `NamedPipeServerStream` and `NamedPipeClientStream`
- One process creates pipe server, another connects as client

### Help File

`HowTo Inter-Process Communication via NamedPipes.vl`

---

## 6. TPL Dataflow

**Source:** `VL.TPL.Dataflow/`
**Category:** `TPL.Dataflow`
**Based on:** [System.Threading.Tasks.Dataflow](https://docs.microsoft.com/en-us/dotnet/standard/parallel-programming/dataflow-task-parallel-library)

Concurrent data processing pipelines with backpressure and batching.

### Block Nodes

| Block | Type | Description |
|-------|------|-------------|
| `BufferBlock<T>` | Buffer | FIFO queue between blocks |
| `ActionBlock<T>` | Consumer | Terminal action for each item |
| `TransformBlock<T,U>` | Transform | Transform each input to output |
| `TransformManyBlock<T,U>` | Fan-out | One input → many outputs |
| `BroadcastBlock<T>` | Broadcast | Stores latest, broadcasts to all |
| `BatchBlock<T>` | Batching | Groups inputs into arrays |
| `JoinBlock<T1,T2>` | Join | Combines inputs from 2 sources |
| `WriteOnceBlock<T>` | Single-write | Stores exactly one value |

### Rx Integration

| Node | Description |
|------|-------------|
| `AsObservable<T>` | Convert `ISourceBlock<T>` → `IObservable<T>` |
| `AsObserver<T>` | Convert `ITargetBlock<T>` → `IObserver<T>` |

### All blocks support:
- Configurable parallelism via `ExecutionDataflowBlockOptions`
- Automatic propagation of completion
- Thread-safe state management via `StateManager<object>`
- Exception reporting through VL's `RuntimeGraph.ReportException`

### Typical Usage
```
BufferBlock<T> → TransformBlock<T,U> → ActionBlock<U>
                                     → BroadcastBlock<U> → multiple ActionBlocks
                                     → BatchBlock<U> → ActionBlock<U[]>
```

### Help Files

| File | Description |
|------|-------------|
| `Explanation Overview.vl` | Dataflow concepts |
| `HowTo producer-consumer dataflow pattern.vl` | Classic pattern |
| `HowTo Run rendering tasks in parallel.vl` | Parallel rendering |

---

## Cross-Cutting Patterns

### Resource Provider Model
File I/O, HTTP, and Sockets all use `IResourceProvider<T>` for lazy, safe resource lifecycle management.

### Reactive Integration
All libraries integrate with `System.Reactive`:
- Redis bindings use Observables
- File readers have async reactive variants
- Dataflow blocks convert to/from Observable/Observer
- UDP senders consume `IObservable<Datagram>`

### VL Channel System
Redis and OSCQuery implement `IModule` to integrate with VL's global Channel Browser, enabling GUI-driven binding configuration.

### Process Node Pattern
Every library exposes its API as `[ProcessNode]`-decorated classes with `Update()` methods, ensuring consistent VL integration.

---

## When to Use What

| Scenario | Use |
|----------|-----|
| Save settings to disk | **File I/O** (+ Persistent node) |
| Download files from web | **HTTP** Downloader |
| REST API calls | **HTTP** WebRequest |
| Shared state across VL instances | **Redis** |
| Real-time data sync on LAN | **Redis** (InOut binding) |
| Integration with media software | **OSCQuery** |
| Fast IPC on same machine | **Named Pipes** |
| Parallel processing pipelines | **TPL Dataflow** |
| High-frequency sensor data | **Redis** (Raw format) or **Named Pipes** |
