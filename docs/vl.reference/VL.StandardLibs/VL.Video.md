# VL.Video

## Video Playback and Capture for vvvv gamma / VL

VL.Video provides video file playback and camera capture functionality, built on top of Windows Media Foundation (MF). It is **Windows-only** (Windows 8+ for playback, Windows 7+ for capture).

---

## Overview

| Feature | Node | Description |
|---------|------|-------------|
| Video Playback (file) | `VideoPlayer` | Play video files from disk |
| Video Playback (URL) | `VideoPlayer (Url)` | Play video from web URLs |
| Camera Capture | `VideoIn` | Capture from USB cameras (UVC 1.1) |
| Audio Playback | Via `VideoPlayer` | Audio is part of video playback |
| System Audio | `SystemVolume` | Control Windows system volume |

---

## 1. VideoPlayer — Video File Playback

The primary playback node. `VideoPlayer` (file path) wraps `VideoPlayer (Url)` internally.

### Input Pins

| Pin | Type | Default | Description |
|-----|------|---------|-------------|
| `Filename` / `Url` | Path / String | — | Media file path or URL |
| `Play` | Boolean | False | Start/pause playback |
| `Rate` | Float32 | 1.0 | Playback speed (0.5 = half, 2.0 = double) |
| `Seek Time` | Float32 | 0 | Target seek position in seconds |
| `Seek` | Boolean | False | Trigger seeking |
| `Loop Start Time` | Float32 | 0 | Loop region start (seconds) |
| `Loop End Time` | Float32 | MaxValue | Loop region end (seconds) |
| `Loop` | Boolean | False | Enable looping |
| `Volume` | Float32 | 1.0 | Audio volume (0.0 to 1.0) |
| `Texture Size` | Int2 | 0,0 | Output resolution override (0 = native) |
| `Source Bounds` | RectangleF? | — | Normalized source rectangle for cropping |
| `Border Color` | Color4? | — | Border fill color |

### Output Pins

| Pin | Type | Description |
|-----|------|-------------|
| `Output` | IVideoSource | Video source for downstream use |
| `Playing` | Boolean | Whether playback is active |
| `Current Time` | Float32 | Current position in seconds |
| `Duration` | Float32 | Total duration in seconds |
| `Ready State` | ReadyState | Media readiness level |
| `Network State` | NetworkState | Network fetch status |
| `Error Code` | ErrorState | Error status |

### Supported Formats

Video: AVI, WMV, MP4, H.264, MJPEG, MPEG, DV, MOV
Audio: MP3, WAV, WMA, AAC, M4A

Format support depends on installed Windows Media Foundation codecs.

### Rendering Paths

The player auto-selects the optimal rendering path:

| Path | When | Output | Performance |
|------|------|--------|-------------|
| **GPU (Direct3D11)** | D3D11 device available (Stride) | `GpuVideoFrame<BgraPixel>` | Best — zero CPU-GPU transfer |
| **CPU (WIC)** | No GPU device (Skia) | `VideoFrame<BgraPixel>` | Good — memory-mapped pixels |

- GPU path uses `IMFDXGIDeviceManager` with `TexturePool` (DXGI_FORMAT_B8G8R8A8_UNORM)
- CPU path uses Windows Imaging Component with `BitmapPool`

---

## 2. VideoIn — Camera Capture

Captures video from USB cameras supporting UVC 1.1 drivers.

### Input Pins

| Pin | Type | Default | Description |
|-----|------|---------|-------------|
| `Device` | VideoCaptureDeviceEnumEntry | Default | Camera device (dynamic enum) |
| `Preferred Size` | Int2 | 1920,1080 | Desired capture resolution |
| `Preferred FPS` | Float32 | 30 | Desired frame rate |
| `Camera Controls` | CameraControls | — | Pan, Tilt, Roll, Zoom, Exposure, Iris, Focus |
| `Video Controls` | VideoControls | — | Brightness, Contrast, Hue, Saturation, Sharpness, Gamma |
| `Enabled` | Boolean | True | Enable/disable capture |

### Output Pins

| Pin | Type | Description |
|-----|------|-------------|
| `Output` | IVideoSource | Captured video source |
| `Actual FPS` | Float32 | Achieved frame rate |
| `Supported Formats` | String | List of device capabilities |

### Camera Controls

| Control | Property |
|---------|----------|
| Pan | `CameraControlProperty.Pan` |
| Tilt | `CameraControlProperty.Tilt` |
| Roll | `CameraControlProperty.Roll` |
| Zoom | `CameraControlProperty.Zoom` |
| Exposure | `CameraControlProperty.Exposure` |
| Iris | `CameraControlProperty.Iris` |
| Focus | `CameraControlProperty.Focus` |

### Video Processing Controls

Brightness, Contrast, Hue, Saturation, Sharpness, Gamma, ColorEnable, WhiteBalance, BacklightCompensation, Gain — all via DirectShow `VideoProcAmpProperty`.

### Device Enumeration

`VideoCaptureDeviceEnum` provides a live-updating list of connected cameras:
- Always includes "Default" entry for auto-selection
- Listens for hardware changes via `HardwareChangedEvents`
- Enumerates via MF `MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID`

---

## 3. Audio Support

### Audio in VideoPlayer

The VideoPlayer handles audio as part of media playback:
- `Volume` pin controls audio (0.0 to 1.0, clamped)
- Audio category: `AudioCategory_GameMedia`
- Supports audio-only files (MP3, WAV, WMA, AAC, M4A)

### Core Audio Types (VL.CoreLib)

Defined in `VL.CoreLib/Audio.vl`:
- **`AudioFrame`** — Single audio buffer frame
- **`IAudioSource`** — Interface for audio sources
- **`AudioStream`** — Streaming audio abstraction

### Windows Audio Nodes (VL.CoreLib.Windows)

| Node | Description |
|------|-------------|
| `SystemVolume` | Get/set Windows system volume |
| `PlaybackDevice` | Select audio output device |
| `CaptureDevice` | Select audio input device |
| `CommunicationDevice` | Set communication devices |

---

## 4. Integration with Renderers

### Architecture

```
VideoPlayer/VideoIn → IVideoSource → VideoSourceToImage<TImage> → Renderer-specific image
```

The `VideoSourceToImage<TImage>` bridge (in VL.CoreLib) supports:
- **Push-based streaming** (default) — uses `IObservable` on dedicated `EventLoopScheduler` thread
- **Pull-based streaming** — uses `IEnumerable`, single-threaded

### Skia Integration

Uses CPU/WIC rendering path. Frames arrive as memory-mapped `BgraPixel` data, converted to Skia images.

Help files:
- `HowTo Play a video file with Skia.vl`
- `HowTo Play a video file from a url with Skia.vl`
- `HowTo Capture video input from a camera for Skia.vl`

### Stride Integration

Uses GPU/Direct3D11 rendering path. Frames are rendered directly into D3D11 textures via `TransferVideoFrame` — no CPU-GPU transfer needed.

Help files:
- `HowTo Play a video file with Stride.vl`
- `HowTo Play a video file from a url with Stride.vl`
- `HowTo Capture video input from a camera for Stride.vl`

---

## 5. Platform Requirements

| Component | Minimum Windows Version |
|-----------|------------------------|
| VideoPlayer | Windows 8.0 |
| VideoIn (capture) | Windows 7 (6.1) |
| MediaFoundation init | Windows Vista (6.0.6000) |

- **Windows-only** — throws `PlatformNotSupportedException` on other platforms
- Uses unsafe code for COM interop with Media Foundation, Direct3D11, WIC, DirectShow
- D3D11 multithread protection enabled (`SetMultithreadProtected(true)`)
- Capture uses async source reader for non-blocking operation

### Cross-Platform Status

Core abstractions (`IVideoSource`, `VideoFrame`, `VideoPlaybackContext`) are in cross-platform `VL.Core.dll`, but all concrete implementations depend on Windows Media Foundation. No Linux/macOS backends exist.

---

## Dependencies

| Dependency | Purpose |
|------------|---------|
| VL.CoreLib | Core types (IVideoSource, AudioFrame) |
| Microsoft.Windows.CsWin32 | P/Invoke bindings |
| Windows Media Foundation | Video decode/encode engine |
| DirectShow | Camera control properties |
| WIC (Windows Imaging Component) | CPU rendering path |
| Direct3D11 | GPU rendering path |
