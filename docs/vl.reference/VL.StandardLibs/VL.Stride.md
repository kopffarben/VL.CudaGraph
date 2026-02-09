# VL.Stride

## 3D Graphics and Game Engine for vvvv gamma / VL

VL.Stride provides a full 3D rendering pipeline built on the open-source Stride game engine (formerly Xenko). It includes an entity-component system, PBR materials, dynamic lighting with shadows, post-processing, physics, VR support, GPU shader authoring with hot-reload, and compute shaders.

---

## Overview

| Feature | Description |
|---------|-------------|
| **Rendering** | Forward rendering via Direct3D 11 |
| **Scene Graph** | Entity-Component System (Scenes → Entities → Components) |
| **Materials** | Full PBR with 7 microfacet models |
| **Lighting** | Ambient, directional, point, spot, skybox IBL, light shafts, shadows |
| **Shaders** | SDSL/HLSL with 4 auto-generated node types + hot-reload |
| **Post-Processing** | 15+ effects (AO, DoF, Bloom, Fog, AA, Tone Map, etc.) |
| **Physics** | Bullet physics (static, dynamic, kinematic, triggers) |
| **VR** | OpenXR with hand tracking and passthrough |
| **GPU Compute** | ComputeFX with direct/indirect/custom dispatch |
| **Color Space** | Linear by default with sRGB conversion |

---

## Package Architecture

### VL.Stride.Runtime (Core)

The foundation package. Aggregates 11 sub-documents:

| Sub-Document | Purpose |
|---|---|
| `VL.Stride.Engine.vl` | Entity/component system, cameras, orbit camera |
| `VL.Stride.Rendering.vl` | Rendering pipeline, compositing |
| `VL.Stride.Graphics.vl` | GPU resources (textures, buffers, pipeline state) |
| `VL.Stride.Games.vl` | Game loop, window management |
| `VL.Stride.Input.vl` | Input handling |
| `VL.Stride.Rendering.TextureFX.vl` | TextureFX multi-pass wrapper nodes |
| `VL.Stride.Rendering.ShaderFX.vl` | Composable shader graph system |
| `VL.Stride.Rendering.Instancer.vl` | GPU instancing |
| `VL.Stride.Rendering.Temp.vl` | Experimental nodes |
| `VL.Stride.Runtime.TypeForwards.vl` | Type forwarding into VL |
| `VL.Stride.Video.vl` | Video/image sequence playback |

### VL.Stride (User-Facing)

Depends on VL.Stride.Runtime + VL.Stride.Windows. Contains all help patches.

### VL.Stride.TextureFX

80+ GPU texture effects (filters, mixers, sources) as `.sdsl` shader files.

---

## 1. Entity-Component System

### Scene Hierarchy

```
Scene (root container)
  └── Entity (has TransformComponent automatically)
        ├── EntityComponent (camera, light, model, physics, renderer, script)
        └── Child Entities
```

### Link System (`Links.cs`)

Four link types manage parent-child relationships with automatic cleanup:

| Link Type | Relationship | Method |
|-----------|-------------|--------|
| `EntityLink` | Entity → child Entity | `SetParent()` |
| `EntitySceneLink` | Scene → Entity | `entity.Scene = parent` |
| `SceneLink` | Scene → child Scene | `child.Parent = parent` |
| `ComponentLink` | Entity → EntityComponent | `entity.Add(component)` |

All check ownership before linking/unlinking to prevent double-parenting. `SceneUtils` enforces single-parent constraints with editor warnings.

### Registered Components

| Component | Category | Description |
|-----------|----------|-------------|
| `CameraComponent` | Stride.Cameras | Perspective/orthographic camera |
| `LightComponent` | Stride.Lights.Advanced | Any ILight type with intensity |
| `ModelComponent` | Stride.Models | 3D model with materials, shadow casting |
| `BackgroundComponent` | Stride.Textures | Background/skybox texture |
| `LightShaftComponent` | Stride.Lights.Advanced | Volumetric light shafts |
| `InputSourceComponent` | Stride.Experimental.Input | Entity input source |
| `InterfaceSyncScript` | Stride.Advanced | VL patch script bridge |

---

## 2. Cameras

### CameraComponent

| Pin | Type | Default | Description |
|-----|------|---------|-------------|
| `UseCustomViewMatrix` | bool | true | Use ViewMatrix pin vs. entity transform |
| `ViewMatrix` | Matrix | — | Direct camera placement |
| `Projection` | CameraProjectionMode | Perspective | Perspective or Orthographic |
| `VerticalFieldOfView` | float | 45/360 cycles | FOV (converted: `value * 360` degrees) |
| `OrthographicSize` | float | — | Size for orthographic mode |
| `NearClipPlane` | float | 0.05 | Near clipping plane |
| `FarClipPlane` | float | 100 | Far clipping plane |
| `AspectRatio` | float | — | Override viewport aspect ratio |

### OrbitCamera

Interactive orbit navigation defined in `VL.Stride.Engine.vl`:
- Pins: Initial Interest Point, Yaw, Pitch, Distance, FOV, Near/Far, Reset
- Combines `CameraInputSourceComponent` with `CameraLogic`

### BasicCameraController

FPS-style controller (`OrbitCameraController.cs`) with WASD movement, mouse rotation, numpad, gamepad, touch gestures, speed factors.

---

## 3. Rendering Pipeline

### Graphics Compositor

The top-level rendering orchestrator with inputs for Game renderer, SingleView renderer, RenderStages, RenderFeatures.

### ForwardRenderer

The main forward rendering pipeline:

| Input | Description |
|-------|-------------|
| `Clear` | ClearRenderer for color/depth/stencil |
| `OpaqueRenderStage` | Stage for opaque objects |
| `TransparentRenderStage` | Stage for transparent objects |
| `ShadowMapRenderStages` | Shadow map render stages |
| `PostEffects` | Post-processing effects chain |
| `LightShafts` | Volumetric light shafts |
| `MSAALevel` / `MSAAResolver` | Multisampling configuration |

### Render Stages

Named rendering passes (e.g., "Opaque", "Transparent", "ShadowMap") with configurable sort mode (None/BackToFront/FrontToBack/StateChange) and filter.

### Stage Selectors

| Selector | Routes To |
|----------|-----------|
| `MeshTransparentRenderStageSelector` | Opaque or transparent based on material |
| `ShadowMapRenderStageSelector` | Shadow map stages |
| `SimpleGroupToRenderStageSelector` | By render group |
| `EntityRendererStageSelector` | BeforeScene/Opaque/Transparent/AfterScene/ShadowCaster |

### Root Render Features

| Feature | Purpose |
|---------|---------|
| `MeshRenderFeature` | Core mesh rendering with sub-features |
| `BackgroundRenderFeature` | Skybox rendering |
| `SpriteRenderFeature` | 2D sprite rendering |
| `EntityRendererRenderFeature` | Custom VL renderer integration |
| `UIRenderFeature` | Stride UI rendering |
| `WireframeRenderFeature` | Wireframe overlay |

---

## 4. Screen Spaces and Coordinate Systems

Seven coordinate spaces defined in `ScreenSpaces.cs`:

| Space | Description |
|-------|-------------|
| **World** | Standard 3D world space |
| **View** | Camera-relative |
| **Projection** | Projection-relative |
| **Normalized** | Height -1 to 1, origin center, aspect-corrected |
| **DIP** | 1 unit = 100 DIP, origin center, Y up |
| **DIPTopLeft** | DIP units, origin top-left |
| **PixelTopLeft** | 1 unit = 100 px, origin top-left |

Renderer nodes: `WithinCommonSpace`, `WithinPhysicalScreenSpace`, `WithinVirtualScreenSpace`.

---

## 5. Materials System

### PBR Material Properties

Full physically-based rendering with configurable microfacet BRDF:

| Parameter | Options |
|-----------|---------|
| **Environment** | GGXLUT, GGXPolynomial, ThinGlass |
| **Fresnel** | None, Schlick, ThinGlass |
| **Normal Distribution** | Beckmann, BlinnPhong, GGX |
| **Visibility** | CookTorrance, Implicit, Kelemen, SmithBeckmann, SmithGGXCorrelated, SmithSchlickBeckmann, SmithSchlickGGX |

### Material Generation (`MaterialExtensions.cs`)

Runtime material creation from descriptors:
1. Creates `MaterialGeneratorContext` with device's graphics profile
2. Sets `ShaderGraph.GraphSubscriptions` for ShaderFX lifetime
3. Calls `MaterialGenerator.Generate()` to compile
4. Resolves attached references (environment lighting LUT textures)

### VLMaterialEmissiveFeature

Extended emissive with VL-specific capabilities:
- `VertexAddition` — custom vertex shader code
- `PixelAddition` — custom pixel shader code
- `MaterialExtension` — full ShaderFX graph composition

### FallbackMaterial

Visual feedback during shader compilation:
- Green glowing: shaders compiling
- Red glowing: compilation failed
- Retries every 5 seconds

---

## 6. Lights

### Light Types

| Renderer | Description |
|----------|-------------|
| `LightAmbientRenderer` | Ambient light |
| `LightSkyboxRenderer` | Image-based skybox lighting |
| `LightDirectionalGroupRenderer` | Directional lights |
| `LightPointGroupRenderer` | Point lights |
| `LightSpotGroupRenderer` | Spot lights |
| `LightClusteredPointSpotGroupRenderer` | Clustered rendering for many lights |
| `LightProbeRenderer` | Light probes |

### Shadow Maps

| Renderer | Description |
|----------|-------------|
| `LightDirectionalShadowMapRenderer` | Directional shadows |
| `LightSpotShadowMapRenderer` | Spot shadows |
| `LightPointShadowMapRendererParaboloid` | Point shadows (paraboloid) |
| `LightPointShadowMapRendererCubeMap` | Point shadows (cube map) |

### Skybox System (`LightNodes.Skybox.cs`)

- Takes cube map or 2D panoramic texture
- Computes diffuse pre-filtering (Spherical Harmonics Order 3 or 5)
- Computes specular pre-filtering (GGX radiance)
- `IsSpecularOnly` mode, configurable specular cube map size (default 256)
- Invalidation-based recomputation

---

## 7. Post-Processing Effects

### Available Effects

| Category | Effect | Key Parameters |
|----------|--------|---------------|
| **PostFX** | AmbientOcclusion | Samples, ProjScale, Intensity, Bias, Radius |
| **PostFX** | LocalReflections | MaxSteps, BRDFBias, GlossinessThreshold |
| **PostFX** | DepthOfField | MaxBokehSize, DOFAreas, Technique, AutoFocus |
| **PostFX** | Fog | Density, Color, FogStart, SkipBackground |
| **PostFX** | Outline | NormalWeight, DepthWeight |
| **PostFX** | BrightFilter | Threshold, Steepness |
| **PostFX** | Bloom | Radius, Amount, DownScale, Distortion, Afterimage |
| **PostFX** | LightStreak | Amount, StreakCount, Attenuation, IsAnamorphic |
| **PostFX** | LensFlare | Amount, ColorAberrationStrength, HaloFactor |
| **AA** | FXAAEffect | Anti-aliasing |
| **AA** | TemporalAntiAliasEffect | Temporal AA |
| **Tone** | ToneMap | 9 operators (Reinhard, ACES, U2Filmic, etc.) + auto-exposure |
| **Tone** | FilmGrain | Film grain overlay |
| **Tone** | Vignetting | Amount, Radius, Color |
| **Tone** | Dither | Dithering |

---

## 8. Meshes and Procedural Models

### Built-in Procedural Meshes

| Mesh | Key Parameters |
|------|---------------|
| `CapsuleProceduralModel` | Length, Radius, Tessellation |
| `GeoSphereProceduralModel` | Radius, Tessellation (1–5) |
| `PlaneProceduralModel` | Size, Tessellation, Normal, GenerateBackFace |
| `SphereProceduralModel` | Radius, Tessellation |
| `TeapotProceduralModel` | Size, Tessellation |
| `TorusProceduralModel` | Radius, Thickness, Tessellation |

### Custom g3 Meshes

Extended parametric meshes using geometry3Sharp:

| Mesh | Parameters |
|------|-----------|
| `BoxMesh` | Size (Vector3), Tessellation, Anchor (Top/Middle/Bottom), SharedVertices |
| `ConeMesh` / `CylinderMesh` / `DiscMesh` / `TubeMesh` | Various parametric |
| `ArrowMesh` | Arrow geometry |
| `BoxSphereMesh` | Rounded box |
| `RoundRectangleMesh` | Rectangle with rounded corners |
| `VerticalGeneralizedCylinderMesh` | Profile-based revolution |

---

## 9. Textures and GPU Resources

### Texture System (`GraphicsNodes.cs`)

**TextureDescription:**
| Property | Default |
|----------|---------|
| Width | 512 |
| Height | 512 |
| Format | R8G8B8A8_UNorm_SRgb |
| Usage | Default |
| Flags | ShaderResource |
| Dimension | Texture2D |

**MipMapGenerator** (`MipMapGenerator.cs`): Generates mip maps via shader-based downsampling.

### Buffer System

| Node | Purpose |
|------|---------|
| `BufferDescription` | SizeInBytes, StructureByteStride, BufferFlags |
| `Buffer` (BufferBuilder) | Creates GPU buffers |
| `BufferView` | Views with specific flags and formats |

### Pipeline State

Full GPU pipeline configuration:
- **BlendState** — per-render-target blend (up to 8 targets)
- **RasterizerState** — FillMode (Solid/Wireframe), CullMode (Back), MultisampleCount (X8)
- **DepthStencilState** — DepthBufferEnable, DepthBufferFunction (LessEqual), stencil operations
- **SamplerState** — Filter (Linear), AddressU/V/W (Clamp), MaxAnisotropy (16)

---

## 10. Shader/Effect System

### Architecture (5 layers)

1. **File Discovery:** Scans `shaders/` directories for `.sdsl` files
2. **Metadata Extraction:** Parses shader attributes for node metadata
3. **Node Description:** Creates VL node blueprints with typed pins
4. **Runtime Instantiation:** Compiles shaders, creates live pin bindings
5. **Execution:** Uploads parameters to GPU each frame

### Four Shader Types

| Suffix | Node Type | Purpose |
|--------|-----------|---------|
| `_TextureFX` | TextureFXNode | GPU texture filters, generators, mixers |
| `_DrawFX` | DrawEffectShaderNode | Custom geometry rendering |
| `_ComputeFX` | ComputeEffectShaderNode | GPU compute dispatches |
| `_ShaderFX` | ShaderFXNode | Composable shader graph fragments |

### Shader Metadata Attributes

- **Shader-level:** `[Category]`, `[Summary]`, `[Remarks]`, `[Tags]`, `[OutputFormat]`
- **Pin-level:** `[Summary]`, `[Remarks]`, `[Optional]`, `[EnumType]`, `[Default]`

### Hot-Reload

File watchers monitor `.sdsl` files and transitive dependencies. On change:
1. Compiler cache reset for changed shader
2. Parser cache reset for all watched shaders
3. Node description invalidated and regenerated
4. Live instances updated without restart

### Compute Dispatchers

| Type | Description |
|------|-------------|
| `DirectComputeEffectDispatcher` | Fixed thread group count (Int3) |
| `IndirectComputeEffectDispatcher` | GPU buffer-driven dispatch |
| `CustomComputeEffectDispatcher` | User-defined thread group selector |

---

## 11. ShaderFX Composition System

~50 C# classes in `src/Shaders/ShaderFX/` for GPU shader graph composition:

### Core Types
- `ComputeNode<T>` / `ComputeValue<T>` — typed shader graph nodes
- `ComputeVoid` — side-effect-only nodes
- `ComputeOrder` — sequences multiple operations

### Variables
- `DeclVar` / `GetVar` / `SetVar` — GPU variable lifecycle
- `DeclConstant` / `GetConstant` — compile-time constants
- `DeclSemantic` / `GetSemantic` — shader semantic access (position, normal, UV)
- `InputValue` — CPU-to-GPU value input

### Operations
- `BinaryOperation` / `UnaryOperation` — math operations
- `BlendOperation` — color blending
- `Transform` — matrix transformation
- `Join` / `GetMember` — vector construction/extraction

### Textures and Buffers
- `DeclTexture` / `SampleTexture` / `LoadTexture` — texture operations
- `DeclBuffer` / `GetItemBuffer` / `SetItemBuffer` — buffer operations
- `DeclSampler` — sampler state declaration

### Domain-Specific
- `SDF3D` / `OpSDF` — 3D signed distance fields
- `SF2D` / `OpSF2D` — 2D scalar fields
- `VF3D` — 3D vector fields
- `RaymarcherMatcap1` — matcap ray marcher

---

## 12. TextureFX Library

80+ GPU texture effects in `VL.Stride.TextureFX/shaders/`:

### Filters (~60 effects)

| Category | Effects |
|----------|---------|
| **Blur** | BlurPass, BlurPerfector, DirectionalBlur |
| **Color** | Anaglyph, ConvertColor, Dither, Grayscale, HSCB, Invert, Levels, Posterize, Sharpen, Threshold |
| **Keying** | ChannelKeying, ChromaKey, LumaKey |
| **Depth** | FilterDepth, LinearDepth |
| **Distortion** | Bump, CartesianToPolar, Displace, Magnify, Transform, Tunnels, Undistort |
| **Glow** | BlurGlow, GlowPre/Main/Mix (multi-pass) |
| **Morphology** | Dilate_Color, Erode_Color, UVDilate |
| **Patterns** | CrossStitch, Dots, Hatch, Kaleidoscope, Mosaic, Scanline |

### Mixers (~6 effects)
BlendMixer, Blood, Dissolve, Pixelate, Ripple

### Sources (~12 effects)
BubbleNoise, Checkerboard, Color, ColorPalette, Electricity, Gradient, Halo, Mandelbrot, Neurons, Noise

### Example Shader

```hlsl
[Category("Utils")]
[Summary("Converts a texture from rgb to grayscale")]
[Tags("convert, desaturate")]
[OutputFormat("R32_Float")]
shader Grayscale_TextureFX : FilterBase
{
    float3 Graymix = float3(0.299f, 0.587f, 0.114f);
    float4 Filter(float4 tex0col)
    {
        return float4(dot(tex0col.rgb, Graymix).rrr, tex0col.a);
    }
};
```

---

## 13. Physics System

**Category:** `Stride.Experimental.Physics` (Bullet physics integration)

### Physics Components

| Component | Type | Key Properties |
|-----------|------|---------------|
| `StaticColliderComponent` | Immovable | ColliderShapes, Restitution (0.5), Friction (0.1) |
| `DynamicColliderComponent` | Physics-driven | Mass (1.0), LinearDamping, AngularDamping |
| `KinematicColliderComponent` | User-driven | Same as Dynamic (isKinematic=true) |
| `TriggerColliderComponent` | Volume trigger | Same as Dynamic (isTrigger=true) |

### Collider Shapes

| Shape | Key Parameters |
|-------|---------------|
| `BoxColliderShapeDesc` | Size (Vector3.One), LocalOffset |
| `SphereColliderShapeDesc` | Radius (0.5) |
| `CapsuleColliderShapeDesc` | Radius (0.5), Length (0.5), Orientation |
| `ConeColliderShapeDesc` | Radius (0.5), Height (1.0) |
| `CylinderColliderShapeDesc` | Radius (0.5), Height (1.0) |
| `StaticPlaneColliderShapeDesc` | Normal (UnitY), Offset (0) |

---

## 14. Virtual Reality

**Category:** `Stride.Experimental.VirtualReality`

### VRDevice Node Outputs

| Output | Type | Description |
|--------|------|-------------|
| `State` | DeviceState | Current VR state |
| `LeftHand` / `RightHand` | TouchController | Hand controllers |
| `HeadPosition` | Vector3 | Head position |
| `HeadRotation` | Quaternion | Head rotation |
| `TrackedItems` | Spread<TrackedItem> | All tracked devices |
| `MirrorTexture` | Texture | Monitor mirror |

### TouchController Data

Position, Rotation, LinearVelocity, AngularVelocity, Trigger (float), Grip (float), ThumbstickAxis (Vector2), IndexPointing, ThumbUp.

---

## 15. Window Management and Game Loop

### VLGame (`VLGame.cs`)

Extends Stride's `Game` class:
- Linear color space by default
- Custom clock (`RawTickProducer`) using `ElapsedUserTime` for VL time control
- Automatic shader path scanning on every Update
- Deferred present for correct swap chain ordering
- RenderDoc frame capture support
- `SchedulerSystem` for deferred GPU work

### GameWindowRenderer (`GameWindowRenderer.cs`)

Individual render window management:
- `SwapChainGraphicsPresenter` with configurable format, multisampling, present interval
- Priority-based input source registration
- Dark title bar styling
- Custom title bar rendering support

---

## 16. Input System

### Input Nodes (`InputNodes.cs`)

Null-safe wrappers:
- `IsKeyPressed` / `IsKeyReleased` / `IsKeyDown`
- `IsButtonPressed` / `IsButtonReleased` / `IsButtonDown`

All return `false` if device is null (safe for disconnected devices).

### Input Sources

Each `GameWindowRenderer` creates its own `IInputSource` with priority. The `InputManager` aggregates all sources. Two component types route input to the scene graph:
- `InputSourceComponent` — generic entity input
- `CameraInputSourceComponent` — screen-to-world coordinate mapping

---

## 17. Skia/Stride Integration

`SkiaRenderer.cs` bridges 2D Skia rendering with 3D Stride:
- Takes a Skia `ILayer` and `CommonSpace`
- Uses EGL/ANGLE to create OpenGL ES surface backed by Stride's D3D11 texture
- Creates `SKSurface` around the EGL surface
- Handles color space conversion (linear mode support)
- Routes Stride window input to Skia's input system
- Applies CommonSpace coordinate transformation

---

## 18. Script System

`InterfaceSyncScript.cs` bridges VL patches with Stride's script system:

```csharp
public interface ISyncScript
{
    void Start(SyncScript script);
    void ScriptUpdate();
    void Cancel();
}
```

Gives VL patches access to Entity, Scene, Services, and the full Stride game loop via the `InterfaceSyncScript` (exposed as "PatchScriptComponent").

---

## Help Patches

30+ help patches organized by topic:

| Category | Topics |
|----------|--------|
| **Overview** | Scene Graph Basics/Advanced, Transformations, Write a Shader |
| **Models** | 2D/3D Primitives, Mesh Primitives, Instancing, Split Mesh |
| **Materials** | PBR Overview, Diffuse, Metalness, Roughness, Normal Map, Emission, Layers |
| **Lights** | Overview, Projected Textures, Volumetric Light Shafts |
| **Cameras** | Handling, Aspect Ratio, Look-At, Head-Tracking |
| **Input** | Mouse/Keyboard/Touch, Ray Intersection (picking) |
| **Rendering** | Buffers/Textures, VR (OpenXR), Skia/Stride Overlay |
| **Textures** | Save, Treeback (feedback), Read Pixels, Load/Unload |
| **PostFX** | All post-effects overview |
| **TextureFX** | 65+ individual effect help patches |

---

## VL.Stride vs VL.Skia

| Dimension | VL.Skia (2D) | VL.Stride (3D) |
|-----------|--------------|----------------|
| **Engine** | SkiaSharp (Google Skia) | Stride Game Engine |
| **Graphics API** | OpenGL ES via ANGLE | Direct3D 11 |
| **Scene Model** | Flat layer stack (ILayer) | Hierarchical Entity-Component System |
| **Camera** | Implicit 2D viewport | Explicit CameraComponent |
| **Lighting** | None | Full PBR + shadows + IBL |
| **Shaders** | SkSL effects | SDSL/HLSL with auto-generated nodes |
| **Physics** | Not built-in | Bullet physics |
| **VR** | Not supported | OpenXR |
| **Compute** | Not available | ComputeFX |
| **Color Space** | sRGB | Linear with sRGB conversion |
| **Integration** | Standalone 2D | Can embed Skia 2D layers |
