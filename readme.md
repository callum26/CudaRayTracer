# Real Time Ray Tracing Renderer

Callum Small - 100463689

A real-time GPU ray tracer built using CUDA and OpenGL.
67+ FPS at 800×800 on a 1000-primitive scene using a bucketed SAH Bounding Volume Hierarchy.

---

## Requirements

| Dependency    | Version           |
| ------------- | ----------------- |
| NVIDIA GPU    | (RTX recommended) |
| CUDA Toolkit  | 13.1              |
| CMake         | 3.18+             |
| Visual Studio | 2022              |
| Windows SDK   | 10.0.22621.0      |
| OpenGL        | 4.6+              |

GLFW 3.4 and GLAD are bundled under `import/` — no separate installation needed.

---

## Build and Run

Double-click `build.bat` or run it from a terminal:

```bat
build.bat
```

This will:

1. Create a `build/` directory
2. Run CMake with Visual Studio 2022
3. Compile in Release mode
4. Launch `build/Release/rayTracer.exe`

To run the executable again without rebuilding:

```bat
build\Release\rayTracer.exe
```

---

## Controls

| Key | Action                                                          |
| --- | --------------------------------------------------------------- |
| `B` | Toggle between BVH and brute-force intersection                 |
| `T` | Swap scene between Cornell box and sphere grid performance test |

Frame rate, GPU kernel time, and rendering mode are shown in the window title bar.
They update every 0.5 seconds.

---

## Project Structure

```
CudaRayTracer/
├── main.cpp            Host: OpenGL context, frame loop, user input
├── kernel.cu           Device: ray generation, BVH traversal, shading, tone mapping
├── import/
│   ├── structs.h       Shared types: Vec3, Ray, Object, BVHNode
│   ├── bvh.cuh         Host SAH BVH construction
│   ├── raytracer.h     Host/device interface (initScene, launchRayTracer, etc.)
│   ├── glad/           OpenGL function loader (included)
│   └── glfw-3.4/       Window and input library (included)
├── shaders/
│   ├── vertex.glsl     Screen-aligned quad vertex shader
│   └── fragment.glsl   Texture display fragment shader
├── CMakeLists.txt      Build configuration
└── build.bat           One-step build and run script
```

---

## Features

- Bucketed SAH BVH — O(N log N) build, ~O(log N) traversal
- Phong direct illumination with area light shadow sampling
- Cosine-weighted hemisphere sampling for indirect illumination
- Schlick's Fresnel approximation at dielectric boundaries
- Beer-Lambert absorption for coloured transparent objects
- Russian roulette bounce termination
- Progressive temporal accumulation
- ACES filmic tone mapping
- CUDA/OpenGL interop via PBO
