# particle-sim (MVP)

CUDA physics -> CUDA render-state generation -> OpenGL draw, with zero
host-side copies of position data. See `mvp-particle-sim-spec.md` for the
full design rationale.

## Prerequisites

- CUDA Toolkit 12.x, NVIDIA driver, CUDA-capable GPU (compute capability 6.0+)
- CMake 3.24+
- A C++17 compiler
- GLFW3 (dev package, e.g. `libglfw3-dev` on Ubuntu)
- GLAD, generated once by hand (not vendored here — see below)

## One-time setup: GLAD

This project loads OpenGL function pointers via GLAD, which is a generated
loader rather than a library you can `apt install`. Generate it once:

1. Go to https://glad.dav1d.de
2. Language: C/C++, Specification: OpenGL, API gl: Version 4.5, Profile: Core
3. Generate a loader, download the zip
4. Unzip so you end up with:
   ```
   particle-sim/third_party/glad/include/glad/glad.h
   particle-sim/third_party/glad/include/KHR/khrplatform.h
   particle-sim/third_party/glad/src/glad.c
   ```

## Build

```bash
cmake -B build -S .
cmake --build build -j
```

## Run

```bash
./build/particle-sim
```

A window opens showing 100,000 points bouncing inside a `[-1, 1]²` box under
constant downward acceleration.

## File map

| File | Role |
|---|---|
| `src/main.cpp` | Single entry point — owns the frame loop, nothing else |
| `src/SimulationState.h` | Physics-owned POD data |
| `src/RenderState.h` | Render-owned POD data (points into the interop buffer) |
| `src/Simulation.h/.cu` | Physics kernels + state ownership |
| `src/RenderStateGenerator.h/.cu` | SimulationState -> RenderState kernel |
| `src/InteropBuffer.h/.cpp` | The shared CUDA/OpenGL VBO |
| `src/Renderer.h/.cpp` | Shaders, VAO, draw call — no CUDA symbols |
| `src/MPIManager.h` | No-op stub, kept for interface stability |
| `shaders/*.vert,.frag` | Minimal point-sprite shaders |
