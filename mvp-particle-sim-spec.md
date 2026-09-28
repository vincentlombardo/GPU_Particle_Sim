# MVP Technical Spec: CUDA Particle Simulation + OpenGL Renderer

## 1. Scope

Single-process, single-GPU particle simulator. Physics runs in CUDA, produces a
`SimulationState`, a CUDA kernel converts that into a `RenderState`, and OpenGL
draws the `RenderState` via CUDA/OpenGL interop (no host-side copy).

**In scope:** physics step, render-state step, render step, single CUDA stream,
single-buffered render state, single-file `main.cpp` entry point.

**Explicitly out of scope for MVP** (architected for, not built):
- MPI / multi-GPU distribution
- Double buffering
- Frustum culling, LOD, mesh/material selection
- Rigid bodies, collisions, constraints

---

## 2. Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| CUDA Toolkit | 12.x | Physics + render-state kernels |
| OpenGL | 4.5+ core profile | Rasterization |
| GLFW | 3.3+ | Window + input |
| GLAD (or GLEW) | latest | OpenGL function loading |
| CMake | 3.24+ | Build (needed for first-class `CUDA` language support + interop) |
| A CUDA-capable GPU | Compute Capability 6.0+ | Required for `cudaGraphicsGLRegisterBuffer` |
| C++ standard | C++17 | Host code |

No math library dependency needed for MVP (`float2` from CUDA vector types is
sufficient). No MPI dependency yet — `MPIManager` is a stub class only.

---

## 3. Project Structure

Single-file startup: everything needed to run boots from `main.cpp`. Supporting
types live in small headers so `main.cpp` stays a thin orchestrator, not a dumping
ground — but there is exactly one `.cpp`/`.cu` compiled unit per component and one
entry point.

```
particle-sim/
├── CMakeLists.txt
├── src/
│   ├── main.cpp                  # single entry point — owns the frame loop
│   ├── SimulationState.h         # POD struct, device pointers
│   ├── RenderState.h             # POD struct, device pointers
│   ├── Simulation.h / .cu        # owns SimulationState, physics kernels
│   ├── RenderStateGenerator.h/.cu# SimulationState -> RenderState kernel
│   ├── Renderer.h / .cpp         # OpenGL: VBO, shaders, draw call
│   ├── InteropBuffer.h / .cpp    # CUDA/OpenGL shared buffer wrapper
│   └── MPIManager.h              # empty stub, no-op for MVP
└── shaders/
    ├── particle.vert
    └── particle.frag
```

`main.cpp` is the only file that knows about the full pipeline order. Every other
class knows only its own inputs/outputs.

---

## 4. Data Structures

```cpp
// SimulationState.h
struct SimulationState
{
    float2* position;
    float2* velocity;
    float2* acceleration;
    float*  mass;
    int     count;
};
```

```cpp
// RenderState.h
struct RenderState
{
    float2* position;   // CUDA/OpenGL interop-mapped pointer
    int     count;
};
```

Both are POD device-pointer bundles — no ownership logic lives in the structs
themselves; owning classes allocate/free them.

---

## 5. Core Classes

### 5.1 Simulation (owns `SimulationState`)

```cpp
class Simulation
{
public:
    void initialize(int particleCount);
    void step(float dt, cudaStream_t stream);
    const SimulationState& getState() const { return state; }
    void shutdown();

private:
    SimulationState state{};
};
```

`step()` launches `integrate<<<>>>` then `solveBoundaries<<<>>>` on the given
stream. No OpenGL symbols anywhere in this file.

### 5.2 InteropBuffer (owns the shared CUDA/OpenGL memory)

```cpp
class InteropBuffer
{
public:
    void initialize(int particleCount);         // creates GL VBO + cudaGraphicsGLRegisterBuffer
    float2* mapForCuda(cudaStream_t stream);     // cudaGraphicsMapResources -> device ptr
    void unmapFromCuda(cudaStream_t stream);     // cudaGraphicsUnmapResources
    GLuint glBuffer() const { return vbo; }
    void shutdown();

private:
    GLuint vbo = 0;
    cudaGraphicsResource_t cudaResource = nullptr;
};
```

This is the object that makes RenderState *be* the OpenGL buffer rather than a
copy of it (see spec doc section 11).

### 5.3 RenderStateGenerator

```cpp
class RenderStateGenerator
{
public:
    void initialize(int particleCount) { count = particleCount; }
    void generate(const SimulationState& sim, RenderState& out, cudaStream_t stream);

private:
    int count = 0;
};
```

`generate()` launches a kernel writing `sim.position -> out.position`, where
`out.position` is the mapped interop pointer from `InteropBuffer::mapForCuda`.

### 5.4 Renderer

```cpp
class Renderer
{
public:
    void initialize();                       // compile/link shaders, VAO setup
    void render(GLuint positionVBO, int count);
    void shutdown();

private:
    GLuint program = 0;
    GLuint vao = 0;
};
```

Knows nothing about `SimulationState`, CUDA, or MPI — only a GL buffer handle
and a count.

### 5.5 MPIManager (stub)

```cpp
class MPIManager
{
public:
    void initialize() {}
    void shutdown() {}
    // exchangeRenderState() intentionally omitted until multi-GPU milestone
};
```

Included now purely so `main.cpp`'s shape doesn't change when MPI is added later.

---

## 6. Frame Loop (`main.cpp`)

```cpp
int main()
{
    constexpr int kParticleCount = 100'000;

    // --- init ---
    GLFWwindow* window = initWindowAndGLContext(1280, 720, "particle-sim");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    Simulation simulation;
    simulation.initialize(kParticleCount);

    InteropBuffer interop;
    interop.initialize(kParticleCount);

    RenderStateGenerator renderStateGen;
    renderStateGen.initialize(kParticleCount);

    Renderer renderer;
    renderer.initialize();

    MPIManager mpi;   // no-op for MVP
    mpi.initialize();

    // --- loop ---
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        float dt = computeDeltaTime();

        simulation.step(dt, stream);

        RenderState renderState{};
        renderState.count = kParticleCount;
        renderState.position = interop.mapForCuda(stream);

        renderStateGen.generate(simulation.getState(), renderState, stream);

        interop.unmapFromCuda(stream);   // signals CUDA is done writing
        cudaStreamSynchronize(stream);   // ensure GL sees finished writes

        renderer.render(interop.glBuffer(), kParticleCount);

        glfwSwapBuffers(window);
    }

    // --- shutdown ---
    mpi.shutdown();
    renderer.shutdown();
    interop.shutdown();
    simulation.shutdown();
    cudaStreamDestroy(stream);
    glfwTerminate();
    return 0;
}
```

This mirrors the pipeline from the design doc exactly:

```
Physics::step() -> RenderState::generate() -> Renderer::render() -> Present
```

All kernels run on a single `cudaStream_t`, so ordering between the physics
kernel and the render-state kernel is guaranteed by CUDA — no explicit event
is needed between them. The only explicit sync point is the CUDA→OpenGL
handoff (`unmapFromCuda` + `cudaStreamSynchronize`).

---

## 7. Kernels

```cu
// Simulation.cu
__global__ void integrate(float2* pos, float2* vel, const float2* acc, float dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vel[i] += acc[i] * dt;
    pos[i] += vel[i] * dt;
}

__global__ void solveBoundaries(float2* pos, float2* vel, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (pos[i].x < -1.f || pos[i].x > 1.f) vel[i].x *= -1.f;
    if (pos[i].y < -1.f || pos[i].y > 1.f) vel[i].y *= -1.f;
}
```

```cu
// RenderStateGenerator.cu
__global__ void generateRenderState(const float2* simPos, float2* outPos, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    outPos[i] = simPos[i];
}
```

---

## 8. Shaders (minimal)

`particle.vert`
```glsl
#version 450 core
layout(location = 0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); gl_PointSize = 3.0; }
```

`particle.frag`
```glsl
#version 450 core
out vec4 fragColor;
void main() { fragColor = vec4(1.0); }
```

---

## 9. CMakeLists.txt (essentials)

```cmake
cmake_minimum_required(VERSION 3.24)
project(particle-sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(particle-sim
    src/main.cpp
    src/Simulation.cu
    src/RenderStateGenerator.cu
    src/Renderer.cpp
    src/InteropBuffer.cpp
)

target_link_libraries(particle-sim PRIVATE OpenGL::GL glfw)
set_target_properties(particle-sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

(GLAD sources added as a normal `.c` compiled into the same target; omitted here
for brevity.)

---

## 10. MVP Acceptance Criteria

- [ ] Window opens, N particles render as points on screen.
- [ ] Particles move under constant acceleration and bounce off a `[-1, 1]²` box.
- [ ] No host-side (CPU) copy of position data occurs anywhere in the frame loop — CUDA writes directly into the mapped GL buffer.
- [ ] `Simulation`, `RenderStateGenerator`, and `Renderer` compile with zero cross-includes of each other's internals (only `SimulationState.h` / `RenderState.h` are shared).
- [ ] Swapping `Renderer` for a stub that does nothing does not require touching `Simulation` or `RenderStateGenerator`.
- [ ] `MPIManager` stub compiles and is called (no-op) so the shape of `main.cpp` doesn't change when multi-GPU work starts.

---

## 11. Deferred Work (post-MVP, in priority order)

1. Double-buffer `RenderState` to overlap compute and render.
2. Expand `RenderState` to `{ position, rotation, meshId, materialId, visible }`.
3. Implement `MPIManager::exchangeRenderState()` and split `SimulationState` by domain across ranks.
4. Move render-state generation from "position copy" to real work: culling, LOD, transform generation.
