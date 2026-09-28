#include <cstdio>
#include <cstdlib>
#include <chrono>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Simulation.h"
#include "RenderStateGenerator.h"
#include "InteropBuffer.h"
#include "Renderer.h"
#include "MPIManager.h"
#include "RenderState.h"

namespace
{
    GLFWwindow* initWindowAndGLContext(int width, int height, const char* title)
    {
        if (!glfwInit())
        {
            std::fprintf(stderr, "Failed to init GLFW\n");
            std::exit(EXIT_FAILURE);
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window)
        {
            std::fprintf(stderr, "Failed to create window\n");
            glfwTerminate();
            std::exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);

        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        {
            std::fprintf(stderr, "Failed to init GLAD\n");
            std::exit(EXIT_FAILURE);
        }

        glViewport(0, 0, width, height);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);

        // Pin the CUDA context to the same device driving this GL context.
        // Required for cudaGraphicsGLRegisterBuffer to succeed on multi-GPU
        // machines; a no-op on single-GPU machines.

        return window;
    }

    float computeDeltaTime()
    {
        static auto last = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        return dt;
    }
}

int main()
{
    constexpr int kParticleCount = 100;


    // --- init ---
    GLFWwindow* window = initWindowAndGLContext(2000, 2000, "particle-sim");

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

    MPIManager mpi;   // no-op for MVP; kept so this loop's shape is stable
    mpi.initialize();

    // --- loop ---
    // Physics::step() -> RenderState::generate() -> Renderer::render() -> Present
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        float dt = computeDeltaTime();
        if (dt > 0.1f) dt = 0.1f;   // clamp on stalls (breakpoints, window drag)

        simulation.step(dt, stream);

        RenderState renderState{};
        renderState.count    = kParticleCount;
        renderState.position = interop.mapForCuda(stream);

        renderStateGen.generate(simulation.getState(), renderState, stream);

        interop.unmapFromCuda(stream);
        cudaStreamSynchronize(stream);   // only explicit sync: CUDA -> OpenGL handoff

        renderer.render(interop.glBuffer(), kParticleCount);

        glfwSwapBuffers(window);
    }

    // --- shutdown ---
    mpi.shutdown();
    renderer.shutdown();
    interop.shutdown();
    simulation.shutdown();
    cudaStreamDestroy(stream);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
