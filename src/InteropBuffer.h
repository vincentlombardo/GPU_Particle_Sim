#pragma once
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>  // float2

// Owns a single GL VBO that is also mapped directly into CUDA.
// This is what makes RenderState *be* the OpenGL buffer rather than
// a copy of it -- there is exactly one allocation for position data
// downstream of the physics step.
class InteropBuffer
{
public:
    void initialize(int particleCount);
    float2* mapForCuda(cudaStream_t stream);
    void unmapFromCuda(cudaStream_t stream);
    GLuint glBuffer() const { return vbo; }
    void shutdown();

private:
    GLuint vbo = 0;
    cudaGraphicsResource_t cudaResource = nullptr;
    int count = 0;
};
