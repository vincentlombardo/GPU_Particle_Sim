#include "InteropBuffer.h"
#include <cstdio>
#include <cstdlib>

void InteropBuffer::initialize(int particleCount)
{
    count = particleCount;

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(float2), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &cudaResource, vbo, cudaGraphicsRegisterFlagsWriteDiscard);

    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

float2* InteropBuffer::mapForCuda(cudaStream_t stream)
{
    cudaGraphicsMapResources(1, &cudaResource, stream);

    float2* devPtr = nullptr;
    size_t numBytes = 0;
    cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&devPtr), &numBytes, cudaResource);

    return devPtr;
}

void InteropBuffer::unmapFromCuda(cudaStream_t stream)
{
    cudaGraphicsUnmapResources(1, &cudaResource, stream);
}

void InteropBuffer::shutdown()
{
    if (cudaResource) cudaGraphicsUnregisterResource(cudaResource);
    if (vbo) glDeleteBuffers(1, &vbo);
    cudaResource = nullptr;
    vbo = 0;
}
