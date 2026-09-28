#include "RenderStateGenerator.h"

namespace
{
    constexpr int kThreadsPerBlock = 256;

    // MVP: a straight copy. This is intentionally the simplest possible
    // kernel -- later this is where interpolation, culling, LOD, and
    // transform generation get added without touching Simulation or Renderer.
    __global__ void generateRenderStateKernel(const float2* simPos, float2* outPos, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        outPos[i] = simPos[i];
    }
}

void RenderStateGenerator::generate(const SimulationState& sim, RenderState& out, cudaStream_t stream)
{
    int blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    generateRenderStateKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        sim.position, out.position, count);
}
