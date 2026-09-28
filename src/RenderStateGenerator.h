#pragma once
#include <cuda_runtime.h>
#include "SimulationState.h"
#include "RenderState.h"

// Converts SimulationState into RenderState. This is the only bridge
// between the physics domain and the rendering domain.
class RenderStateGenerator
{
public:
    void initialize(int particleCount) { count = particleCount; }
    void generate(const SimulationState& sim, RenderState& out, cudaStream_t stream);

private:
    int count = 0;
};
