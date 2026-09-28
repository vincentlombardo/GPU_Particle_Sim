#pragma once
#include <cuda_runtime.h>
#include "SimulationState.h"

// Owns SimulationState. Knows nothing about OpenGL, rendering, or MPI.
class Simulation
{
public:
    void initialize(int particleCount);
    void step(float dt, cudaStream_t stream);
    void shutdown();

    const SimulationState& getState() const { return state; }

private:
    SimulationState state{};
};
