#pragma once
#include <vector_types.h>  // float2

// Physics-owned state. Renderer never touches this directly.
struct SimulationState
{
    float2* position     = nullptr;
    float2* velocity      = nullptr;
    float2* acceleration = nullptr;
    float*  mass         = nullptr;

    int count = 0;
};
