#pragma once

// Intentionally empty for the MVP. Exists only so main.cpp's shape
// doesn't change when multi-GPU / MPI work begins -- see spec section 21.
class MPIManager
{
public:
    void initialize() {}
    void shutdown() {}

    // void exchangeRenderState();  // added at the multi-GPU milestone
};
