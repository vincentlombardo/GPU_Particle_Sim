#include "Simulation.h"
#include <vector>
#include <cstdlib>

namespace
{
    constexpr int kThreadsPerBlock = 256;

    __global__ void integrate(float2* pos, float2* vel, const float2* acc, float dt, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        vel[i].x += acc[i].x * dt;
        vel[i].y += acc[i].y * dt;

        pos[i].x += vel[i].x * dt;
        pos[i].y += vel[i].y * dt;
    }

    __global__ void solveBoundaries(float2* pos, float2* vel, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        if (pos[i].x < -1.f) { pos[i].x = -1.f; vel[i].x *= -1.f; }
        if (pos[i].x >  1.f) { pos[i].x =  1.f; vel[i].x *= -1.f; }
        if (pos[i].y < -1.f) { pos[i].y = -1.f; vel[i].y *= -1.f; }
        if (pos[i].y >  1.f) { pos[i].y =  1.f; vel[i].y *= -1.f; }
    }
}

void Simulation::initialize(int particleCount)
{
    state.count = particleCount;

    cudaMalloc(&state.position,     particleCount * sizeof(float2));
    cudaMalloc(&state.velocity,     particleCount * sizeof(float2));
    cudaMalloc(&state.acceleration, particleCount * sizeof(float2));
    cudaMalloc(&state.mass,         particleCount * sizeof(float));

    // Seed initial conditions on the host, then upload once.
    std::vector<float2> hPos(particleCount);
    std::vector<float2> hVel(particleCount);
    std::vector<float2> hAcc(particleCount);
    std::vector<float>  hMass(particleCount);

    for (int i = 0; i < particleCount; ++i)
    {
        hPos[i]  = make_float2((std::rand() / (float)RAND_MAX) * 2.f - 1.f,
                                (std::rand() / (float)RAND_MAX) * 2.f - 1.f);
        hVel[i]  = make_float2((std::rand() / (float)RAND_MAX) * 0.4f - 0.2f,
                                (std::rand() / (float)RAND_MAX) * 0.4f - 0.2f);
        hAcc[i]  = make_float2(0.f, -0.5f);
        hMass[i] = 1.f;
    }

    cudaMemcpy(state.position,     hPos.data(),  particleCount * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(state.velocity,     hVel.data(),  particleCount * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(state.acceleration, hAcc.data(),  particleCount * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(state.mass,         hMass.data(), particleCount * sizeof(float),  cudaMemcpyHostToDevice);
}

void Simulation::step(float dt, cudaStream_t stream)
{
    int blocks = (state.count + kThreadsPerBlock - 1) / kThreadsPerBlock;

    integrate<<<blocks, kThreadsPerBlock, 0, stream>>>(
        state.position, state.velocity, state.acceleration, dt, state.count);

    solveBoundaries<<<blocks, kThreadsPerBlock, 0, stream>>>(
        state.position, state.velocity, state.count);
}

void Simulation::shutdown()
{
    cudaFree(state.position);
    cudaFree(state.velocity);
    cudaFree(state.acceleration);
    cudaFree(state.mass);
    state = SimulationState{};
}
