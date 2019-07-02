#pragma once
#include <curand.h>
#include <curand_kernel.h>

__global__ void init_rand_state(curandState *state)
{
    curand_init(1, threadIdx.x, 0, &state[threadIdx.x]);
}

__device__ float rand_uni_range(curandState* state, float a, float b)
{
    float randf = curand_uniform(&state[threadIdx.x]);
    return randf*(b-a) + a;
}