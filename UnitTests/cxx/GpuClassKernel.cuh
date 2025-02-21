//
// Created by mbahassan on 2/21/25.
//

#ifndef GPUCLASSKERNEL_CUH
#define GPUCLASSKERNEL_CUH

__global__
void kernel(const Particle* devParticle, const int size)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        printf("particle id: %d\n", devParticle[idx].getId());
    }
}

#endif //GPUCLASSKERNEL_CUH
