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
        printf("particle id: %f %f %f\n", devParticle[idx].position.x,
            devParticle[idx].position.y, devParticle[idx].position.z);
    }
}

#endif //GPUCLASSKERNEL_CUH
