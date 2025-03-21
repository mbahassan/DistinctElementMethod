//
// Created by mbahassan on 2/21/25.
//

#ifndef GPUCLASSKERNEL_CUH
#define GPUCLASSKERNEL_CUH

__global__
inline void kernel(Spherical* devParticle, const int size)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        const Spherical& pi = devParticle[idx];
        printf("particle id:%d pos: (%f %f %f)\n",pi.getId(), pi.position.x,
            pi.position.y, pi.position.z);
    }
}

#endif //GPUCLASSKERNEL_CUH
