//
// Created by iqraa on 21-2-25.
//

#include "Base.cuh"
#include <cstdio>

template<typename ParticleType>
Base<ParticleType>::Base()
{
    int deviceId = 0;
    cudaGetDeviceProperties(&deviceProperties, deviceId);

    threadsPerBlock = deviceProperties.maxThreadsPerBlock;
}

template<typename ParticleType>
void Base<ParticleType>::printDeviceInfo() const
{
    printf("Warp size:                                     %d\n",
       deviceProperties.warpSize);
    printf("Maximum number of threads per multiprocessor:  %d\n",
           deviceProperties.maxThreadsPerMultiProcessor);
    printf("Maximum number of threads per block:           %d\n",
           deviceProperties.maxThreadsPerBlock);
    printf("Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProperties.maxThreadsDim[0], deviceProperties.maxThreadsDim[1],
           deviceProperties.maxThreadsDim[2]);
    printf("Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n\n",
           deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1],
           deviceProperties.maxGridSize[2]);
}
