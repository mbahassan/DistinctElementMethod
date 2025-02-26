
// includes, system
#include <iostream>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector>

#include "GpuClass.cuh"


int main(int argc, char **argv)
{
    int N = 10;
    std::vector<Particle> particle(N);

    for(int i = 0; i < N; i++)
    {
        particle[i].setId(i);
    }

    const Particle* devParticle =  particle.data();
    GpuClass gpu(devParticle, N);
    gpu.printHellow();

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    // exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}