
// includes, system
#include <iostream>

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector>
#include <Particle/Shape/Sphere/Sphere.hpp>

#include "GpuClass.cuh"


int main(int argc, char **argv)
{
    int N = 10;
    Sphere sphere(0.1);

    Material glass;
    glass.setName("glass");

    std::vector<Particle> particle(N);
    for(int i = 0; i < N; i++)
    {
        particle[i] = Particle(glass, sphere);
        particle[i].setId(i);
        particle[i].setRadius(i+0.1f);
    }

    std::cout << "particle radius: "<< particle[0].getRadius() << std::endl;
    std::cout << "particle Material: "<< particle[0].getName() << std::endl;
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