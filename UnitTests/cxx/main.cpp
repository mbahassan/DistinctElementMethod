
// includes, system


// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector>
#include <ContactDetection/ContactDetection.cuh>
#include <Particle/Shape/Sphere/Sphere.hpp>
#include <Tools/Config/Parser.h>
#include "GpuClass.cuh"
#include "Insertion/InsertionCpu.h"


int main(int argc, char **argv)
{
    auto config = Parser::getConfig("particle.json");

    int N = config.numberOfParticles;
    Sphere sphere(0.1);

    Material glass(config.materialConfigPath);

    std::vector<Particle> particles(N);
    for(int i = 0; i < N; i++)
    {
        particles[i] = Particle(glass, sphere);
        particles[i].setId(i);
        particles[i].setRadius(i+0.1f);
    }

    Insertion insertion;
    insertion.fillGrid(particles, {0,0,0}, {1,1,1},0.2);

    ContactDetection contactDetection(QUADTREE);
    auto potential_pairs = contactDetection.braodPhase(particles);
    actual_contacts = contactDetection.narrowPhase(potential_pairs);

    std::cout << "particle radius: "<< particles[0].getRadius() << std::endl;
    std::cout << "particle Material: "<< particles[0].getMaterialName() << std::endl;
    const Particle* devParticle =  particles.data();
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
