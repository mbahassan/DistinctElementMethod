
// includes, system


// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector>
#include <ContactDetection/ContactDetection.cuh>
#include <Output/Output.cuh>
#include <Particle/Shape/Sphere/Sphere.hpp>
#include <Tools/Config/Parser.h>
#include "GpuClass.cuh"
#include "Insertion/InsertionCpu.h"
#include <Particle/Material/ConfigMaterial.h>

int main(int argc, char **argv)
{
    auto config = Parser::getConfig("particle.json");

    int N = config.numberOfParticles;

    Sphere sphere(0.1);

    Material glass(config.materialConfigPath);

    std::vector<Particle> particles(11);
    for(int i = 0; i < 11; i++)
    {
        particles[i] = Particle(glass, sphere);
        particles[i].setId(i);
        particles[i].setRadius(0.03f);
    }

    // Insertion insertion;
    // insertion.fillGrid2D(particles, {0,0,0}, {1,1,0},0.2);

    // Top-left quadrant (x in [0, 0.5], y in [0.5, 1]): 4 particles -> should subdivide.
    particles[0].position = {0.10f, 0.90f, 0.0f};
    particles[1].position = {0.20f, 0.80f, 0.0f};
    particles[2].position = {0.30f, 0.70f, 0.0f};
    particles[3].position = {0.40f, 0.60f, 0.0f};

    // Top-right quadrant (x in [0.5, 1], y in [0.5, 1]): 3 particles -> no subdivision.
    particles[4].position = {0.60f, 0.90f, 0.0f};
    particles[5].position = {0.70f, 0.80f, 0.0f};
    particles[6].position = {0.80f, 0.70f, 0.0f};

    // Bottom-left quadrant (x in [0, 0.5], y in [0, 0.5]): 3 particles -> no subdivision.
    particles[7].position = {0.10f, 0.10f, 0.0f};
    particles[8].position = {0.20f, 0.20f, 0.0f};
    particles[9].position = {0.30f, 0.30f, 0.0f};

    // Bottom-right quadrant (x in [0.5, 1], y in [0, 0.5]): 2 particles -> no subdivision.
    particles[10].position = {0.60f, 0.10f, 0.0f};
    particles[11].position = {0.70f, 0.20f, 0.0f};



    ContactDetection contactDetection(QUADTREE);
    auto potential_pairs = contactDetection.broadPhase(particles);
    // actual_contacts = contactDetection.narrowPhase(potential_pairs);

    Output output("results");
    output.writeParticles(particles, 0);
    output.writeTree(contactDetection.getTree(),0);

    std::cout << "particle radius: "<< particles[0].getRadius() << std::endl;
    std::cout << "particle Material id: "<< particles[0].getMaterialId() << std::endl;
    const Particle* devParticle =  particles.data();

    GpuClass gpu(devParticle, N);
    //gpu.printHellow();

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits

    cudaDeviceReset();
    // exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
