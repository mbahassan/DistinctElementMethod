
// includes, system


// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <ContactDetection/ContactDetection.cuh>
#include <Output/Output.cuh>
#include <Particle/Spherical.hpp>
#include <Particle/Polyhedral.hpp>

#include <Tools/Config/Parser.h>
#include "Insertion/InsertionCpu.h"


int main(int argc, char **argv)
{
    auto config = Parser::getConfig("input.json");

    int N = config.numberOfParticles;

    Sphere sphere(0.1);

    Polytope pSphere("sphere0.03R.obj");

    Material glass(config.materialConfigPath);

    std::vector<Polyhedral> polytopes(1);
    for(int i = 0; i < 1; i++)
    {
        polytopes[i] = Polyhedral(glass, pSphere);
        polytopes[i].setId(i);
    }
    // polytopes[0].position = {0.10f, 0.90f, 0.0f};

    // std::vector<Spherical> particles(11);
    // for(int i = 0; i < 11; i++)
    // {
    //     particles[i] = Spherical(glass, sphere);
    //     particles[i].setId(i);
    //     particles[i].setRadius(0.03f);
    // }

    // Insertion insertion;
    // insertion.fillGrid2D(particles, {0,0,0}, {1,1,0},0.2);

    // Top-left quadrant (x in [0, 0.5], y in [0.5, 1]): 4 particles -> should subdivide.
    polytopes[0].position = {0.10f, 0.90f, 0.0f};
    polytopes[1].position = {0.20f, 0.80f, 0.0f};
    polytopes[2].position = {0.30f, 0.70f, 0.0f};
    polytopes[3].position = {0.40f, 0.60f, 0.0f};

    // Top-right quadrant (x in [0.5, 1], y in [0.5, 1]): 3 particles -> no subdivision.
    polytopes[4].position = {0.60f, 0.90f, 0.0f};
    polytopes[5].position = {0.70f, 0.80f, 0.0f};
    polytopes[6].position = {0.80f, 0.70f, 0.0f};

    // Bottom-left quadrant (x in [0, 0.5], y in [0, 0.5]): 3 particles -> no subdivision.
    polytopes[7].position = {0.10f, 0.10f, 0.0f};
    polytopes[8].position = {0.20f, 0.20f, 0.0f};
    polytopes[9].position = {0.30f, 0.30f, 0.0f};

    // Bottom-right quadrant (x in [0.5, 1], y in [0, 0.5]): 2 particles -> no subdivision.
    polytopes[10].position = {0.60f, 0.10f, 0.0f};
    polytopes[11].position = {0.70f, 0.20f, 0.0f};


    ContactDetection cd("input.json");
    auto potential_pairs = cd.runBroadPhase(polytopes);
    // auto actual_contacts = cd.runNarrowPhase(particles, potential_pairs);

    Output output("results");
    output.writeParticles(polytopes, 0);
    output.writeParticles(polytopes, 0);
    // output.writeTree(cd.getTree(),0);

    std::cout << "particle radius: "<< particles[0].getRadius() << std::endl;
    std::cout << "particle Material id: "<< particles[0].getMaterialId() << std::endl;
    const Spherical* devParticle =  particles.data();

    // GpuClass gpu(devParticle, N);
    // gpu.printHellow();

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits

    cudaDeviceReset();
    // exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
