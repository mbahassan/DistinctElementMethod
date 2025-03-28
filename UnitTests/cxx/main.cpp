
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
    Material glass(config.materialConfigPath);

    Polytope cube("sphere.stl");



    std::vector<Polyhedral> poly(N);

    for(int i = 0; i < N; i++)
    {
        poly[i] = Polyhedral(glass, cube);
        poly[i].id = i;
    }

    std::vector<Spherical> particles(N);
    for(int i = 0; i < N; i++)
    {
        particles[i] = Spherical(glass, sphere);
        particles[i].setId(i);
        particles[i].setRadius(0.03f);
    }

    Insertion insertion;
    insertion.fillRandomly2D(poly, {0,0,0}, {2.,2.,0});

    ContactDetection<Polyhedral> cd("input.json");
    auto potential_pairs = cd.broadPhase(poly);
    // auto actual_contacts = cd.runNarrowPhase(particles, potential_pairs);

    Output output("results");
    output.writeParticles(particles, 0);
    output.writeParticles(poly, 0);
    // output.writeTree(cd.getTree(),0);

    // std::cout << "particle radius: "<< particles[0].getRadius() << std::endl;
    // std::cout << "particle Material id: "<< particles[0].getMaterialId() << std::endl;
    // const Spherical* devParticle =  particles.data();

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
