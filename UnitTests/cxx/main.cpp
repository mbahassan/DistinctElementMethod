
// includes, system


// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <ContactDetection/ContactDetection.cuh>
#include <Particle/Spherical.h>
#include <Particle/Polyhedral.h>

#include <Tools/Config/Parser.h>
#include "Insertion/InsertionCpu.h"

#include <Simulate/Simulate.cuh>

int main(int argc, char **argv)
{

    auto config = Parser::getConfig("input.json");
    int N = config.numberOfParticles;

    Sphere sphere(0.1);
    Material glass(config.materialPath);

    Polytope cube("sphere.stl");

    std::vector<Polyhedral> poly(N);

    for(int i = 0; i < N; i++)
    {
        poly[i].setId(i);
        poly[i] = Polyhedral(glass, cube);
    }

    std::vector<Spherical> particles(N);
    for(int i = 0; i < N; i++)
    {
        particles[i].setId(i);
        particles[i] = Spherical(glass, sphere);
        particles[i].setRadius(0.03f);
    }

    Insertion insertion;
    insertion.fillRandomly2D(poly, {0,0,0}, {2.,2.,0});

    /// Simulate
    Simulate<Polyhedral> simulate(0.0001, LSD, Euler, "input.json");
    simulate.addParticles(poly);
    simulate.solve(3*0.0001);

    // simulate.run(0.001);
    // ContactDetection<Spherical> cd("input.json");
    // auto potential_pairs = cd.broadPhase(particles);
    // auto actual_contacts = cd.narrowPhase(particles, potential_pairs);

    // Output output("results");
    // output.writeParticles(particles, 0);
    // output.writeParticles(poly, 0);
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

    // cudaDeviceReset();
    // exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
    return 0;
}
