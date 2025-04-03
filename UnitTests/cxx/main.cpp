
// includes, system

// Required to include CUDA vector types
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
    simulate.solve(0.3);

    return 0;
}
