
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

    Domain domain({0.f,0.f,0.f}, {2.f,2.f,2.f});

    CubeRegion region(domain);
    region.setMin({0.f,0.f,0.f});
    region.setMax({2.f,2.f,2.f});

    Sphere sphere(0.03f);
    Material glass(config.materialPath);


    Polytope cube("sphere0.03R2D.stl");

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

    Insertion insertion(domain);
    insertion.fillRandomly2D(poly);

    /// Simulate
    Simulate<Polyhedral> simulate(0.01, LSD, Euler, "input.json");
    simulate.addParticles(poly);
    simulate.setGravity({0.f,-9.81, 0.f});
    simulate.solve(0.3);

    return 0;
}
