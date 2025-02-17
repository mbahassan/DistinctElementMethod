
#include <iostream>

#include "Insertion/InsertionCpu.h"
#include "Particle/Particle.hpp"

int main() {

    Insertion system;

    // Create materials and shapes
    Material glass{
            /* parameters */};
    CylinderShape cylinder(/* parameters */);

    // Initialize particles in a grid
    float3 boxMin = make_float3(-5.0f, -5.0f, -5.0f);
    float3 boxMax = make_float3(5.0f, 5.0f, 5.0f);
    float spacing = 1.0f;
    int numParticles = 1000;

    // Choose either grid or random initialization
    auto particles = system.initializeGridParticles(numParticles, glass, cylinder,
                                                  boxMin, boxMax, spacing);
    // OR
    auto particles = system.initializeRandomParticles(numParticles, glass, cylinder,
                                                    boxMin, boxMax, spacing);

    return 0;
}

