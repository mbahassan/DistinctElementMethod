//
// Created by iqraa on 15-2-25.
//

#ifndef INSERTIONCPU_H
#define INSERTIONCPU_H


#include <iostream>
#include <vector>
#include <random>
#include "Particle/Particle.h"

class Insertion {
public:
    Insertion() = default;

    // Initialize particles in a cubic grid arrangement
    std::vector<Particle> fillGrid(int numParticles,
                                                const Material& material,
                                                const Shape& shape,
                                                float3 boxMin,
                                                float3 boxMax,
                                                float spacing) {
        std::vector<Particle> particles;
        particles.reserve(numParticles);

        float3 currentPos = boxMin;
        int particlesCreated = 0;

        // Calculate number of particles in each dimension
        int numPerSide = static_cast<int>(std::cbrt(numParticles));

        for(int x = 0; x < numPerSide && particlesCreated < numParticles; x++)
        {
            currentPos.y = boxMin.y;
            for(int y = 0; y < numPerSide && particlesCreated < numParticles; y++)
            {
                currentPos.z = boxMin.z;
                for(int z = 0; z < numPerSide && particlesCreated < numParticles; z++)
                {
                    // Create particle with current position
                    Particle particle(material, shape, currentPos, make_float3(0.0f, 0.0f, 0.0f));

                    /// Initialize random orientation for non-spherical particles
                    if (shape.getType() != ShapeType::Sphere)
                    {
                        Quaternion randomOrientation = getRandomOrientation();
                        particle.setOrientation(randomOrientation);
                    }

                    particles.push_back(particle);
                    particlesCreated++;
                    currentPos.z += spacing;
                }
                currentPos.y += spacing;
            }
            currentPos.x += spacing;
        }

        return particles;
    }

    // Initialize particles with random positions within a box
    std::vector<Particle> fillRandomly(int numParticles,
                                                  const Material& material,
                                                  const Shape& shape,
                                                  float3 boxMin,
                                                  float3 boxMax,
                                                  float minSpacing) {
        std::vector<Particle> particles;
        particles.reserve(numParticles);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distX(boxMin.x, boxMax.x);
        std::uniform_real_distribution<float> distY(boxMin.y, boxMax.y);
        std::uniform_real_distribution<float> distZ(boxMin.z, boxMax.z);

        int maxAttempts = 100;  // Maximum attempts to place a particle

        for(int i = 0; i < numParticles; i++)
        {
            bool validPosition = false;
            int attempts = 0;

            while(!validPosition && attempts < maxAttempts)
            {
                float3 testPos = make_float3(distX(gen), distY(gen), distZ(gen));
                validPosition = true;

                // Check distance from all existing particles
                for(auto& particle : particles) {
                    float3 diff;
                    diff.x = testPos.x - particle.getPosition().x;
                    diff.y = testPos.y - particle.getPosition().y;
                    diff.z = testPos.z - particle.getPosition().z;

                    float distance = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
                    if(distance < minSpacing)
                    {
                        validPosition = false;
                        break;
                    }
                }

                if(validPosition) {
                    Particle particle(material, shape, testPos, make_float3(0.0f, 0.0f, 0.0f));

                    // Initialize random orientation for non-spherical particles
                    if (shape.getType() != ShapeType::Sphere) {
                        Quaternion randomOrientation = getRandomOrientation();
                        particle.setOrientation(randomOrientation);
                    }

                    particles.push_back(particle);
                }
                attempts++;
            }

            if(attempts >= maxAttempts) {
                // Could not place particle after maximum attempts
                std::cerr << "Warning: Could not place particle " << i << " after "
                         << maxAttempts << " attempts" << std::endl;
                break;
            }
        }

        return particles;
    }

private:
    Quaternion getRandomOrientation()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        // Generate random rotation using uniform distribution method
        float u1 = dist(gen);
        float u2 = dist(gen);
        float u3 = dist(gen);

        float w = sqrtf(1.0f - u1) * sinf(2.0f * M_PI * u2);
        float x = sqrtf(1.0f - u1) * cosf(2.0f * M_PI * u2);
        float y = sqrtf(u1) * sinf(2.0f * M_PI * u3);
        float z = sqrtf(u1) * cosf(2.0f * M_PI * u3);

        return {w, x, y, z};
    }
};


#endif //INSERTIONCPU_H
