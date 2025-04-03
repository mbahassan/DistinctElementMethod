//
// Created by iqraa on 15-2-25.
//

#ifndef INSERTIONCPU_H
#define INSERTIONCPU_H


#include <iostream>
#include <vector>
#include <random>
// #include <Particle/Spherical.h>
#include <Particle/Polyhedral.h>


class Insertion {
public:
    Insertion() = default;

    // Initialize particles in a cubic grid arrangement
    template<typename ParticleType>
    std::vector<ParticleType> fillGrid(std::vector<ParticleType>& particles,
                                                float3 boxMin,
                                                float3 boxMax,
                                                float spacing) {
        float3 currentPos = boxMin;
        int particlesCreated = 0;
        const unsigned numParticles = particles.size();

        // Calculate number of particles in each dimension
        int numPerSide = static_cast<int>(std::cbrt(numParticles));

        // Calculate balanced dimensions
        int a = static_cast<int>(std::cbrt(numParticles));
        int b = a, c = a;
        while (a * b * c < numParticles) {
            if (a <= b && a <= c) ++a;
            else if (b <= c) ++b;
            else ++c;
        }

        for(int x = 0; x < a && particlesCreated < numParticles; x++)
        {
            currentPos.y = boxMin.y;
            for(int y = 0; y < b && particlesCreated < numParticles; y++)
            {
                currentPos.z = boxMin.z;
                for(int z = 0; z < c && particlesCreated < numParticles; z++)
                {
                    // Create particle with current position
                    // Particle particle(material, shape, currentPos, make_float3(0.0f, 0.0f, 0.0f));
                    particles[particlesCreated].position = (currentPos);
                    /// Initialize random orientation for non-spherical particles
                    if (particles[particlesCreated].getShapeType() != Shape::SPHERE)
                    {
                        Quaternion randomOrientation = getRandomOrientation();
                        particles[particlesCreated].setOrientation(randomOrientation);
                    }

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
    template<typename ParticleType>
    std::vector<ParticleType> fillRandomly(std::vector<ParticleType> particles,
                                                  float3 boxMin,
                                                  float3 boxMax,
                                                  float minSpacing) {
        // std::vector<Particle> particles;
        // particles.reserve(numParticles);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distX(boxMin.x, boxMax.x);
        std::uniform_real_distribution<float> distY(boxMin.y, boxMax.y);
        std::uniform_real_distribution<float> distZ(boxMin.z, boxMax.z);

        int maxAttempts = 100;  // Maximum attempts to place a particle
        const size_t numParticles = particles.size();
        for(int i = 0; i < numParticles; i++)
        {
            bool validPosition = false;
            int attempts = 0;

            while(!validPosition && attempts < maxAttempts)
            {
                float3 testPos = make_float3(distX(gen), distY(gen), distZ(gen));
                validPosition = true;

                // Check distance from all existing particles
                for(auto& particle : particles)
                {
                    float3 diff = testPos - particle.position;

                    float distance = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
                    if(distance < minSpacing)
                    {
                        validPosition = false;
                        break;
                    }
                }

                if(validPosition)
                {
                    particles[attempts].position = (testPos);
                    // Initialize random orientation for non-spherical particles
                    if (particles[attempts].getShapeType() != Shape::SPHERE) {
                        Quaternion randomOrientation = getRandomOrientation();
                        particles[attempts].setOrientation(randomOrientation);
                    }
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

    template<typename ParticleType>
    std::vector<ParticleType> fillGrid2D(std::vector<ParticleType>& particles,
                                 float3 boxMin,
                                 float3 boxMax,
                                 float spacing) {
        int particlesCreated = 0;
        const unsigned numParticles = particles.size();

        // Calculate balanced grid dimensions using a square grid (2D)
        int a = static_cast<int>(std::sqrt(numParticles));
        int b = a;
        while (a * b < numParticles) {
            if (a <= b) ++a;
            else ++b;
        }

        float3 currentPos = boxMin;

        // Create particles on a 2D grid (x and y), keeping z constant
        for (int x = 0; x < a && particlesCreated < numParticles; x++) {
            currentPos.y = boxMin.y;
            for (int y = 0; y < b && particlesCreated < numParticles; y++) {
                // Set the particle position; using boxMin.z as a constant z-value.
                particles[particlesCreated].position = make_float3(currentPos.x, currentPos.y, boxMin.z);

                // For non-spherical particles, initialize a random orientation.
                if (particles[particlesCreated].getShapeType() != Shape::SPHERE) {
                    Quaternion randomOrientation = getRandomOrientation();
                    particles[particlesCreated].setOrientation(randomOrientation);
                }

                particlesCreated++;
                currentPos.y += spacing;
            }
            currentPos.x += spacing;
        }

        return particles;
    }

    template<typename ParticleType>
    std::vector<ParticleType> fillRandomly2D(std::vector<ParticleType>& particles,
                                         float3 regionMin,
                                         float3 regionMax) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution distX(regionMin.x, regionMax.x);
        std::uniform_real_distribution distY(regionMin.y, regionMax.y);

        int maxAttempts = 1000;  // Maximum attempts to place a particle
        const size_t numParticles = particles.size();

        int valid = 0;
        for (int i = 0; i < numParticles; ++i) {
            bool validPosition = false;
            int attempts = 0;

            // Get current particle's bounding box size for spacing calculation
            auto ithParticleBBox = particles[i].boundingBox;
            float3 currentSize = ithParticleBBox.getSize();
            float currentRadius = std::max(std::max(currentSize.x, currentSize.y), currentSize.z) * 0.5f;

            while (!validPosition && attempts < maxAttempts) {
                float testX = distX(gen);
                float testY = distY(gen);
                float3 testPos = make_float3(testX, testY, regionMin.z); // Fixed Z plane

                validPosition = true;

                // Check distance from all already placed particles
                for (int j = 0; j < valid; ++j) {
                    float3 diff = testPos - particles[j].position;
                    float distance = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

                    // Calculate minimum spacing between these two specific particles
                    auto jthParticleBBox = particles[j].boundingBox;
                    float3 otherSize = jthParticleBBox.getSize();
                    float otherRadius = std::max(std::max(otherSize.x, otherSize.y), otherSize.z) * 0.5f;

                    // Minimum distance should be sum of radii plus a small buffer (5%)
                    float minSpacing = (currentRadius + otherRadius) * 1.05f;

                    if (distance < minSpacing) {
                        validPosition = false;
                        break;
                    }

                    // Also check if this placement would put us outside the region
                    if (testX - currentRadius < regionMin.x || testX + currentRadius > regionMax.x ||
                        testY - currentRadius < regionMin.y || testY + currentRadius > regionMax.y)
                    {
                        validPosition = false;
                        break;
                    }
                }

                if (validPosition) {
                    particles[i].position = testPos;

                    // For non-spherical particles, initialize random orientation
                    if (particles[i].shapeType != Shape::SPHERE) {
                        particles[i].orientation = getRandomOrientation();
                    }

                    // Update bounding box after position is set
                    particles[i].boundingBox.min + testPos; // Assuming this method exists
                    particles[i].boundingBox.max + testPos; // Assuming this method exists
                    valid++;
                }

                attempts++;
            }

            if (attempts >= maxAttempts) {
                std::cerr << "Warning: Could not place particle " << i
                          << " after " << maxAttempts << " attempts. "
                          << "Successfully placed " << valid << " out of " << numParticles << std::endl;
                break;
            }
        }

        std::cout << "Successfully placed " << valid << " out of " << numParticles << " particles." << std::endl;

        // If some particles couldn't be placed, resize the vector
        if (valid < numParticles) {
            particles.resize(valid);
        }

        return particles;
    }


private:
    static Quaternion getRandomOrientation()
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
