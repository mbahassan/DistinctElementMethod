//
// Created by iqraa on 15-2-25.
//

#ifndef INSERTIONCPU_H
#define INSERTIONCPU_H

#include <iostream>
#include <vector>
#include <random>
#include <Particle/Polyhedral.h>
#include <thrust/for_each.h>

#include "Domain/Domain.h"
#include "Regions/CubeInsertionRegion.h"
#include "Regions/CylindricalRegion.h"

// Forward declarations
class CubeRegion;
class CylindricalRegion;


class Insertion : public Domain
{
public:
    // Constructor using Domain as the base class
    explicit Insertion(const Domain& domain) : Domain(domain), insertRegion_(nullptr), useFullDomain_(true)
    {
        // Create a default insert region using the full domain
        insertRegion_ = std::make_shared<CubeInsertionRegion>(domain.getMin(), domain.getMax());
    }

    // Destructor to clean up allocated memory
    ~Insertion() = default;

    // Set a custom cube insertion region
    void setInsertionRegion(const float3& min, const float3& max)
    {
        insertRegion_ = std::make_shared<CubeInsertionRegion>(min, max);
        useFullDomain_ = false;
    }

    // Set a custom cylindrical insertion region
    void setInsertionRegion(const float3& center, float radius, float height)
    {
        insertRegion_ = std::make_shared<CylindricalInsertionRegion>(center, radius, height);
        useFullDomain_ = false;
    }

    // Use the domain as the insertion region
    void useFullDomainAsInsertRegion()
    {
        insertRegion_ = std::make_shared<CubeInsertionRegion>(getMin(), getMax());
        useFullDomain_ = true;
    }

    // Initialize particles in a cubic grid arrangement
    template<typename ParticleType>
    std::vector<ParticleType> fillGrid(std::vector<ParticleType>& particles, const float spacing)
    {
        float3 currentPos = insertRegion_->getMin();
        int particlesCreated = 0;
        const unsigned numParticles = particles.size();

        // Calculate balanced dimensions
        int a = static_cast<int>(std::cbrt(numParticles));
        int b = a, c = a;
        while (a * b * c < numParticles) {
            if (a <= b && a <= c) ++a;
            else if (b <= c) ++b;
            else ++c;
        }

        std::vector<ParticleType> validParticles;
        validParticles.reserve(numParticles);

        for(int x = 0; x < a && particlesCreated < numParticles; x++) {
            currentPos.y = insertRegion_->getMin().y;
            for(int y = 0; y < b && particlesCreated < numParticles; y++) {
                currentPos.z = insertRegion_->getMin().z;
                for(int z = 0; z < c && particlesCreated < numParticles; z++) {
                    // Create a copy of the current particle to test
                    ParticleType testParticle = particles[particlesCreated];

                    // Set position
                    testParticle.position = currentPos;

                    // Initialize orientation for non-spherical particles
                    if (testParticle.getShapeType() != Shape::SPHERE)
                    {
                        Quaternion randomOrientation = getRandomOrientation();
                        testParticle.setOrientation(randomOrientation);

                        // Update vertices positions
                        updateParticleVertices(testParticle);
                    }

                    // Update bounding box
                    updateParticleBoundingBox(testParticle);

                    // Check if the particle fits within the insertion region
                    if (insertRegion_->contains(testParticle.position, testParticle.boundingBox) &&
                        !overlapsWithExisting(testParticle, validParticles))
                    {
                        validParticles.push_back(testParticle);
                        particlesCreated++;
                    }

                    currentPos.z += spacing;
                }
                currentPos.y += spacing;
            }
            currentPos.x += spacing;
        }

        std::cout << "Successfully placed " << validParticles.size()
                  << " out of " << numParticles << " particles." << std::endl;

        return validParticles;
    }

    // Initialize particles with random positions
    template<typename ParticleType>
    std::vector<ParticleType> fillRandomly(std::vector<ParticleType>& particles, int maxAttempts = 100) {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::vector<ParticleType> validParticles;
        validParticles.reserve(particles.size());

        const size_t numParticles = particles.size();

        for (size_t i = 0; i < numParticles; i++) {
            bool validPosition = false;
            int attempts = 0;

            while (!validPosition && attempts < maxAttempts) {
                // Create a test particle
                ParticleType testParticle = particles[i];

                // Generate random position
                float3 testPos = make_float3(
                    insertRegion_->getRandomX(gen),
                    insertRegion_->getRandomY(gen),
                    insertRegion_->getRandomZ(gen)
                );

                testParticle.position = testPos;

                // Initialize random orientation for non-spherical particles
                if (testParticle.getShapeType() != Shape::SPHERE) {
                    Quaternion randomOrientation = getRandomOrientation();
                    testParticle.setOrientation(randomOrientation);

                    // Update vertices positions
                    updateParticleVertices(testParticle);
                }

                // Update bounding box
                updateParticleBoundingBox(testParticle);

                // Check if the particle fits within the insertion region and doesn't overlap
                if (insertRegion_->contains(testParticle.position, testParticle.boundingBox) &&
                    !overlapsWithExisting(testParticle, validParticles)) {
                    validParticles.push_back(testParticle);
                    validPosition = true;
                }

                attempts++;
            }

            if (attempts >= maxAttempts) {
                std::cerr << "Warning: Could not place particle " << i
                         << " after " << maxAttempts << " attempts." << std::endl;
            }
        }

        std::cout << "Successfully placed " << validParticles.size()
                  << " out of " << numParticles << " particles." << std::endl;

        return validParticles;
    }

    // Initialize particles in a 2D grid arrangement
    template<typename ParticleType>
    std::vector<ParticleType> fillGrid2D(std::vector<ParticleType>& particles, const float spacing) {
        const unsigned numParticles = particles.size();

        // Calculate balanced grid dimensions using a square grid (2D)
        int a = static_cast<int>(std::sqrt(numParticles));
        int b = a;
        while (a * b < numParticles) {
            if (a <= b) ++a;
            else ++b;
        }

        std::vector<ParticleType> validParticles;
        validParticles.reserve(numParticles);

        float3 currentPos = insertRegion_->getMin();
        int particlesCreated = 0;

        // Create particles on a 2D grid (x and y), keeping z constant
        for (int x = 0; x < a && particlesCreated < numParticles; x++) {
            currentPos.y = insertRegion_->getMin().y;
            for (int y = 0; y < b && particlesCreated < numParticles; y++) {
                // Create a test particle
                ParticleType testParticle = particles[particlesCreated];

                // Set position; using lower z-value for 2D arrangement
                testParticle.position = make_float3(currentPos.x, currentPos.y, insertRegion_->getMin().z);

                // For non-spherical particles, initialize a 2D orientation (rotation around z-axis only)
                if (testParticle.getShapeType() != Shape::SPHERE) {
                    Quaternion randomOrientation = getRandomOrientation2D();
                    testParticle.setOrientation(randomOrientation);

                    // Update vertices positions
                    updateParticleVertices(testParticle);
                }

                // Update bounding box
                updateParticleBoundingBox(testParticle);

                // Check if the particle fits within the insertion region and doesn't overlap
                if (insertRegion_->contains(testParticle.position, testParticle.boundingBox) &&
                    !overlapsWithExisting(testParticle, validParticles)) {
                    validParticles.push_back(testParticle);
                    particlesCreated++;
                }

                currentPos.y += spacing;
            }
            currentPos.x += spacing;
        }

        std::cout << "Successfully placed " << validParticles.size()
                  << " out of " << numParticles << " particles." << std::endl;

        return validParticles;
    }

    // Initialize particles with random positions in 2D
    template<typename ParticleType>
    void fillRandomly2D(std::vector<ParticleType>& particles, int maxAttempts = 1000)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        // std::vector<ParticleType> validParticles;
        // validParticles.reserve(particles.size());

        const size_t numParticles = particles.size();

        // std::for_each(particles.begin(), particles.end(), [](ParticleType& particle) {
        //
        // });

        for (int i = 0; i< particles.size(); i++)
        {
            bool validPosition = false;
            int attempts = 0;

            while (!validPosition && attempts < maxAttempts) {
                // Create a test particle
                // ParticleType testParticle = particles[i];
                // particles[i];

                // Generate random position (2D - fixed z)
                float3 testPos = make_float3(
                    insertRegion_->getRandomX(gen),
                    insertRegion_->getRandomY(gen),
                    insertRegion_->getMin().z  // Fixed Z for 2D
                );

                particles[i].position = testPos;

                // Initialize random 2D orientation for non-spherical particles
                if (particles[i].getShapeType() != Shape::SPHERE)
                {
                    Quaternion randomOrientation = getRandomOrientation2D();
                    particles[i].orientation = randomOrientation;

                    // Update vertices positions
                    updateParticleVertices(particles[i]);
                }

                // Update bounding box
                updateParticleBoundingBox(particles[i]);

                // Check if the particle fits within the insertion region and doesn't overlap
                if (insertRegion_->contains(particles[i].boundingBox) &&
                    !overlapsWithExisting(particles[i], particles, i))
                {
                    // validParticles.push_back(testParticle);
                    validPosition = true;
                }

                attempts++;
            }

            if (attempts >= maxAttempts)
                {
                std::cerr << "Warning: Could not place particle " << i
                         << " after " << maxAttempts << " attempts." << std::endl;
            }
        }

        std::erase_if(particles,[this](ParticleType& p) {
            return !isValid(p);
        });

        std::cout << "Successfully placed " << particles.size()
                  << " out of " << numParticles << " particles." << std::endl;

        if (particles.empty()) exit(0);
        // return validParticles;
    }

private:
    std::shared_ptr<InsertionRegion> insertRegion_;
    bool useFullDomain_;

    template<typename ParticleType>
    static bool isValid(ParticleType& p)
    {
        if (p.position.x == -1.f && p.position.y == -1.f &&p.position.z == -1.f )
            return false;
        return true;
    }

    // Generate a random 3D orientation
    static Quaternion getRandomOrientation()

    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dist(0.0f, 1.0f);

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

    // Generate a random 2D orientation (rotation around z-axis only)
    static Quaternion getRandomOrientation2D()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * M_PI);

        // Random angle in radians for rotation around Z-axis
        float theta = angleDist(gen);

        float w = std::cos(theta / 2.0f);
        float x = 0.0f;
        float y = 0.0f;
        float z = std::sin(theta / 2.0f);

        return {w, x, y, z};  // This is a 2D rotation quaternion (rotation around z-axis)
    }

    // Update vertices positions based on particle orientation and position
    template<typename ParticleType>
    void updateParticleVertices(ParticleType& particle)
    {
        const Quaternion& q = particle.orientation;

        for (int n = 0; n < particle.numVertices; ++n)
        {
            // Rotate the vertex by the orientation quaternion
            particle.vertices[n] = q.rotateVector(particle.vertices[n]);

            // Translate the vertex by the particle position
            particle.vertices[n] += particle.position;
        }
    }

    // Update the particle's bounding box based on its current position and vertices
    template<typename ParticleType>
    void updateParticleBoundingBox(ParticleType& particle)
    {
        particle.updateBoundingBox();
    }

    // Check if a particle overlaps with any existing particles
    template<typename ParticleType>
    bool overlapsWithExisting(const ParticleType& iParticle,
                             const std::vector<ParticleType>& particles, const int i)
    {
        for (int j = 0; j < i ; j++)
        {
            // Check for bounding box overlap as a quick test
            if (boundingBoxesOverlap(iParticle.boundingBox, particles[j].boundingBox))
            {
                // For detailed collision detection, you might want to add more sophisticated
                // collision detection for polyhedral particles here
                //\todo: maybe it is good to use GJK algorithm here
                return true;
            }
        }
        return false;
    }

    // Check if two bounding boxes overlap
    static bool boundingBoxesOverlap(const BoundingBox<float3>& a, const BoundingBox<float3>& b)
    {
        // Check if boxes overlap in all three dimensions
        return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
               (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
               (a.min.z <= b.max.z && a.max.z >= b.min.z);
    }
};

#endif //INSERTIONCPU_H