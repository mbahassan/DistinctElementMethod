//
// Created by iqraa on 12-3-25.
//

#ifndef GJK_CUH
#define GJK_CUH
#include <random>
#include <Particle/Particle.hpp>
#include "ContactDetection/NarrowPhase/Simplex/Simplex.cuh"

class GJK
{
    public:

      // GJK algorithm implementation following the pseudocode provided
    bool gjkOverlap(const Particle& particleA, const Particle& particleB) {
        // Initialize search direction as random unit vector
        float3 d = randomUnitVec();

        // Initialize empty simplex
        Simplex simplex;

        // Get first support point C
        float3 C = sATMB(particleA, particleB, d);
        simplex.add(C);

        // Early exit check
        if (C.dot(d) < 0) {
            return false;
        }

        // Set new search direction to -C
        d = -C;

        // Get second support point B
        float3 B = sATMB(particleA, particleB, d);
        simplex.add(B);

        // Early exit check
        if (B.dot(d) < 0) {
            return false;
        }

        // Calculate new search direction (line perpendicular to AB through origin)
        float3 BC = C - B;
        d = BC.cross(-B).cross(BC);

        // If d is too small, pick a perpendicular direction
        if (d.lengthSquared() < 0.0001f) {
            // Get perpendicular direction to the line
            float3 perpDir = std::abs(BC.x) > std::abs(BC.y) ?
                float3(0, 1, 0) : float3(1, 0, 0);
            d = BC.cross(perpDir);
        }

        // Main GJK loop
        while (true) {
            // Get new support point A
            float3 A = sATMB(particleA, particleB, d);
            simplex.add(A);

            // Early exit check
            if (A.dot(d) < 0) {
                return false;
            }

            // Update simplex and search direction
            if (updateSimplexAndSearchDirection(simplex, d)) {
                return true; // Origin is inside simplex
            }

            // Check if origin is inside simplex
            if (isOriginInsideSimplex(simplex)) {
                return true;
            }

            // Check for infinite loop: if simplex hasn't changed in a few iterations
            // Too complex to implement here, but would be a good addition
        }
    }
    private:

      // Generate a random unit vector for initial direction
    float3 randomUnitVec() {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        float3 vec(dist(rng), dist(rng), dist(rng));

        // Keep generating until we have a non-zero vector
        while (vec.lengthSquared() < 0.0001f) {
            vec = float3(dist(rng), dist(rng), dist(rng));
        }

        return vec.normalized();
    }

     // Support function for GJK Support function - Minkowski difference
    float3 sATMB(const Particle& particleA, const Particle& particleB, const float3& direction)
    {
        // Get the furthest point of particle A in direction
        float3 supportA = particleA.supportMapping(direction) + particleA.position;
        // float3 supportA = getFurthestPoint(particleA, direction);

        // Get furthest point of particle B in opposite direction
        // float3 supportB = getFurthestPoint(particleB, float3(-direction.x, -direction.y, -direction.z));
        float3 supportB = particleB.supportMapping(-direction) + particleB.position;

        // Return the Minkowski difference
        return supportA - supportB;
    }

    // Line case: determine if origin is in voronoi region of the line
    bool updateSimplexLine(Simplex& simplex, float3& direction) {
        const float3& A = simplex[0]; // Last point added
        const float3& B = simplex[1]; // Second to last point

        const float3 AB = B - A;
        const float3 AO = -A; // Vector from A to origin

        if ((AB&AO) > 0) {
            // Origin is in the direction of the line segment
            direction = (AB^AO)^(AB);
            if (magSquared(direction) < 0.0001f) {
                // Origin is on the line, pick any perpendicular direction
                float3 perpDir = std::abs(AB.x) > std::abs(AB.y) ?
                    float3(0, 1, 0) : float3(1, 0, 0);
                direction = AB^(perpDir);
            }
        } else {
            // Origin is behind A, only keep A
            simplex.getPoints() = {A};
            direction = AO;
        }

        return false;
    }

    // Triangle case
    bool updateSimplexTriangle(Simplex& simplex, float3& direction) {
        const float3& A = simplex[0]; // Last point added
        const float3& B = simplex[1];
        const float3& C = simplex[2];

        const float3 AB = B - A;
        const float3 AC = C - A;
        const float3 AO = -A; // Vector from A to origin

        const float3 ABC = AB^(AC);

        // Check if origin is above or below triangle
        const float3 ABCperp = ABC^(AB);
        if ((ABCperp&AO) > 0) {
            // Origin is on the outside of edge AB
            if ((AB&AO) > 0) {
                // Origin is in voronoi region of AB edge
                simplex.getPoints() = {A, B};
                direction = (AB^AO)^(AB);
            } else {
                // Check AC direction
                const float3 ACperp = AC^ABC;
                if ((ACperp^AO) > 0) {
                    // Origin is in voronoi region of AC edge
                    simplex.getPoints() = {A, C};
                    direction = (AC^AO)^(AC);
                } else {
                    // Origin is in voronoi region of A
                    simplex.getPoints() = {A};
                    direction = AO;
                }
            }
        } else {
            // Origin is on the other side of edge AB
            const float3 ACperp = AC^(ABC);
            if ((ACperp&AO) > 0) {
                // Origin is in voronoi region of AC edge
                simplex.getPoints() = {A, C};
                direction = AC^(AO)^(AC);
            } else {
                // Origin is either in voronoi region of the triangle or on the other side
                if ((ABC&AO) > 0) {
                    // Origin is above the triangle
                    simplex.getPoints() = {A, B, C};
                    direction = ABC;
                } else {
                    // Origin is below the triangle
                    simplex.getPoints() = {A, C, B};
                    direction = -ABC;
                }
            }
        }

        return false;
    }

    // Tetrahedron case
    bool updateSimplexTetrahedron(Simplex& simplex, float3& direction) {
        const float3& A = simplex[0]; // Last point added
        const float3& B = simplex[1];
        const float3& C = simplex[2];
        const float3& D = simplex[3];

        const float3 AB = B - A;
        const float3 AC = C - A;
        const float3 AD = D - A;
        const float3 AO = -A; // Vector from A to origin

        // Check each face of the tetrahedron
        const float3 ABC = AB^(AC);
        const float3 ACD = AC^(AD);
        const float3 ADB = AD^(AB);

        // If the origin is on the same side of all faces, we have a collision
        // Otherwise, we need to find which voronoi region the origin is in

        if ((ABC&AO) > 0) {
            // Origin is on positive side of ABC, remove D
            simplex.getPoints() = {A, B, C};
            return updateSimplexTriangle(simplex, direction);
        }

        if ((ACD&AO) > 0) {
            // Origin is on positive side of ACD, remove B
            simplex.getPoints() = {A, C, D};
            return updateSimplexTriangle(simplex, direction);
        }

        if ((ADB&AO) > 0) {
            // Origin is on positive side of ADB, remove C
            simplex.getPoints() = {A, D, B};
            return updateSimplexTriangle(simplex, direction);
        }

        // If we got here, the origin is inside the tetrahedron
        return true;
    }

    // Update simplex and search direction
    bool updateSimplexAndSearchDirection(Simplex& simplex, float3& direction) {
        switch (simplex.size()) {
            case 2:
                return updateSimplexLine(simplex, direction);
            case 3:
                return updateSimplexTriangle(simplex, direction);
            case 4:
                return updateSimplexTetrahedron(simplex, direction);
            default:
                // Should never get here
                    return false;
        }
    }

    bool isOriginInsideSimplex(Simplex& simplex);

    std::mt19937 rng;
};


#endif //GJK_CUH
