//
// Created by iqraa on 12-3-25.
//

#ifndef GJK_CUH
#define GJK_CUH

#include <random>
#include "Particle/Spherical.hpp"
#include "ContactDetection/NarrowPhase/Simplex/Simplex.h"

class GJK {
public:
    GJK() : rng(std::random_device{}()) {}

    // GJK algorithm to detect overlap between two particles
    bool gjkOverlap(const Spherical& particleA, const Spherical& particleB) {
        // Initialize search direction as random unit vector
        float3 d = randomUnitVec();

        // Initialize empty simplex
        Simplex simplex;

        // Get first support point C
        float3 C = sATMB(particleA, particleB, d);
        simplex.add(C);

        // Early exit check
        if ((C&d) < 0) {
            return false;
        }

        // Set new search direction to -C
        d = -C;

        // Get second support point B
        float3 B = sATMB(particleA, particleB, d);
        simplex.add(B);

        // Early exit check
        if ((B&d) < 0) {
            return false;
        }

        // Calculate new search direction (line perpendicular to AB through origin)
        float3 BC = C - B;
        d = (BC^-B)^BC;

        // If d is too small, pick a perpendicular direction
        if (magSquared(d) < 0.0001f) {
            // Get perpendicular direction to the line
            float3 perpDir = std::abs(BC.x) > std::abs(BC.y) ?
                float3(0, 1, 0) : float3(1, 0, 0);
            d = BC ^ perpDir;
        }

        // Main GJK loop
        int maxIterations = 20; // Prevent infinite loop
        for (int i = 0; i < maxIterations; i++) {
            // Get new support point A
            float3 A = sATMB(particleA, particleB, d);
            simplex.add(A);

            // Early exit check
            if ((A&d) < 0) {
                return false;
            }

            // Update simplex and search direction
            if (updateSimplexAndSearchDirection(simplex, d)) {
                return true; // Origin is inside simplex
            }
        }

        return false;
    }

    // Modified version of GJK that also returns the simplex
    bool gjkOverlapWithSimplex(const Spherical& particleA, const Spherical& particleB, Simplex& simplex) {
        // Initialize search direction as random unit vector
        float3 d = randomUnitVec();

        // Clear simplex
        simplex.clear();

        // Get first support point C
        float3 C = sATMB(particleA, particleB, d);
        simplex.add(C);

        // Early exit check
        if ((C&d) < 0) {
            return false;
        }

        // Set new search direction to -C
        d = -C;

        // Get second support point B
        float3 B = sATMB(particleA, particleB, d);
        simplex.add(B);

        // Early exit check
        if ((B&d) < 0) {
            return false;
        }

        // Calculate new search direction (line perpendicular to AB through origin)
        float3 BC = C - B;
        d = (BC^-B)^BC;

        // If d is too small, pick a perpendicular direction
        if (magSquared(d) < 0.0001f) {
            // Get perpendicular direction to the line
            float3 perpDir = std::abs(BC.x) > std::abs(BC.y) ?
                float3(0, 1, 0) : float3(1, 0, 0);
            d = BC ^ perpDir;
        }

        /// Main GJK loop
        int maxIterations = 20;
        for (int i = 0; i < maxIterations; i++) {
            // Get new support point A
            float3 A = sATMB(particleA, particleB, d);
            simplex.add(A);

            // Early exit check
            if ((A&d) < 0) {
                return false;
            }

            // Update simplex and search direction
            if (updateSimplexAndSearchDirection(simplex, d)) {
                return true; // Origin is inside simplex
            }
        }

        return false;
    }

private:
    std::mt19937 rng;

    // Generate a random unit vector for initial direction
    float3 randomUnitVec() {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        float3 vec(dist(rng), dist(rng), dist(rng));

        // Keep generating until we have a non-zero vector
        while (magSquared(vec) < 0.0001f) {
            vec = float3(dist(rng), dist(rng), dist(rng));
        }

        return normalize(vec);
    }

    // Support function for GJK - Minkowski difference
    float3 sATMB(const Spherical& particleA, const Spherical& particleB, const float3& direction) {
        // Get the furthest point of particle A in direction
        float3 supportA = particleA.supportMapping(direction) + particleA.position;

        // Get furthest point of particle B in opposite direction
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
            direction = (AB^AO)^AB;
            if (magSquared(direction) < 0.0001f) {
                // Origin is on the line, pick any perpendicular direction
                float3 perpDir = std::abs(AB.x) > std::abs(AB.y) ?
                    float3(0, 1, 0) : float3(1, 0, 0);
                direction = AB ^ perpDir;
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

        const float3 ABC = AB^AC;

        // Check if origin is above or below triangle
        const float3 ABCperp = ABC^AB;
        if ((ABCperp&AO) > 0) {
            // Origin is on the outside of edge AB
            if ((AB&AO) > 0) {
                // Origin is in voronoi region of AB edge
                simplex.getPoints() = {A, B};
                direction = (AB^AO)^AB;
            } else {
                // Check AC direction
                const float3 ACperp = AC^ABC;
                if ((ACperp&AO) > 0.0f) {
                    // Origin is in voronoi region of AC edge
                    simplex.getPoints() = {A, C};
                    direction = (AC^AO)^AC;
                } else {
                    // Origin is in voronoi region of A
                    simplex.getPoints() = {A};
                    direction = AO;
                }
            }
        } else {
            // Origin is on the other side of edge AB
            const float3 ACperp = AC^ABC;
            if ((ACperp&AO) > 0) {
                // Origin is in voronoi region of AC edge
                simplex.getPoints() = {A, C};
                direction = AC^AO^AC;
            } else {
                // Origin is either in voronoi region of the triangle or on the other side
                if ((ABC&AO) > 0) {
                    // Origin is above the triangle
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
        const float3 ABC = AB^AC;
        const float3 ACD = AC^AD;
        const float3 ADB = AD^AB;

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

    // Check if origin is inside simplex
    bool isOriginInsideSimplex(const Simplex& simplex) {
        // For a tetrahedron, we need to check if origin is inside
        if (simplex.size() == 4) {
            const float3& A = simplex[0];
            const float3& B = simplex[1];
            const float3& C = simplex[2];
            const float3& D = simplex[3];

            const float3 AB = B - A;
            const float3 AC = C - A;
            const float3 AD = D - A;
            const float3 AO = -A;

            // Check if origin is on the same side of all faces
            float3 ABC = AB^AC;
            float3 ACD = AC^AD;
            float3 ADB = AD^AB;
            float3 BDC = (D-B)^(C-B);

            if ((ABC&AO) > 0 && (ACD&AO) > 0 && (ADB&AO) > 0 && (BDC&(B-float3(0,0,0))) > 0) {
                return true;
            }
        }
        return false;
    }
};

#endif // GJK_CUH