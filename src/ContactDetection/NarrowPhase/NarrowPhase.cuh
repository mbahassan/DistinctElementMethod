//
// Created by iqraa on 11-3-25.
//

#ifndef NARROWPHASE_CUH
#define NARROWPHASE_CUH

#include <cfloat>
#include <vector>
#include <Particle/Particle.hpp>
#include "Simplex/Simplex.cuh"

struct PotentialContact
{
    int nodeId;
    int particleIdI;
    int particleIdJ;
};

struct Contact
{
    Particle pi;
    Particle pj;
    float3 normal;
    float3 contactPoint;
};

class NarrowPhase
{
public:
    NarrowPhase(std::vector<PotentialContact>& potentialContacts) {

    }

    // Support function for GJK
    float3 support(const Particle& particleA, const Particle& particleB, const float3& direction)
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

    bool gjkOverlap();

private:
    // Closest point on a line segment to origin
    bool closestPointOnLineToOrigin(const float3& a, const float3& b, float3& direction)
    {
        float3 ab = b - a;
        float3 ao = float3(0, 0, 0) - a;

        float t = ao & ab / magSquared(ab);

        if (t < 0)
        {
            // Closest to point A
            direction = a;
            return false;
        }
        if (t > 1)
        {
            // Closest to point B
            direction = b;
            return false;
        }
        // Closest to point on a line segment
        direction = a + ab * t;
        return true;
    }

    // Closest point on triangle to origin
    bool closestPointOnTriangleToOrigin(const float3& a, const float3& b, const float3& c, float3& direction)
    {
        // Check if origin is above the triangle
        float3 ab = b - a;
        float3 ac = c - a;
        float3 normal = ab^ac; // cross product;
        float3 ao = float3(0, 0, 0) - a;

        // If origin is on the same side as the normal
        float signedDist = ao&(normal);

        // Check edges
        bool onAB = closestPointOnLineToOrigin(a, b, direction);
        if (onAB) return true;

        bool onBC = closestPointOnLineToOrigin(b, c, direction);
        if (onBC) return true;

        bool onCA = closestPointOnLineToOrigin(c, a, direction);
        if (onCA) return true;

        // Check if origin projects onto the triangle
        // Barycentric coordinates
        float3 v0 = b - a;
        float3 v1 = c - a;
        float3 v2 = float3(0, 0, 0) - a;
        // dot product operator&
        float d00 = v0&v0;
        float d01 = v0&v1;
        float d11 = v1&v1;
        float d20 = v2&v0;
        float d21 = v2&v1;

        float denom = d00 * d11 - d01 * d01;
        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0f - v - w;

        if (v >= 0 && w >= 0 && u >= 0) {
            direction = a * u + b * v + c * w;
            return true;
        }

        return false;
    }

    // Closest point on tetrahedron to origin
    bool closestPointOnTetrahedronToOrigin(Simplex& simplex, float3& direction)
    {
        const float3& a = simplex[0];
        const float3& b = simplex[1];
        const float3& c = simplex[2];
        const float3& d = simplex[3];

        // Check if origin is inside the tetrahedron
        float3 ab = b - a;
        float3 ac = c - a;
        float3 ad = d - a;
        float3 ao = float3(0, 0, 0) - a;

        float3 abc = ab^(ac);
        float3 acd = ac^(ad);
        float3 adb = ad^(ab);

        // Check if origin is on the same side of all faces
        if ((abc & ao) > 0.0f && (acd & ao) > 0 && (adb & ao) > 0)
        {
            // Origin is inside tetrahedron
            direction = float3(0, 0, 0);
            return true;
        }

        // Check each face
        float3 closestPoint;
        float minDistSq = FLT_MAX;

        float3 tempDir;
        if (closestPointOnTriangleToOrigin(a, b, c, tempDir))
        {
            float distSq = magSquared(tempDir);
            if (distSq < minDistSq) {
                minDistSq = distSq;
                closestPoint = tempDir;
                simplex.clear();
                simplex.push_front(c);
                simplex.push_front(b);
                simplex.push_front(a);
            }
        }

        if (closestPointOnTriangleToOrigin(a, c, d, tempDir))
        {
            float distSq = magSquared(tempDir);
            if (distSq < minDistSq) {
                minDistSq = distSq;
                closestPoint = tempDir;
                simplex.clear();
                simplex.push_front(d);
                simplex.push_front(c);
                simplex.push_front(a);
            }
        }

        if (closestPointOnTriangleToOrigin(a, d, b, tempDir))
        {
            float distSq = magSquared(tempDir);
            if (distSq < minDistSq) {
                minDistSq = distSq;
                closestPoint = tempDir;
                simplex.clear();
                simplex.push_front(b);
                simplex.push_front(d);
                simplex.push_front(a);
            }
        }

        if (closestPointOnTriangleToOrigin(b, c, d, tempDir))
        {
            float distSq = magSquared(tempDir);
            if (distSq < minDistSq)
            {
                minDistSq = distSq;
                closestPoint = tempDir;
                simplex.clear();
                simplex.push_front(d);
                simplex.push_front(c);
                simplex.push_front(b);
            }
        }

        direction = closestPoint;
        return false;
    }

    // Process simplex and update direction
    bool processSimplex(Simplex& simplex, float3& direction) {
        if (simplex.get_size() == 1) {
            // Simplex is a point
            direction = float3(0, 0, 0) - simplex[0];
        } else if (simplex.get_size() == 2) {
            // Simplex is a line segment
            const float3& a = simplex[0]; // Last point added
            const float3& b = simplex[1]; // First point

            float3 ab = b - a;
            float3 ao = float3(0, 0, 0) - a;

            if ((ao & ab) > 0) {
                // Origin is in the direction of the line segment
                float3 abPerp = ab^ao^(ab);
                if (magSquared(abPerp) > 0) {
                    direction = abPerp;
                } else {
                    // Origin is on the line, pick any perpendicular direction
                    float3 perpDir = abs(ab.x) > abs(ab.y) ?
                        float3(0, 1, 0) : float3(1, 0, 0);
                    direction = ab^(perpDir);
                }
            } else {
                // Origin is behind a
                simplex.remove_point(1); // Remove b
                direction = ao;
            }
        } else if (simplex.get_size() == 3) {
            // Simplex is a triangle
            const float3& a = simplex[0]; // Last point added
            const float3& b = simplex[1];
            const float3& c = simplex[2];

            float3 ab = b - a;
            float3 ac = c - a;
            float3 ao = float3(0, 0, 0) - a;

            float3 abc = ab^ac;

            if (((abc^ac) & ao) > 0) {
                if ((ac & ao) > 0) {
                    // Origin is in the AC region
                    simplex.remove_point(1); // Remove b
                    direction = (ac^(ao))^(ac);
                } else {
                    if ((ab & ao) > 0) {
                        // Origin is in the AB region
                        simplex.remove_point(2); // Remove c
                        direction = (ab^(ao))^(ab);
                    } else {
                        // Origin is in the A region
                        simplex.set_size(1); // Keep only a
                        direction = ao;
                    }
                }
            } else {
                if (((ab ^ abc) & ao) > 0) {
                    if ((ab & ao) > 0) {
                        // Origin is in the AB region
                        simplex.remove_point(2); // Remove c
                        direction = (ab^ao)^(ab);
                    } else {
                        // Origin is in the A region
                        simplex.set_size(1); // Keep only a
                        direction = ao;
                    }
                } else {
                    if ((abc & ao) > 0) {
                        // Origin is above the triangle
                        direction = abc;
                    } else {
                        // Origin is below the triangle
                        simplex.remove_point(0); // Remove a
                        simplex.push_front(c); // Add c to front
                        simplex.remove_point(2); // Remove original c
                        direction = abc * -1;
                    }
                }
            }
        } else if (simplex.get_size() == 4) {
            // Simplex is a tetrahedron
            const float3& a = simplex[0]; // Last point added
            const float3& b = simplex[1];
            const float3& c = simplex[2];
            const float3& d = simplex[3];

            float3 ab = b - a;
            float3 ac = c - a;
            float3 ad = d - a;
            float3 ao = float3(0, 0, 0) - a;

            float3 abc = ab^(ac);
            float3 acd = ac^(ad);
            float3 adb = ad^(ab);

            if ((abc & ao) > 0) {
                // Origin is on the positive side of abc
                simplex.remove_point(3); // Remove d
                return processSimplex(simplex, direction);
            }

            if ( (acd & ao) > 0) {
                // Origin is on the positive side of acd
                simplex.remove_point(1); // Remove b
                return processSimplex(simplex, direction);
            }

            if ((adb & ao) > 0) {
                // Origin is on the positive side of adb
                simplex.remove_point(2); // Remove c
                return processSimplex(simplex, direction);
            }

            // Origin is inside the tetrahedron
            return true;
        }

        return false;
    }

    // Compute contact information
    Contact computeContact(const Particle& particleA, const Particle& particleB, const Simplex& simplex) {
        Contact contact;
        contact.pi = particleA;
        contact.pj = particleB;

        // EPA algorithm would go here for computing exact contact info
        // This is a simplified version

        // Compute contact normal (pointing from B to A)
        float3 centerA = particleA.position; // Assuming this method exists
        float3 centerB = particleB.position; // Assuming this method exists

        float3 normal = centerA - centerB;
        float length = mag(normal);

        if (length > 0.0001f)
        {
            normal = normal * (1.0f / length);
        } else
        {
            normal = float3(0, 1, 0); // Default normal if centers are too close
        }

        contact.normal = normal;

        // Compute contact point (midpoint of penetration)
        float radiusA = particleA.getRadius(); // Assuming this method exists
        float radiusB = particleB.getRadius(); // Assuming this method exists

        float3 pointOnA = centerA - normal * radiusA;
        float3 pointOnB = centerB + normal * radiusB;

        contact.contactPoint = (pointOnA + pointOnB) * 0.5f;

        return contact;
    }
};



#endif //NARROWPHASE_CUH
