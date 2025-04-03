//
// Created by iqraa on 12-3-25.
//

#include "EPA.h"

#include <Particle/Polyhedral.h>

template<typename ParticleType>
EPA<ParticleType>::Triangle::Triangle(int a, int b, int c, const std::vector<float3>& vertices) {
    indices[0] = a;
    indices[1] = b;
    indices[2] = c;

    float3 ab = vertices[b] - vertices[a];
    float3 ac = vertices[c] - vertices[a];
    normal = normalize(ab ^ ac);
    distance = normal & vertices[a];

    if (distance < 0) {
        normal = -normal;
        distance = -distance;
        std::swap(indices[1], indices[2]);
    }
}

template<typename ParticleType>
bool EPA<ParticleType>::Triangle::isFrontFacing(const float3 &point, const std::vector<float3> &vertices) const{
    return ((point - vertices[indices[0]]) & normal) > 0;
}

template<typename ParticleType>
EPA<ParticleType>::Edge::Edge(int a_, int b_) : a(a_), b(b_) {
    if (a > b) std::swap(a, b);
}

template<typename ParticleType>
bool EPA<ParticleType>::Edge::operator==(const Edge& other) const {
    return a == other.a && b == other.b;
}

template<typename ParticleType>
bool EPA<ParticleType>::Edge::operator<(const Edge& other) const {
    return (a != other.a) ? (a < other.a) : (b < other.b);
}


// EPA implementation
template<typename ParticleType>
std::pair<float3, float> EPA<ParticleType>::ePAlgorithm(
    const ParticleType &particleA,
    const ParticleType &particleB,
    Simplex &gjkSimplex) {
    // Initialize EPA polytope with GJK simplex
    std::vector<float3> polytope = gjkSimplex.getPoints();
    std::vector<Triangle> faces;

    // If GJK simplex is not a tetrahedron, we need to expand it
    if (polytope.size() < 4) {
        // Find some additional support points to form a tetrahedron
        float3 dirs[6] = {
            float3(1, 0, 0), float3(-1, 0, 0),
            float3(0, 1, 0), float3(0, -1, 0),
            float3(0, 0, 1), float3(0, 0, -1)
        };

        for (int i = 0; i < 6 && polytope.size() < 4; ++i) {
            float3 supportPoint = sATMB(particleA, particleB, dirs[i]);

            // Check if this point is unique
            bool unique = true;
            for (const auto &point: polytope) {
                if (magSquared(point - supportPoint) < 0.0001f) {
                    unique = false;
                    break;
                }
            }

            if (unique) {
                polytope.push_back(supportPoint);
            }
        }
    }

    // Create initial tetrahedron
    if (polytope.size() >= 4) {
        faces.emplace_back(0, 1, 2, polytope);
        faces.emplace_back(0, 2, 3, polytope);
        faces.emplace_back(0, 3, 1, polytope);
        faces.emplace_back(1, 3, 2, polytope);
    } else {
        // Failed to create a proper simplex, return a default result
        return std::make_pair(float3(0, 1, 0), 0.0f);
    }

    float EPSILON = 0.0001f;
    float closestDistance = std::numeric_limits<float>::max();
    float oldClosestDistance = std::numeric_limits<float>::max();
    size_t closestFaceIndex = 0;

    // Main EPA loop
    int maxIterations = 32; // Prevent infinite loops
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Find closest face to origin
        closestDistance = std::numeric_limits<float>::max();
        for (size_t i = 0; i < faces.size(); ++i) {
            if (faces[i].distance < closestDistance) {
                closestDistance = faces[i].distance;
                closestFaceIndex = i;
            }
        }

        // Check for convergence
        if (std::abs(closestDistance - oldClosestDistance) < EPSILON) {
            // We've found the closest face
            break;
        }

        oldClosestDistance = closestDistance;

        // Get search direction (normal of closest face)
        float3 searchDir = faces[closestFaceIndex].normal;

        // Get new support point
        float3 supportPoint = sATMB(particleA, particleB, searchDir);

        // Check if we're expanding in the right direction
        float projDistance = supportPoint & searchDir;
        if (projDistance - closestDistance < EPSILON) {
            // We've reached the boundary, exit
            break;
        }

        // Add new point to polytope
        size_t newPointIndex = polytope.size();
        polytope.push_back(supportPoint);

        // Remove faces that can see the new point
        std::vector<Edge> uniqueEdges;
        for (size_t i = faces.size() - 1; i-- > 0;) {
            if (faces[i].isFrontFacing(supportPoint, polytope)) {
                // This face can see the new point, so it will be removed
                // Save its edges for creating new faces
                Edge e1(faces[i].indices[0], faces[i].indices[1]);
                Edge e2(faces[i].indices[1], faces[i].indices[2]);
                Edge e3(faces[i].indices[2], faces[i].indices[0]);

                for (auto &e: {e1, e2, e3}) {
                    auto it = std::find(uniqueEdges.begin(), uniqueEdges.end(), e);
                    if (it == uniqueEdges.end()) {
                        uniqueEdges.push_back(e);
                    } else {
                        uniqueEdges.erase(it); // If edge appears twice, it's internal
                    }
                }

                // Remove this face
                faces[i] = faces.back();
                faces.pop_back();
            }
        }

        // Create new faces using unique edges
        for (const auto &edge: uniqueEdges) {
            faces.emplace_back(edge.a, edge.b, newPointIndex, polytope);
        }
    }

    // The closest face normal is our contact normal
    float3 normal = faces[closestFaceIndex].normal;
    float penetrationDepth = closestDistance;

    return std::make_pair(normal, penetrationDepth);
}

template<typename ParticleType>
Contact EPA<ParticleType>::computeContactEPA(
    const ParticleType &particleA,
    const ParticleType &particleB,
    Simplex &gjkSimplex) {
    Contact contact;
    contact.pi = particleA.getId();
    contact.pj = particleB.getId();

    // Use EPA to find penetration depth and normal
    auto epaResult = ePAlgorithm(particleA, particleB, gjkSimplex);
    float3 normal = epaResult.first;
    float penetrationDepth = epaResult.second;

    // The normal from EPA points from B to A
    contact.normal = normal;
    contact.penetrationDepth = penetrationDepth;

    // Get positions of particles
    float3 posA = particleA.position;
    float3 posB = particleB.position;

    // For arbitrary shapes, get actual support points rather than using radius
    float3 deepestPointA = particleA.supportMapping(-normal) + posA;
    float3 deepestPointB = particleB.supportMapping(normal) + posB;

    // Calculate contact point as the midpoint of the penetration
    contact.contactPoint = deepestPointA + (deepestPointB - deepestPointA) * 0.5f;

    return contact;
}


// Support function for EPA - similar to GJK's support function
template<typename ParticleType>
float3 EPA<ParticleType>::sATMB(const ParticleType &particleA, const ParticleType &particleB, const float3 &direction) {
    // Get the furthest point of particle A in direction
    float3 supportA = particleA.supportMapping(direction) + particleA.position;

    // Get the furthest point of particle B in opposite direction
    float3 supportB = particleB.supportMapping(-direction) + particleB.position;

    // Return the Minkowski difference
    return supportA - supportB;
}


// At the end of EPA.cpp
template Contact EPA<Polyhedral>::computeContactEPA(const Polyhedral& a, const Polyhedral& b, Simplex& simplex);