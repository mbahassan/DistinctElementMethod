//
// Created by iqraa on 12-3-25.
//

#ifndef EPA_CUH
#define EPA_CUH

#include <vector>
#include <limits>
#include "Particle/Spherical.h"
#include "ContactDetection/NarrowPhase/Simplex/Simplex.h"

struct Contact
{
    int pi;
    int pj;
    float3 normal;
    float3 contactPoint;
    float penetrationDepth;
};

template<typename ParticleType>
class EPA
{
public:

    EPA() = default;


    // EPA implementation
    static std::pair<float3, float> ePAlgorithm(
        const ParticleType& particleA,
        const ParticleType& particleB,
        Simplex& gjkSimplex);


    // Updated contact computation for arbitrary shapes
    static Contact computeContactEPA(
        const ParticleType& particleA,
        const ParticleType& particleB,
        Simplex& gjkSimplex) ;

private:
    struct Triangle {
        int indices[3]{}; // Indices to vertices in the polytope
        float3 normal{};  // Outward facing normal
        float distance; // Distance from origin to face along normal

        __host__ __device__
        Triangle(int a, int b, int c, const std::vector<float3>& vertices);

        bool isFrontFacing(const float3& point, const std::vector<float3>& vertices) const;
    };

    struct Edge
    {
        int a, b;

        Edge(int a_, int b_) ;

        bool operator==(const Edge& other) const;

        bool operator<(const Edge& other) const ;
    };

    // Support function for EPA - similar to GJK's support function
    static float3 sATMB(const ParticleType& particleA, const ParticleType& particleB, const float3& direction);
};

#endif // EPA_CUH