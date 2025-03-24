#ifndef POLYHEDRAL_PARTICLE_H
#define POLYHEDRAL_PARTICLE_H

#include <cuda_runtime_api.h>


#include "Material/Material.hpp"
#include "Shape/Polytope/Polytope.hpp"
#include "Tools/quaternion/quaternion.hpp"
#include "Tools/AABB/AABB.hpp"
#include <Tools/ArthmiticOperator/MathOperators.hpp>


class Polyhedral : public Material, public Polytope
{
public:

    __host__ __device__
    Polyhedral() = default;


    __host__ __device__
    Polyhedral(Polyhedral &particle)
    {
        position = particle.position;
        velocity = particle.velocity;
        acceleration = particle.acceleration;
        angularVel = particle.angularVel;
        angularAcc = particle.angularAcc;
        force = particle.force;
        boundingBox = particle.boundingBox;

        boundingBox.min = particle.position - particle.getMin();
        boundingBox.max = particle.position + particle.getMax();
    }

    /// Move Constructors
    Polyhedral(const Material& material, const Polytope& polytope);

    Polyhedral(Material& material, Polytope& polytope);


    /// Destructors
    ~Polyhedral() override = default;


    float3 getAxisDirection() const; // Returns the current axis direction of the particle

    float3 position  ;      // Position in 3D space

    float3 velocity {0.f,0.f,0.f};      // Linear velocity

    float3 acceleration {0.f,0.f,0.f};  // Linear acceleration

    float3 angularVel {0.f,0.f,0.f};   // Angular velocity

    float3 angularAcc {0.f,0.f,0.f};   // Angular acceleration

    float3 force {0.f,0.f,0.f};

    float mass = getVolume() * getDensity();

    float3 torque  {0.f,0.f,0.f};

    float inertia= 0.f;

    AxisAlignedBoundingBox<float3> boundingBox;


};


#endif //POLYHEDRAL_PARTICLE_H