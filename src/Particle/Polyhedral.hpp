#ifndef POLYHEDRAL_PARTICLE_H
#define POLYHEDRAL_PARTICLE_H

#include <cuda_runtime_api.h>


#include "Material/Material.hpp"
#include "Shape/Polytope/Polytope.hpp"
#include "Tools/quaternion/quaternion.hpp"
#include "Tools/AABB/AABB.hpp"
#include <Tools/ArthmiticOperator/MathOperators.hpp>
#include <Tools/Position/Position.h>




class Polyhedral : public Material, public Polytope
{
public:

    __host__ __device__
    Polyhedral() = default;

    __host__ __device__
    Polyhedral(const Polyhedral &);

    __host__ __device__
    Polyhedral(Polyhedral &particle)
    {
        position = particle.position;
        velocity = particle.velocity;
        acceleration = particle.acceleration;
        orientation = particle.orientation;
        angularVel = particle.angularVel;
        angularAcc = particle.angularAcc;
        force = particle.force;
        boundingBox = particle.boundingBox;

        boundingBox.min = particle.position - particle.getMin();
        boundingBox.max = particle.position + particle.getMax();
    }

    Polyhedral(const Material& material, const Polytope& shape);

    Polyhedral(Material& material, Polytope& shape);


    // Other methods remain the same ...
    ~Polyhedral() override = default;

    // Orientation-specific methods
    void setOrientation(const Quaternion& q) { orientation = q; }

    float3 getAxisDirection() const; // Returns the current axis direction of the particle

    float3 position  ;      // Position in 3D space

    float3 velocity {0.f,0.f,0.f};      // Linear velocity

    float3 acceleration {0.f,0.f,0.f};  // Linear acceleration

    Quaternion orientation; // Rotational orientation - CRUCIAL for cylinders

    float3 angularVel {0.f,0.f,0.f};   // Angular velocity

    float3 angularAcc {0.f,0.f,0.f};   // Angular acceleration

    float3 force {0.f,0.f,0.f};

    float mass = getVolume() * getDensity();

    float3 torque  {0.f,0.f,0.f};

    float inertia= 0.f;

    AxisAlignedBoundingBox<float3> boundingBox;

private:

    // Allow Position<Polyhedral> to access private members [update(float3)].
    // friend class Position<Polyhedral>;

    // void update(float3 updatedPosition) {
    //     // Add the new position to all vertices
    //
    //     // Update center of mass
    //
    //     // Update BBox
    //     return ;
    // }

};


#endif //POLYHEDRAL_PARTICLE_H