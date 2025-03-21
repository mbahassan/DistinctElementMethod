#ifndef POLYHEDRAL_PARTICLE_H
#define POLYHEDRAL_PARTICLE_H

#include <cuda_runtime_api.h>


#include "Material/Material.hpp"
#include "Shape/Polytope/Polytope.hpp"
#include "Tools/Quaternion.hpp"
#include "Tools/AABB/AABB.hpp"
#include <Tools/ArthmiticOperator/MathOperators.hpp>
#include <Tools/Position/Position.h>




class Polyhedral : public Material, public Polytope
{
public:
    // Add this constructor
    Polyhedral() = default;

    __host__ __device__
    // Polyhedral() = default;

    Polyhedral(const Material& material, const Polytope& shape);

    __host__ __device__
    Polyhedral(const Polyhedral &);

    Polyhedral(const Material& material, Polytope& shape);

    __host__ __device__
    Polyhedral(Polyhedral &particle)
    {
        position = particle.position;
        velocity = particle.velocity;

        boundingBox.min = particle.position - particle.getMin();
        boundingBox.max = particle.position + particle.getMax();
    }

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
    friend class Position<Polyhedral>;

    void update(float3 updatedPosition) {
        // Add the new position to all vertices

        // Update center of mass

        // Update BBox
        return ;
    }

};


#endif //POLYHEDRAL_PARTICLE_H