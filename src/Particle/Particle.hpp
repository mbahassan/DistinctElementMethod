#ifndef PARTICLE_LIBRARY_H
#define PARTICLE_LIBRARY_H

#include <cuda_runtime_api.h>


#include "Material/Material.hpp"
#include "Shape/Sphere/Sphere.hpp"
#include "Tools/Quaternion.hpp"
#include "Tools/AABB/AABB.hpp"
#include <Tools/ArthmiticOperator/MathOperators.hpp>

class Particle final : public Material, public Sphere
{
public:

    __host__ __device__
    Particle() = default;

    Particle(const Material& material, const Sphere& shape);

    __host__ __device__
    Particle(const Particle &);

    Particle(const Material& material, Sphere& shape);

    __host__ __device__
    Particle(Particle &particle)
    {
        position = particle.position;
        velocity = particle.velocity;

        boundingBox.min = particle.position - particle.getRadius();
        boundingBox.max = particle.position + particle.getRadius();
    }

    // Other methods remain the same ...
    ~Particle() override = default;

    // Orientation-specific methods
    void setOrientation(const Quaternion& q) { orientation = q; }

    Quaternion getOrientation() const {return orientation;}

    float3 getAxisDirection() const; // Returns the current axis direction of the particle

    float3 position {0.f,0.f,0.f};      // Position in 3D space

    float3 velocity {0.f,0.f,0.f};      // Linear velocity

    float3 acceleration {0.f,0.f,0.f};  // Linear acceleration

    Quaternion orientation; // Rotational orientation - CRUCIAL for cylinders

    float3 angularVel {0.f,0.f,0.f};   // Angular velocity

    float3 angularAcc {0.f,0.f,0.f};   // Angular acceleration

    float3 force {0.f,0.f,0.f};

    float mass = getVolume() * getDensity();

    float3 torque = {0.f,0.f,0.f};

    float inertia= 0.f;

    AxisAlignedBoundingBox<float3> boundingBox;

private:


};

#endif //PARTICLE_LIBRARY_H