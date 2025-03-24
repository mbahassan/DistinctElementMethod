#ifndef SPHERICAL_PARTICLE_H
#define SPHERICAL_PARTICLE_H

#include <cuda_runtime_api.h>


#include "Material/Material.hpp"
#include "Shape/Sphere/Sphere.hpp"
#include "Tools/quaternion/quaternion.hpp"
#include "Tools/AABB/AABB.hpp"
#include <Tools/ArthmiticOperator/MathOperators.hpp>



class Spherical : public Material, public Sphere
{
public:

    __host__ __device__
    Spherical() = default;

    __host__ __device__
    Spherical(const Spherical &particle)
     : Material(particle), Sphere(particle) {
        position = particle.position;
        velocity = particle.velocity;

        boundingBox.min = particle.position - particle.getMin();
        boundingBox.max = particle.position + particle.getMax();
    };

    __host__ __device__
    Spherical(Spherical &particle)
     : Material(particle), Sphere(particle) {
        position = particle.position;
        velocity = particle.velocity;

        boundingBox.min = particle.position - particle.getMin();
        boundingBox.max = particle.position + particle.getMax();
    }

    Spherical(const Material& material, const Sphere& shape);

    Spherical(const Material& material, Sphere& shape);

    
    // Other methods remain the same ...
    ~Spherical() override = default;

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

    AxisAlignedBoundingBox<float3> boundingBox {};


};

#endif //SPHERICAL_PARTICLE_H