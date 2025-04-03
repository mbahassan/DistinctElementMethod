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
    Spherical(const Spherical& spherical)
    {
        position = spherical.position;
        velocity = spherical.velocity;
        boundingBox.min = spherical.boundingBox.min;
        boundingBox.max = spherical.boundingBox.max;

        calculateMass();
    };

    __host__ __device__
    Spherical(Spherical &spherical)
    {
        position = spherical.position;
        velocity = spherical.velocity;
        boundingBox.min = spherical.boundingBox.min;
        boundingBox.max = spherical.boundingBox.max;

        calculateMass();
    }

    __host__ __device__
    Spherical(const Material& material, const Sphere& sphere):
    Material(material), Sphere(sphere)
    {
        calculateMass();
    }


    __host__ __device__
    Spherical(Material& material, Sphere& sphere):
    Material(material), Sphere(sphere)
    {
        boundingBox.min = position - Sphere::getMin();
        boundingBox.max = position + Sphere::getMax();
        calculateMass();
    }

    __host__ __device__
    ~Spherical() override = default;

    float3 supportMapping(const float3 &direction) const
    {
        const float vecMag = mag(direction);
        float3 s = {0.0f,0.0f,0.0f};

        if (vecMag > 0.0f) {
            s = normalize(direction) * getRadius();
        }

        return s;
    }

    float3 position {0.f,0.f,0.f};      // Position in 3D space

    float3 velocity {0.f,0.f,0.f};      // Linear velocity

    float3 acceleration {0.f,0.f,0.f};  // Linear acceleration

    Quaternion orientation; // Rotational orientation - CRUCIAL for cylinders

    float3 angularVel {0.f,0.f,0.f};   // Angular velocity

    float3 angularAcc {0.f,0.f,0.f};   // Angular acceleration

    float3 force {0.f,0.f,0.f};

    float mass = 0.0f;

    float3 torque = {0.f,0.f,0.f};

    float inertia= 0.f;

    BoundingBox<float3> boundingBox {};

private:

    __host__ __device__
    void calculateMass() {
        mass = getVolume() * getDensity();
    }
};

#endif //SPHERICAL_PARTICLE_H