#ifndef PARTICLE_LIBRARY_H
#define PARTICLE_LIBRARY_H

#include <cuda_runtime.h>


#include "Material/Material.hpp"
#include "Shape/Sphere/Sphere.hpp"
#include "Tools/Quaternion.hpp"

class Particle final :public Material, public Sphere
{
public:

    Particle();

    Particle(const Material& material, const Sphere& shape);

    Particle(const Particle &);

    Particle(const Material& material, Sphere& shape);

    Particle( Particle &);

    // Other methods remain the same...
    ~Particle() = default;


    // Orientation-specific methods
    void setOrientation(const Quaternion& q) { orientation = q; }

    void setPosition(float3 position);

    void setVelocity(float3 velocity);

    float3 getPosition() const;

    float3 getVelocity() const;

    Quaternion getOrientation() const { return orientation; }

    float3 getAxisDirection() const; // Returns the current axis direction of the particle

private:

    float3 position_ {0.f,0.f,0.f};      // Position in 3D space

    float3 velocity_ {0.f,0.f,0.f};      // Linear velocity

    float3 acceleration {0.f,0.f,0.f};  // Linear acceleration

    Quaternion orientation; // Rotational orientation - CRUCIAL for cylinders

    float3 angularVel {0.f,0.f,0.f};   // Angular velocity

    float3 angularAcc {0.f,0.f,0.f};   // Angular acceleration
};

#endif //PARTICLE_LIBRARY_H