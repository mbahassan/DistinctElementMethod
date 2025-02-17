#ifndef PARTICLE_LIBRARY_H
#define PARTICLE_LIBRARY_H

#include <cuda_runtime.h>
#include <memory>

#include "Particle/Shape/Shape.hpp"
#include "Material/Material.hpp"
#include "Tools/Quaternion.hpp"

class Particle :public Shape
{
public:
    //TypeName("Particle");

    Particle();

    Particle(Material& material, Shape& shape);

    Particle(const Particle &);

    Particle(Material& material, Shape& shape, float3 position, float3 velocity);

    // Constructors
    Particle(std::unique_ptr<Shape> shape, std::shared_ptr<Material> material);

    // Other methods remain the same...
    ~Particle();

    void setMaterial(Material& material);

    void setShape(Shape& shape);

    // Orientation-specific methods
    void setOrientation(const Quaternion& q) { orientation = q; }

    void setPosition(float3 position);

    void setVelocity(float3 velocity);

    float3 getPosition();

    float3 getVelocity();

    Quaternion getOrientation() const { return orientation; }

    float3 getAxisDirection() const; // Returns the current axis direction of the particle
private:

    /// Material for the Particle
    std::shared_ptr<Material> material_;

    /// Shape of the Particle [e.g. Sphere, Cylinder, or from mesh]
    std::unique_ptr<Shape> shape;

    float3 position_ {0.f,0.f,0.f};      // Position in 3D space

    float3 velocity_ {0.f,0.f,0.f};      // Linear velocity

    float3 acceleration;  // Linear acceleration

    Quaternion orientation; // Rotational orientation - CRUCIAL for cylinders

    float3 angularVel;   // Angular velocity

    float3 angularAcc;   // Angular acceleration
};

#endif //PARTICLE_LIBRARY_H