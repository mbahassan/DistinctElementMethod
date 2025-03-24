#ifndef POLYHEDRAL_PARTICLE_H
#define POLYHEDRAL_PARTICLE_H

#include <cuda_runtime_api.h>

#include "Material/Material.hpp"
#include "Shape/Polytope/Polytope.hpp"
#include "Tools/quaternion/quaternion.hpp"
#include "Tools/AABB/AABB.hpp"

class Polyhedral : public Material, public Polytope {
public:
    __host__ __device__
    Polyhedral() = default;

    /// Move Constructors
    Polyhedral(const Material& material, const Polytope& polytope)
        : Material(material), Polytope(polytope) // Explicitly call copy constructors
    {
        initializeProperties();
    }

    Polyhedral(Material& material, Polytope& polytope)
        : Material(material), Polytope(polytope) // Explicitly call copy constructors
    {
        initializeProperties();
    }

    /// Destructors
    ~Polyhedral() override = default;

    // Get face with bounds checking
    Face getFace(int i) const {
        return Polytope::getFace(i);
    }

    void setOrientation(const Quaternion& q) { orientation = q; }

    float3 getAxisDirection() const; // Returns the current axis direction of the particle

    float3 position{0.f, 0.f, 0.f};      // Position in 3D space
    float3 velocity{0.f, 0.f, 0.f};      // Linear velocity
    float3 acceleration{0.f, 0.f, 0.f};  // Linear acceleration
    float3 angularVel{0.f, 0.f, 0.f};   // Angular velocity
    float3 angularAcc{0.f, 0.f, 0.f};   // Angular acceleration
    float3 force{0.f, 0.f, 0.f};
    float mass = 0.f;
    float3 torque{0.f, 0.f, 0.f};
    float inertia = 0.f;
    AxisAlignedBoundingBox<float3> boundingBox;
    Quaternion orientation;
private:
    void initializeProperties() {
        // Initialize mass based on current state
        mass = getVolume() * getDensity();

        // Initialize bounding box
        float3 min = getMin();
        float3 max = getMax();
        boundingBox = AxisAlignedBoundingBox<float3>(min, max);
    }
};

#endif //POLYHEDRAL_PARTICLE_H