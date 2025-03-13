//
// Created by iqraa on 14-2-25.
//

#ifndef SPHERE_H
#define SPHERE_H

#include <Particle/Shape/Shape.hpp>

#include "Tools/AABB/AABB.hpp"

class Sphere :public Shape
{
public:
    Sphere();

    explicit Sphere(float radius);

    Sphere(const Sphere& sphere);

    ~Sphere() = default;

    void setRadius(float radius);

    __host__ __device__
    float getRadius() const {return radius_;}

    float getVolume() const;

    float3 supportMapping(const float3& direction) const
    {
        return normalize(direction) * radius_;
    }


private:
    float radius_ = 0;

    AxisAlignedBoundingBox<float3> boundingBox_;
};



#endif //SPHERE_H
