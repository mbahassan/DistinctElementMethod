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

    ~Sphere();

    void setRadius(float radius);

    float getRadius() const override;

    float getVolume() const;


private:
    float radius_ = 0;

    AxisAlignedBoundingBox<float3> boundingBox_;
};



#endif //SPHERE_H
