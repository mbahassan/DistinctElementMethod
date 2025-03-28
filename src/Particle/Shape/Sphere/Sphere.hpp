//
// Created by iqraa on 14-2-25.
//

#ifndef SPHERE_H
#define SPHERE_H

#include <Particle/Shape/Shape.hpp>
#include "Tools/AABB/AABB.hpp"

class Sphere : public Shape
{
public:

    __host__ __device__
    Sphere()
    {
        setShapeType(SPHERE);
    }

    __host__ __device__
    Sphere(const Sphere& sphere)
    {
        radius_ = sphere.radius_;
        setShapeType(SPHERE);
    };

    /// Constructor by variable/file input
    explicit Sphere(float radius): radius_(radius)
    {
        setShapeType(SPHERE);
    }

    __host__ __device__
    ~Sphere() override = default;

    __host__ __device__
    void setRadius(float radius) {radius_ = radius;}

    __host__ __device__
    float getRadius() const {return radius_;}

    __host__ __device__
    float3 getMin() override {return make_float3(radius_, radius_ ,radius_);}

    __host__ __device__
    float3 getMin() const {return make_float3(radius_, radius_ ,radius_);}

    __host__ __device__
    float3 getMax() override {return make_float3(radius_, radius_ ,radius_);}

    __host__ __device__
    float3 getMax() const {return make_float3(radius_, radius_ ,radius_);}

    __host__ __device__
    float getVolume() override {return 4.0f * radius_*radius_*radius_ / 3.0f;}

    __host__ __device__
    float3 supportMapping(const float3& direction) const
    {
        return normalize(direction) * radius_;
    }


private:
    float radius_ = 0;

};



#endif //SPHERE_H
