//
// Created by mbahassan on 4/8/25.
//

#ifndef INSERTIONREGION_H
#define INSERTIONREGION_H

#include <random>
#include <vector_types.h>
#include <Tools/AABB/AABB.hpp>


// Base class for insertion regions
class InsertionRegion
{
public:
    virtual ~InsertionRegion() = default;
    virtual bool contains(const BoundingBox<float3>& bbox) const = 0;
    virtual float3 getMin() const = 0;
    virtual float3 getMax() const = 0;
    virtual float getRandomX(std::mt19937& gen) const = 0;
    virtual float getRandomY(std::mt19937& gen) const = 0;
    virtual float getRandomZ(std::mt19937& gen) const = 0;
};



#endif //INSERTIONREGION_H
