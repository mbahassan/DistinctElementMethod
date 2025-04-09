//
// Created by mbahassan on 4/8/25.
//

#ifndef CUBEINSERTIONREGION_H
#define CUBEINSERTIONREGION_H

#include <Insertion/Domain/Domain.h>

#include "InsertionRegion.h"


// Cube region implementation
class CubeInsertionRegion : public InsertionRegion
{
public:
    CubeInsertionRegion(const float3& min, const float3& max) : min_(min), max_(max) {}
    explicit CubeInsertionRegion(const CubeRegion& region)
        : min_(region.getMin()), max_(region.getMax()) {}

    [[nodiscard]] bool contains(const BoundingBox<float3>& bbox) const override
    {
        return (bbox.min.x >= min_.x && bbox.max.x <= max_.x &&
                bbox.min.y >= min_.y && bbox.max.y <= max_.y &&
                bbox.min.z >= min_.z && bbox.max.z <= max_.z);
    }

    [[nodiscard]] float3 getMin() const override { return min_; }

    [[nodiscard]] float3 getMax() const override { return max_; }

    float getRandomX(std::mt19937& gen) const override
    {
        std::uniform_real_distribution dist(min_.x, max_.x);
        return dist(gen);
    }

    float getRandomY(std::mt19937& gen) const override
    {
        std::uniform_real_distribution dist(min_.y, max_.y);
        return dist(gen);
    }

    float getRandomZ(std::mt19937& gen) const override
    {
        std::uniform_real_distribution dist(min_.z, max_.z);
        return dist(gen);
    }

private:
    float3 min_;
    float3 max_;
};



#endif //CUBEINSERTIONREGION_H
