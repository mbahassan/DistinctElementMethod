//
// Created by mbahassan on 4/8/25.
//

#ifndef CYLINDRICAL_REGION_H
#define CYLINDRICAL_REGION_H


#include "InsertionRegion.h"

// Cylindrical region implementation
class CylindricalInsertionRegion : public InsertionRegion
{
public:
    CylindricalInsertionRegion(const float3& center, float radius, float height)
        : center_(center), radius_(radius), height_(height)
    {
        min_ = {center_.x - radius_, center_.y - radius_, center_.z};
        max_ = {center_.x + radius_, center_.y + radius_, center_.z + height_};
    }

    bool contains(const BoundingBox<float3>& bbox) const override
    {
        // Check height bounds first
        if (bbox.min.z <= center_.z || bbox.max.z > center_.z + height_) {
            return false;
        }

        // Calculate the maximum distance from the center axis
        float dx_min = bbox.min.x - center_.x;
        float dy_min = bbox.min.y - center_.y;
        float dx_max = bbox.max.x - center_.x;
        float dy_max = bbox.max.y - center_.y;

        // Calculate the corners of the bounding box projection on the XY plane
        float corners_dist_sq[4] = {
            dx_min * dx_min + dy_min * dy_min,
            dx_min * dx_min + dy_max * dy_max,
            dx_max * dx_max + dy_min * dy_min,
            dx_max * dx_max + dy_max * dy_max
        };

        // Find the maximum distance
        float max_dist_sq = corners_dist_sq[0];
        for (int i = 1; i < 4; i++) {
            if (corners_dist_sq[i] > max_dist_sq) {
                max_dist_sq = corners_dist_sq[i];
            }
        }

        // Check if the maximum distance is less than or equal to the radius
        return max_dist_sq <= radius_ * radius_;
    }

    float3 getMin() const override { return min_; }
    float3 getMax() const override { return max_; }

    float getRandomX(std::mt19937& gen) const override
    {
        std::uniform_real_distribution<float> dist(-radius_, radius_);
        float r = dist(gen);
        return center_.x + r;
    }

    float getRandomY(std::mt19937& gen) const override {
        std::uniform_real_distribution<float> dist(-radius_, radius_);
        float r = dist(gen);
        return center_.y + r;
    }

    float getRandomZ(std::mt19937& gen) const override {
        std::uniform_real_distribution<float> dist(0, height_);
        return center_.z + dist(gen);
    }

private:
    float3 center_;
    float radius_;
    float height_;
    float3 min_ {};
    float3 max_ {};
};



#endif //CYLINDRICAL_REGION_H
