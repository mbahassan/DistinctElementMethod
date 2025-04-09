//
// Created by mbahassan on 4/8/25.
//

#ifndef DOMAIN_H
#define DOMAIN_H

#include <vector_types.h>


struct Domain
{
private:
    float3 min_ {0.f,0.f,0.f};
    float3 max_ {0.f,0.f,0.f};

public:
    Domain() = default;
    Domain(const float3 _min, const float3 _max) : min_(_min), max_(_max) {}

    [[nodiscard]] float3 getMin() const { return min_; }
    [[nodiscard]] float3 getMax() const { return max_; }

     void setMin(const float3 min) { min_= min; }
     void setMax(const float3 max) { max_= max; }
};

struct CubeRegion: Domain
{

};

#endif //DOMAIN_H
