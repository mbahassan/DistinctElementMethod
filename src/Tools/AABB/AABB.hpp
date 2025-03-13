//
// Created by mbahassan on 2/28/25.
//

#ifndef AABB_H
#define AABB_H
#include <Tools/ArthmiticOperator/MathOperators.hpp>
template <typename T>
struct AxisAlignedBoundingBox
{
    T min;
    T max;

    __host__ __device__ bool Check(const T& point) const
    {
        return point >= min && point < max;
    }

    __host__ __device__ T getCenter() const
    {
        return (min + max) * 0.5f;
    }

    __host__ __device__ T getSize() const
    {
        return max - min;
    }
};
#endif //AABB_H
