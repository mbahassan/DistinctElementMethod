//
// Created by iqraa on 28-2-25.
//

#ifndef ITREE_H
#define ITREE_H

#include <Tools/AABB/AABB.hpp>

template<typename T>
struct ITree
{
    int id;
    int startId;
    int endId;
    AxisAlignedBoundingBox<T> bounds;

    __host__ __device__ bool Check(const T& point) const
    {
        return bounds.Check(point);
    }

    __host__ __device__ int maxParticlesPerNode() const
    {
        return endId - startId;
    }
};

#endif //ITREE_H
