//
// Created by iqraa on 28-2-25.
//

#ifndef ITREE_H
#define ITREE_H

#include <Tools/AABB/AABB.hpp>

template<typename T>
struct ITree
{
    int id = 0;
    int startId = 0;
    int endId = 0;
    bool isLeaf;
    AxisAlignedBoundingBox<T> bounds;

    __host__ __device__ bool Check(const T& point) const
    {
        return bounds.Check(point);
    }

    __host__ __device__ int particlesCountInNode() const
    {
        return endId - startId;
    }
};

#endif //ITREE_H
