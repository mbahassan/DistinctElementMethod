//
// Created by iqraa on 28-2-25.
//

#ifndef TREECONFIG_H
#define TREECONFIG_H

#include "ContactDetection/BroadPhase/Config/TreeType.h"

struct TreeConfig
{
    TreeType type = QUADTREE;
    float3 origin;
    float3 size;
    int maxDepth = 2;

    const int threadsPerBlock = 128;
    int minPointsToDivide= 3;

    inline int GetNodesCount() const
    {
        int maxNodes = 0;
        for (int i = 0; i < maxDepth; ++i)
        {
            maxNodes += std::pow(type == TreeType::QUADTREE ? 4 : 8, i);
        }

        return maxNodes;
    }
};


template<int DIMENSION>
__host__ __device__ static int getNumNodesInCurrentDepth(const int depth)
{
    return 0;
}

template<>
__host__ __device__ inline int getNumNodesInCurrentDepth<2>(const int depth)
{
    int sum = 1;
    for (int i = 0; i < depth; ++i)
    {
        sum *= 4;
    }

    return sum;
}

template<>
__host__ __device__ inline int getNumNodesInCurrentDepth<3>(const int depth)
{
    int sum = 1;
    for (int i = 0; i < depth; ++i)
    {
        sum *= 8;
    }

    return sum;
}

#endif //TREECONFIG_H
