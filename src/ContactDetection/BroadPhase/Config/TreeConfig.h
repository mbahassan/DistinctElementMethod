//
// Created by iqraa on 28-2-25.
//

#ifndef TREECONFIG_H
#define TREECONFIG_H

#include "ContactDetection/BroadPhase/Config/TreeType.h"

struct TreeConfig
{
    TreeType type = QUADTREE;
    float3 origin = {0.0f,0.0f,0.0f};
    float3 size =  {1.0f,1.0f,0.0f};

    int depth = -1;
    int maxDepth ;
    int minPointsPerNode ;
    int numNodesAtThisLevel = 0;
    const int threadsPerBlock = 128;//256;//128;

    // Copy assignment operator
    TreeConfig& operator=(const TreeConfig& other) {
        if (this != &other) { // Prevent self-assignment
            this->maxDepth = other.maxDepth;
            this->minPointsPerNode = other.minPointsPerNode;
        }
        return *this;
    }

    // Constructor set to default values.
    __host__ __device__
    TreeConfig():
          depth(0),
          maxDepth(0),
          minPointsPerNode(0),
          numNodesAtThisLevel(1) {}

    __host__ __device__
    TreeConfig(int maxDepth, int minPointsPerNode):
          depth(0),
          maxDepth(maxDepth),
          minPointsPerNode(minPointsPerNode),
          numNodesAtThisLevel(1) {}

    // Copy constructor. Changes the values for next iteration.
    __host__ __device__
    TreeConfig(const TreeConfig &params, bool):
          depth(params.depth + 1),
          maxDepth(params.maxDepth),
          minPointsPerNode(params.minPointsPerNode),
          numNodesAtThisLevel(4 * params.numNodesAtThisLevel) {}


    int GetNodesCount() const
    {
        int maxNodes = 0;
        // for (int i = 0; i < maxDepth; ++i)
        // {
        //     const int val  = type == QUADTREE ? 4 : 8;
        //     maxNodes += std::pow(val, i);
        // }

        for (int i = 0, num_nodes_at_level = 1; i < maxDepth; ++i, num_nodes_at_level *= 4)
            maxNodes += num_nodes_at_level;

        return maxNodes;
    }
} ;


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
