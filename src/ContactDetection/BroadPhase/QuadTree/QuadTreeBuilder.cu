//
// Created by iqraa on 28-2-25.
//



#include <iostream>
#include <chrono>

#include "QuadTreeBuilder.cuh"
#include "QuadTreeBuilderKernel.cuh"
#include "Tools/CudaHelper.hpp"
#include "Tools/ArthmiticOperator/MathOperators.hpp"

template<typename ParticleType>
QuadTreeBuilder<ParticleType>::QuadTreeBuilder(const TreeConfig& treeConfig)
    : treeConfig {treeConfig}
{
}

template<typename ParticleType>
void QuadTreeBuilder<ParticleType>::initialize(const int capacity)
{
    cudaMalloc((void**)&pointsExch, capacity * sizeof(Spherical));
    GET_CUDA_ERROR("cudaMalloc() pointsExch");

    // tree:
    const int maxNodes = treeConfig.GetNodesCount();
    cudaMallocManaged((void**)&this->tree, maxNodes * sizeof(QuadTree)); // plus one for the root
    GET_CUDA_ERROR("cudaMallocManaged() tree");
    cudaDeviceSynchronize();
    GET_CUDA_ERROR("cudaDeviceSynchronize");
}

template<typename ParticleType>
void QuadTreeBuilder<ParticleType>::build(ParticleType* points, const int size)
{
    reset();

    this->tree->id = 0;
    this->tree->bounds.min = treeConfig.origin;
    const auto maxDim = treeConfig.origin + treeConfig.size;
    this->tree->bounds.max = maxDim;
    this->tree->startId = 0;
    this->tree->endId = size;

    std::cout << "Build()" << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    const int warpsPerBlock = treeConfig.threadsPerBlock / 32;

    cudaMemcpy(pointsExch, points, size * sizeof(ParticleType), cudaMemcpyDeviceToDevice);

    QuadTreeKernel<128, ParticleType><<<1, treeConfig.threadsPerBlock, warpsPerBlock * 4 * sizeof(int)>>>
    (
        this->tree,
        points,
        pointsExch,
        treeConfig
        );

    GET_CUDA_ERROR("KernelError");
    cudaDeviceSynchronize();
    GET_CUDA_ERROR("SyncError");
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "Kernel() duration: " <<
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()) << std::endl;

    QuadTree* tree2 = this->tree;
    int totalCount = 0;

    for (int depth = 0; depth < treeConfig.maxDepth; ++depth)
    {
        const auto leafs = getNumNodesInCurrentDepth<2>(depth);
        for (int leaf = 0; leaf < leafs; ++leaf)
        {
            const QuadTree* subTree = &tree2[leaf];

            if ((subTree->particlesCountInNode() < treeConfig.minPointsPerNode ||
                depth == treeConfig.maxDepth - 1) && subTree->particlesCountInNode() > 0)
            {
                totalCount += subTree->particlesCountInNode();
            }
        }

        tree2 += leafs;
    }

    std::cout << "total points: " << totalCount << " / " << size << "\n";

    if (totalCount != size)
    {
        throw "Invalid tree: totalCount != size\n";
    }
}

template<typename ParticleType>
void QuadTreeBuilder<ParticleType>::reset()
{
    std::cout << "Reset()" << std::endl;
    const int maxNodes = treeConfig.GetNodesCount();

    for (int i = 0; i < maxNodes; ++i)
    {
        this->tree[i].id = 0;
        this->tree[i].bounds.min = {0.0f, 0.0f, 0.0f};
        this->tree[i].bounds.max = {0.0f, 0.0f, 0.0f};
        this->tree[i].startId = 0;
        this->tree[i].endId = 0;
    }
}

template<typename ParticleType>
QuadTreeBuilder<ParticleType>::~QuadTreeBuilder()
{
    cudaFree(pointsExch);
    cudaFree(this->tree);
}