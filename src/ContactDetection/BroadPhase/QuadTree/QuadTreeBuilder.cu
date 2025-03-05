//
// Created by iqraa on 28-2-25.
//



#include <iostream>
#include <chrono>

#include "QuadTreeBuilder.cuh"
#include "QuadTreeBuilderKernel.cuh"
#include "Tools/CudaHelper.hpp"
#include "Tools/ArthmiticOperator/MathOperators.hpp"


QuadTreeBuilder::QuadTreeBuilder(const TreeConfig& treeConfig)
    : treeConfig {treeConfig}
{
}

void QuadTreeBuilder::initialize(const int capacity)
{
    cudaMalloc((void**)&pointsExch, capacity * sizeof(Particle));
    GET_CUDA_ERROR("cudaMalloc() pointsExch");
    // tree:
    const int maxNodes = treeConfig.GetNodesCount();
    cudaMallocManaged((void**)&tree, maxNodes * sizeof(QuadTree));
    GET_CUDA_ERROR("cudaMallocManaged() tree");
    cudaDeviceSynchronize();
    GET_CUDA_ERROR("cudaDeviceSynchronize");
}

void QuadTreeBuilder::build(Particle* points, const int size)
{
    reset();

    tree->id = 0;
    tree->bounds.min = treeConfig.origin;
    const auto maxDim = treeConfig.origin + treeConfig.size;
    tree->bounds.max = maxDim;
    tree->startId = 0;
    tree->endId = size;

    std::cout << "Build()" << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    const int warpsPerBlock = treeConfig.threadsPerBlock / 32;

    cudaMemcpy(pointsExch, points, size * sizeof(Particle), cudaMemcpyDeviceToDevice);

    QuadTreeKernel<<<1, treeConfig.threadsPerBlock, warpsPerBlock * 4 * sizeof(int)>>>
    (
        points,
        pointsExch,
        tree,
        0,
        treeConfig
        );

    GET_CUDA_ERROR("KernelError");
    cudaDeviceSynchronize();
    GET_CUDA_ERROR("SyncError");
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "Kernel() duration: " <<
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()) << std::endl;

    QuadTree* tree2 = tree;
    int totalCount = 0;

    for (int depth = 0; depth < treeConfig.maxDepth; ++depth)
    {
        const auto leafs = getNumNodesInCurrentDepth<2>(depth);
        for (int leaf = 0; leaf < leafs; ++leaf)
        {
            const QuadTree* subTree = &tree2[leaf];

            if ((subTree->maxParticlesPerNode() < treeConfig.minPointsToDivide ||
                depth == treeConfig.maxDepth - 1) && subTree->maxParticlesPerNode() > 0)
            {
                totalCount += subTree->maxParticlesPerNode();
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

void QuadTreeBuilder::reset()
{
    std::cout << "Reset()" << std::endl;
    const int maxNodes = treeConfig.GetNodesCount();

    for (int i = 0; i < maxNodes; ++i)
    {
        tree[i].id = 0;
        tree[i].bounds.min = {0.0f, 0.0f};
        tree[i].bounds.max = {0.0f, 0.0f};
        tree[i].startId = 0;
        tree[i].endId = 0;
    }
}

QuadTreeBuilder::~QuadTreeBuilder()
{
    cudaFree(pointsExch);
    cudaFree(tree);
}