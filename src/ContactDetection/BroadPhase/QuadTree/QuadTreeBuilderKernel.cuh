//
// Created by iqraa on 28-2-25.
//

#ifndef QUADTREEBUILDERKERNEL_CUH
#define QUADTREEBUILDERKERNEL_CUH

////////////////////////////////////////////////////////////////////////////////
// Build a quadtree on the GPU. Use CUDA Dynamic Parallelism.
//
// The algorithm works as follows. The host (CPU) launches one block of
// NUM_THREADS_PER_BLOCK threads. That block will do the following steps:
//
// 1- Check the number of points and its depth.
//
// We impose a maximum depth to the tree and a minimum number of points per
// node. If the maximum depth is exceeded or the minimum number of points is
// reached. The threads in the block exit.
//
// Before exiting, they perform a buffer swap if it is needed. Indeed, the
// algorithm uses two buffers to permute the points and make sure they are
// properly distributed in the quadtree. By design we want all points to be
// in the first buffer of points at the end of the algorithm. It is the reason
// why we may have to swap the buffer before leavin (if the points are in the
// 2nd buffer).
//
// 2- Count the number of points in each child.
//
// If the depth is not too high and the number of points is sufficient, the
// block has to dispatch the points into four geometrical buckets: Its
// children. For that purpose, we compute the center of the bounding box and
// count the number of points in each quadrant.
//
// The set of points is divided into sections. Each section is given to a
// warp of threads (32 threads). Warps use __ballot and __popc intrinsics
// to count the points. See the Programming Guide for more information about
// those functions.
//
// 3- Scan the warps' results to know the "global" numbers.
//
// Warps work independently from each other. At the end, each warp knows the
// number of points in its section. To know the numbers for the block, the
// block has to run a scan/reduce at the block level. It's a traditional
// approach. The implementation in that sample is not as optimized as what
// could be found in fast radix sorts, for example, but it relies on the same
// idea.
//
// 4- Move points.
//
// Now that the block knows how many points go in each of its 4 children, it
// remains to dispatch the points. It is straightforward.
//
// 5- Launch new blocks.
//
// The block launches four new blocks: One per children. Each of the four blocks
// will apply the same algorithm.

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


__global__ void QuadTreeKernel(Particle* points, Particle* pointsExch,
    QuadTree* tree,int depth, TreeConfig configTree) {

    // Handle to thread block group
    auto thisWarp = cg::coalesced_threads();
    auto thisBlock = cg::this_thread_block();
    // auto thisWarp = cg::tiled_partition<32>(thisBlock);

    // Compute the coordinates of the threads in the block.
    const int warpsPerBlock = configTree.threadsPerBlock / warpSize;
    const int warpId = threadIdx.x / warpSize;
    const int laneId = threadIdx.x % warpSize;

    // Mask for compaction.
    int lane_mask_lt = (1 << laneId) - 1;


    // The current node.
    QuadTree& node = tree[blockIdx.x];
    const auto aabb = node.bounds;
    const auto center = aabb.getCenter();

    // The number of points in the node.
    const int numPoints = node.endId - node.startId;
    const int pointsPerWarp = max(warpSize, (numPoints + warpsPerBlock - 1) / warpsPerBlock);
    const int startId = node.startId + warpId * pointsPerWarp;
    const int endId = min(startId + pointsPerWarp, node.endId);
    thisBlock.sync();

    //
    // 1- Check the number of points and its depth.
    //

    // Stop the recursion here. Make sure points[0] contains all the points.
    if (numPoints <= configTree.minPointsToDivide || depth >= configTree.maxDepth) {
        if (depth % 2 == 1) {
            int start = node.startId;

            for (start += threadIdx.x; start < node.endId; start += configTree.threadsPerBlock)
            {
                if (start < node.endId)
                    points[start] = pointsExch[start];
            }
        }
        return;
    }


    // Shared memory to store the number of points
    extern __shared__ int pointsInCell[];

    if (thisBlock.thread_rank() < warpsPerBlock * 4)
        pointsInCell[thisBlock.thread_rank()] = 0;

    int pointsInCellLocal[4] = {0, 0, 0, 0};
    thisBlock.sync();

    //
    // 2- Count the number of points in each child.
    //

    // Compute the number of points.
    for (int i = startId + thisWarp.thread_rank(); thisWarp.any(i < endId); i += warpSize) {
        // Is it still an active thread?
        bool isInRange = i < endId;

        // Load the coordinates of the point.
        auto p = isInRange ? pointsExch[i].position : make_float3(0.0f, 0.0f, 0.0f);

        // Count top-left points.
        const auto isTopLeft = isInRange && p.x < center.x && p.y >= center.y;
        pointsInCellLocal[0] += __popc(thisWarp.ballot(isTopLeft));

        // Count top-right points.
        const auto isTopRight = isInRange && p.x >= center.x && p.y >= center.y;
        pointsInCellLocal[1] += __popc(thisWarp.ballot(isTopRight));

        // Count bottom-left points.
        const auto isBottomLeft = isInRange && p.x < center.x && p.y < center.y;
        pointsInCellLocal[2] += __popc(thisWarp.ballot(isBottomLeft));

        // Count bottom-right points.
        const auto isBottomRight = isInRange && p.x >= center.x && p.y < center.y;
        pointsInCellLocal[3] += __popc(thisWarp.ballot(isBottomRight));
    }

    if (thisWarp.thread_rank() == 0) {
        pointsInCell[warpId * 4 + 0] = pointsInCellLocal[0];
        pointsInCell[warpId * 4 + 1] = pointsInCellLocal[1];
        pointsInCell[warpId * 4 + 2] = pointsInCellLocal[2];
        pointsInCell[warpId * 4 + 3] = pointsInCellLocal[3];
    }
    // Make sure warps have finished counting.
    thisBlock.sync();

    //
    // 3- Scan the warps' results to know the "global" numbers.
    //

    // First 4 warps scan the numbers of points per child (inclusive scan).
    if (warpId < 4) {
        int totalPointsCountPerCell = thisWarp.thread_rank() < warpsPerBlock
                          ? pointsInCell[thisWarp.thread_rank() * 4 + warpId]
                          : 0;
#pragma unroll
        for (int offset = 1; offset < warpsPerBlock; offset *= 2) {
            int n = thisWarp.shfl_up(totalPointsCountPerCell, offset);

            if (thisWarp.thread_rank() >= offset) totalPointsCountPerCell += n;
        }

        if (thisWarp.thread_rank() < warpsPerBlock)
            pointsInCell[thisWarp.thread_rank() * 4 + warpId] = totalPointsCountPerCell;
    }
    thisBlock.sync();

    // Compute global offsets.
    if (warpId == 0)
    {
        int itemsInCell = pointsInCell[(warpsPerBlock - 1) * 4 + 0];

        for (int row = 1; row < 4; ++row)
        {
            int itemsCount = pointsInCell[(warpsPerBlock - 1) * 4 + row];
            cg::sync(thisWarp);

            if (thisWarp.thread_rank() < warpsPerBlock)
                pointsInCell[thisWarp.thread_rank() * 4 + row] += itemsInCell;

            cg::sync(thisWarp);
            itemsInCell += itemsCount;
        }
    }
    cg::sync(thisBlock);

    // Make the scan exclusive
    int changeValue = 0;
    int changePointsId = 0;

    if (thisWarp.thread_rank() < warpsPerBlock)
    {
        changePointsId = thisWarp.thread_rank() * 4 + warpId;

        if (changePointsId != 0)
        {
            const int idPrev = (changePointsId < 4) ?
                (warpsPerBlock - 1) * 4 + (changePointsId - 1) :
                (changePointsId - 4);
            changeValue = pointsInCell[idPrev];
        }
        changeValue += node.startId;
    }
    thisBlock.sync();

    if (thisWarp.thread_rank() < warpsPerBlock)
    {
        pointsInCell[changePointsId] = changeValue;
    }
    thisBlock.sync();

    // Get offsets for each cell
    int offset1 = pointsInCell[warpId * 4 + 0];
    int offset2 = pointsInCell[warpId * 4 + 1];
    int offset3 = pointsInCell[warpId * 4 + 2];
    int offset4 = pointsInCell[warpId * 4 + 3];

    thisBlock.sync();

    //
    // 4- Move points.
    //

    // Reorder points.
    for (int i = startId + thisWarp.thread_rank(); thisWarp.any(i < endId); i += warpSize)
    {
        // Is it still an active thread?
        bool isInRange = i < endId;

        // Load the coordinates of the point
        const auto p = isInRange ? pointsExch[i].position : float3{};

        /// Count top-left points.
        bool isTopLeft = isInRange && p.x < center.x && p.y >= center.y;
        const auto mask1 = thisWarp.ballot(isTopLeft);
        const auto destId1 = offset1 + __popc(mask1 & lane_mask_lt);

        if (isTopLeft)
            points[destId1].position = p;

        offset1 += thisWarp.shfl(__popc(mask1), 0);

        /// Process top-right points
        const auto isTopRight = isInRange && p.x >= center.x && p.y >= center.y;
        const auto mask2 = thisWarp.ballot(isTopRight);
        const auto destId2 = offset2 + __popc(mask2 & lane_mask_lt);

        if (isTopRight)
            points[destId2].position = p;

        offset2 += thisWarp.shfl(__popc(mask2), 0);

        /// Process bottom-left points
        const auto isBottomLeft = isInRange && p.x < center.x && p.y < center.y;
        const auto mask3 = thisWarp.ballot(isBottomLeft);
        const auto destId3 = offset3 + __popc(mask3 & lane_mask_lt);

        if (isBottomLeft)
            points[destId3].position = p;

        offset3 += thisWarp.shfl(__popc(mask3), 0);

        /// Process bottom-right points
        const auto isBottomRight = isInRange && p.x >= center.x && p.y < center.y;
        const auto mask4 = thisWarp.ballot(isBottomRight);
        const auto destId4 = offset4 + __popc(mask4 & lane_mask_lt);

        if (isBottomRight)
            points[destId4].position = p;

        offset4 += thisWarp.shfl(__popc(mask4), 0);
    }
    cg::sync(thisBlock);



    //
    // 5- Launch new blocks.
    //
    if ((depth < configTree.maxDepth && numPoints > configTree.minPointsToDivide))
    {
        // The last thread launches new blocks.
        if (threadIdx.x == configTree.threadsPerBlock - 1)
        {
            auto childOffset = getNumNodesInCurrentDepth<2>(depth) - (node.id & ~3);
            // The children.
            QuadTree* children = &tree[childOffset];

            // The offsets of the children at their level.
            int treeIdNext = 4 * node.id;

            // Set IDs and bounds for children
            children[treeIdNext + 0].id = treeIdNext + 0;
            children[treeIdNext + 0].bounds.min = {aabb.min.x, center.y};
            children[treeIdNext + 0].bounds.max = {center.x, aabb.max.y};
            children[treeIdNext + 0].startId = node.startId;
            children[treeIdNext + 0].endId = offset1;

            children[treeIdNext + 1].id = treeIdNext + 1;
            children[treeIdNext + 1].bounds.min = center;
            children[treeIdNext + 1].bounds.max = aabb.max;
            children[treeIdNext + 1].startId = offset1;
            children[treeIdNext + 1].endId = offset2;

            children[treeIdNext + 2].id = treeIdNext + 2;
            children[treeIdNext + 2].bounds.min = aabb.min;
            children[treeIdNext + 2].bounds.max = center;
            children[treeIdNext + 2].startId = offset2;
            children[treeIdNext + 2].endId = offset3;

            children[treeIdNext + 3].id = treeIdNext + 3;
            children[treeIdNext + 3].bounds.min = {center.x, aabb.min.y};
            children[treeIdNext + 3].bounds.max = {aabb.max.x, center.y};
            children[treeIdNext + 3].startId = offset3;
            children[treeIdNext + 3].endId = offset4;

            // Launch 4 children.
            QuadTreeKernel<<<4, thisBlock.size(), warpsPerBlock * 4 * sizeof(int)>>>(
            pointsExch, points, &children[treeIdNext],
            depth + 1, configTree);
        }
    }
}


#endif //QUADTREEBUILDERKERNEL_CUH
