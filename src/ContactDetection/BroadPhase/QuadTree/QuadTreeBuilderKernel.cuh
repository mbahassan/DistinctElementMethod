//
// Created by iqraa on 28-2-25.
//
#ifndef QUADTREEBUILDERKERNEL_CUH
#define QUADTREEBUILDERKERNEL_CUH

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cooperative_groups.h>
// namespace cg = cooperative_groups;

/*
template<int NUM_THREADS_PER_BLOCK, class ParticleType>
__global__ void QuadTreeKernel(QuadTree *nodes, ParticleType *points, ParticleType *pointsExch, TreeConfig params)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // The number of warps in a block.
    const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warpSize;

    // Shared memory to store the number of points.
    extern __shared__ int sharedMem[];

    // s_num_pts[4][NUM_WARPS_PER_BLOCK];
    // Addresses of shared memory.
    volatile int *s_num_pts[4];

    for (int i = 0; i < 4; ++i)
        s_num_pts[i] = static_cast<volatile int *>(&sharedMem[i * NUM_WARPS_PER_BLOCK]);

    // Compute the coordinates of the threads in the block.
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    // Mask for compaction.
    // Same as: asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt) );
    int lane_mask_lt = (1 << lane_id) - 1;

    // The current node.
    QuadTree &node = nodes[blockIdx.x];

    // The number of points in the node.
    int num_points = node.endId - node.startId;


    int range_begin, range_end;
    int warp_cnts[4] = {0, 0, 0, 0};
    //
    // 1- Check the number of points and its depth.
    //

    // Stop the recursion here. Make sure points[0] contains all the points.
    if (params.depth >= params.maxDepth || num_points <= params.minPointsPerNode) {
        if (params.depth > 0 && params.depth % 2 == 0) {
            int it = node.startId, end = node.endId;

            for (it += threadIdx.x; it < end; it += NUM_THREADS_PER_BLOCK)
                if (it < end) points[it] = pointsExch[it];
        }
        return;
    }

    // Compute the center of the bounding box of the points.
    const auto &bbox = node.bounds;

    float3 center = bbox.getCenter();

    // Find how many points to give to each warp.
    int num_points_per_warp = max(warpSize, (num_points + NUM_WARPS_PER_BLOCK - 1) / NUM_WARPS_PER_BLOCK);

    // Each warp of threads will compute the number of points to move to each
    // quadrant.
    range_begin = node.startId + warp_id * num_points_per_warp;
    range_end = min(range_begin + num_points_per_warp, node.endId);

    //
    // 2- Count the number of points in each child.
    //

    // Input points.
    // const Points &in_points = points[params.point_selector];

    // Get input and output buffers based on depth
    // Even depths: read from points, write to pointsExch
    // Odd depths: read from pointsExch, write to points
    ParticleType *inputPoints = (params.depth % 2 == 0) ? points : pointsExch;
    ParticleType *outputPoints = (params.depth % 2 == 0) ? pointsExch : points;

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    // Compute the number of points.
    for (int range_it = range_begin + tile32.thread_rank();
         tile32.any(range_it < range_end); range_it += warpSize) {
        // Is it still an active thread?
        bool is_active = range_it < range_end;

        auto bbCenter = (inputPoints[range_it].boundingBox.min + inputPoints[range_it].boundingBox.max) /2.0f;

        // Load the coordinates of the point.
        float3 p = is_active ? bbCenter : make_float3(0.0f, 0.0f,0.0f);

        // Count top-left points.
        int num_pts =
                __popc(tile32.ballot(is_active && p.x < center.x && p.y >= center.y));
        warp_cnts[0] += tile32.shfl(num_pts, 0);

        // Count top-right points.
        num_pts =
                __popc(tile32.ballot(is_active && p.x >= center.x && p.y >= center.y));
        warp_cnts[1] += tile32.shfl(num_pts, 0);

        // Count bottom-left points.
        num_pts =
                __popc(tile32.ballot(is_active && p.x < center.x && p.y < center.y));
        warp_cnts[2] += tile32.shfl(num_pts, 0);

        // Count bottom-right points.
        num_pts =
                __popc(tile32.ballot(is_active && p.x >= center.x && p.y < center.y));
        warp_cnts[3] += tile32.shfl(num_pts, 0);
    }

    if (tile32.thread_rank() == 0) {
        s_num_pts[0][warp_id] = warp_cnts[0];
        s_num_pts[1][warp_id] = warp_cnts[1];
        s_num_pts[2][warp_id] = warp_cnts[2];
        s_num_pts[3][warp_id] = warp_cnts[3];
    }

    // Make sure warps have finished counting.
    cg::sync(cta);

    //
    // 3- Scan the warps' results to know the "global" numbers.
    //

    // First 4 warps scan the numbers of points per child (inclusive scan).
    if (warp_id < 4) {
        int num_pts = tile32.thread_rank() < NUM_WARPS_PER_BLOCK
                          ? s_num_pts[warp_id][tile32.thread_rank()]
                          : 0;
#pragma unroll

        for (int offset = 1; offset < NUM_WARPS_PER_BLOCK; offset *= 2) {
            int n = tile32.shfl_up(num_pts, offset);

            if (tile32.thread_rank() >= offset) num_pts += n;
        }

        if (tile32.thread_rank() < NUM_WARPS_PER_BLOCK)
            s_num_pts[warp_id][tile32.thread_rank()] = num_pts;
    }

    cg::sync(cta);

    // Compute global offsets.
    if (warp_id == 0) {
        int sum = s_num_pts[0][NUM_WARPS_PER_BLOCK - 1];

        for (int row = 1; row < 4; ++row) {
            int tmp = s_num_pts[row][NUM_WARPS_PER_BLOCK - 1];
            cg::sync(tile32);

            if (tile32.thread_rank() < NUM_WARPS_PER_BLOCK)
                s_num_pts[row][tile32.thread_rank()] = s_num_pts[row][tile32.thread_rank()] + sum;

            cg::sync(tile32);
            sum += tmp;
        }
    }
    cg::sync(cta);

    // Make the scan exclusive.
    int val = 0;
    if (threadIdx.x < 4 * NUM_WARPS_PER_BLOCK) {
        val = threadIdx.x == 0 ? 0 : sharedMem[threadIdx.x - 1];
        val += node.startId;
    }
    cg::sync(cta);

    if (threadIdx.x < 4 * NUM_WARPS_PER_BLOCK) {
        sharedMem[threadIdx.x] = val;
    }

    cg::sync(cta);

    //
    // 4- Move points.
    //
    if (!(params.depth >= params.maxDepth || num_points <= params.minPointsPerNode)) {
        // Output points.
        // Points &out_points = points[(params.point_selector + 1) % 2];

        warp_cnts[0] = s_num_pts[0][warp_id];
        warp_cnts[1] = s_num_pts[1][warp_id];
        warp_cnts[2] = s_num_pts[2][warp_id];
        warp_cnts[3] = s_num_pts[3][warp_id];

        // const Points &in_points = points[params.point_selector];

        // Reorder points.
        for (int range_it = range_begin + tile32.thread_rank();
             tile32.any(range_it < range_end); range_it += warpSize) {
            // Is it still an active thread?
            bool is_active = range_it < range_end;

            // using BoundingBox of the particle instead of the center
            auto bbCenter = (inputPoints[range_it].boundingBox.min + inputPoints[range_it].boundingBox.max) /2.0f;

            // Load the coordinates of the point.
            float3 p = is_active ? bbCenter : make_float3(0.0f, 0.0f,0.0f);

            // Get the full particle data
            // Particle particle = is_active ? inputPoints[range_it] : Particle();


            // Count top-left points.
            bool pred = is_active && p.x < center.x && p.y >= center.y;
            int vote = tile32.ballot(pred);
            int dest = warp_cnts[0] + __popc(vote & lane_mask_lt);

            // if (pred) out_points.set_point(dest, p);
            if (pred) outputPoints[dest].position = p;

            warp_cnts[0] += tile32.shfl(__popc(vote), 0);

            // Count top-right points.
            pred = is_active && p.x >= center.x && p.y >= center.y;
            vote = tile32.ballot(pred);
            dest = warp_cnts[1] + __popc(vote & lane_mask_lt);

            // if (pred) out_points.set_point(dest, p);
            if (pred) outputPoints[dest].position = p;

            warp_cnts[1] += tile32.shfl(__popc(vote), 0);

            // Count bottom-left points.
            pred = is_active && p.x < center.x && p.y < center.y;
            vote = tile32.ballot(pred);
            dest = warp_cnts[2] + __popc(vote & lane_mask_lt);

            // if (pred) out_points.set_point(dest, p);
            if (pred) outputPoints[dest].position = p;

            warp_cnts[2] += tile32.shfl(__popc(vote), 0);

            // Count bottom-right points.
            pred = is_active && p.x >= center.x && p.y < center.y;
            vote = tile32.ballot(pred);
            dest = warp_cnts[3] + __popc(vote & lane_mask_lt);

            // if (pred) out_points.set_point(dest, p);
            if (pred) outputPoints[dest].position = p;

            warp_cnts[3] += tile32.shfl(__popc(vote), 0);
        }
    }
    cg::sync(cta);

    if (tile32.thread_rank() == 0) {
        s_num_pts[0][warp_id] = warp_cnts[0];
        s_num_pts[1][warp_id] = warp_cnts[1];
        s_num_pts[2][warp_id] = warp_cnts[2];
        s_num_pts[3][warp_id] = warp_cnts[3];
    }

    cg::sync(cta);

    //
    // 5- Launch new blocks.
    //
    if (!(params.depth >= params.maxDepth ||
          num_points <= params.minPointsPerNode)) {
        // The last thread launches new blocks.
        if (threadIdx.x == NUM_THREADS_PER_BLOCK - 1) {
            // The children.
            QuadTree *children = &nodes[params.numNodesAtThisLevel - (node.id & ~3)];

            // The offsets of the children at their level.
            int child_offset = 4 * node.id;

            // Set IDs.
            children[child_offset + 0].id = (4 * node.id + 0);
            children[child_offset + 1].id = (4 * node.id + 1);
            children[child_offset + 2].id = (4 * node.id + 2);
            children[child_offset + 3].id = (4 * node.id + 3);

            const auto &bbox = node.bounds;
            // Points of the bounding-box.
            const float2 &p_min = {bbox.min.x, bbox.min.y};
            const float2 &p_max = {bbox.max.x, bbox.max.y};

            // Set the bounding boxes of the children.
            children[child_offset + 0].bounds.min = {p_min.x, center.y}; // Top-left.
            children[child_offset + 0].bounds.max = {center.x, p_max.y}; // Top-left.

            children[child_offset + 1].bounds.min = {center.x, center.y}; // Top-right.
            children[child_offset + 1].bounds.max = {p_max.x, p_max.y}; // Top-right.

            children[child_offset + 2].bounds.min = {p_min.x, p_min.y}; // Bottom-left.
            children[child_offset + 2].bounds.max = {center.x, center.y}; // Bottom-left.

            children[child_offset + 3].bounds.min = {center.x, p_min.y}; // Bottom-right.
            children[child_offset + 3].bounds.max = {p_max.x, center.y}; // Bottom-right.

            // Set the ranges of the children.
            children[child_offset + 0].startId = node.startId;
            children[child_offset + 0].endId = s_num_pts[0][warp_id];

            children[child_offset + 1].startId = s_num_pts[0][warp_id];
            children[child_offset + 1].endId = s_num_pts[1][warp_id];

            children[child_offset + 2].startId = s_num_pts[1][warp_id];
            children[child_offset + 2].endId = s_num_pts[2][warp_id];

            children[child_offset + 3].startId = s_num_pts[2][warp_id];
            children[child_offset + 3].endId = s_num_pts[3][warp_id];

            // Launch 4 children.
            QuadTreeKernel<NUM_THREADS_PER_BLOCK, ParticleType><<<
                    4, NUM_THREADS_PER_BLOCK, 4 * NUM_WARPS_PER_BLOCK * sizeof(int)>>>(
                        &children[child_offset], pointsExch, points, TreeConfig(params, true));
        }
    }
}
*/


#include "ContactDetection/BroadPhase/QuadTree/QuadTree.h"
#include "ContactDetection/BroadPhase/Config/TreeConfig.h"

namespace cg = cooperative_groups;

// will divide at least 1 time
template<class ParticleType>
__global__ void QuadTreeKernel(
    QuadTree* tree, ParticleType* points, ParticleType* pointsExch, int depth, int maxDepth, int minPointsToDivide)
{
    // threads within a warp
    auto thisWarp = cg::coalesced_threads();
    auto thisBlock = cg::this_thread_block();

    const int warpsPerBlock = thisBlock.size() / warpSize;
    const int warpId = thisBlock.thread_rank() / warpSize;
    const int laneId = thisWarp.thread_rank() % warpSize;
    // to get a 'global' warp id
    // unsigned warpId2;
    // asm volatile("mov.u32 %0, %warpid;" : "=r"(warpId2));

    QuadTree& subTree = tree[blockIdx.x];
    const auto aabb = subTree.bounds;
    const auto center = aabb.getCenter();

    const int pointsCount = subTree.endId - subTree.startId;
    const int pointsPerWarp = (pointsCount + warpsPerBlock - 1) / warpsPerBlock;
    const int startId = subTree.startId + warpId * pointsPerWarp;
    const int endId = min(startId + pointsPerWarp, subTree.endId);
    thisBlock.sync();

    if (pointsCount < minPointsToDivide || depth == maxDepth - 1)
    {
        // to be sure that the first array contains all the changes
        // here pointsExch contains the latest chages, because
        // in the prev step Kernel got pointsExch as a first argument
        // **check a QuadTreeKernel call below
        if (depth > 0 && depth % 2 == 0)
        {
            int start = subTree.startId;
            for (start += threadIdx.x; start < subTree.endId; start += thisBlock.size())
            {
                points[start] = pointsExch[start];
            }
        }

        return;
    }

    extern __shared__ int pointsInCell[];

    if (threadIdx.x < warpsPerBlock * 4)
        pointsInCell[threadIdx.x] = 0;

    int pointsInCellLocal[4] = {0, 0, 0, 0};

    thisBlock.sync();


    // each warp fills 4 subdivisions
    for (int i = startId + thisWarp.thread_rank();
             thisWarp.any(i < endId);
             i += thisWarp.size())
    {
        const auto isInRange = i < endId;

        const auto point = isInRange ? pointsExch[i].position : float3{};

        const auto isUpLeft = isInRange && point.x <= center.x && point.y > center.y;
        // auto summ = __popc(thisWarp.ballot(isUpLeft));
        pointsInCellLocal[0] += __popc(thisWarp.ballot(isUpLeft));//thisWarp.shfl(summ, 0);

        const auto isUpRight = isInRange && point.x > center.x && point.y > center.y;
        // summ = __popc(thisWarp.ballot(isUpRight));
        pointsInCellLocal[1] += __popc(thisWarp.ballot(isUpRight));//thisWarp.shfl(summ, 0);

        const auto isDownLeft = isInRange && point.x <= center.x && point.y <= center.y;
        // summ = __popc(thisWarp.ballot(isDownLeft));
        pointsInCellLocal[2] += __popc(thisWarp.ballot(isDownLeft));//thisWarp.shfl(summ, 0);

        const auto isDownRight = isInRange && point.x > center.x && point.y <= center.y;
        // summ = __popc(thisWarp.ballot(isDownRight));
        pointsInCellLocal[3] += __popc(thisWarp.ballot(isDownRight));//thisWarp.shfl(summ, 0);

    }
    thisBlock.sync();

    // counts in each cell of each layer
    if (thisWarp.thread_rank() == 0)
    {
        pointsInCell[warpId * 4 + 0] = pointsInCellLocal[0];
        pointsInCell[warpId * 4 + 1] = pointsInCellLocal[1];
        pointsInCell[warpId * 4 + 2] = pointsInCellLocal[2];
        pointsInCell[warpId * 4 + 3] = pointsInCellLocal[3];
    }
    thisBlock.sync();

    // Store counts for each leaf in the latest index of a Shared Memory
    // one need atleat 4 warp to perform this action (fill each subdivision, i.e. leaf)
    if (warpId < 4)
    {
        // warpId - a cell number
        // thread_rank = a warp number (but max = 32!)
        int totalPointsCountPerCell = thisWarp.thread_rank() < warpsPerBlock
                        ? pointsInCell[thisWarp.thread_rank() * 4 + warpId]
                        : 0;

        // ccalc offset for each cell and for each warp
        for (int offset = 1; offset < warpsPerBlock; offset *= 2)
        {
            int n = thisWarp.shfl_up(totalPointsCountPerCell, offset);

            if (thisWarp.thread_rank() >= offset)
                totalPointsCountPerCell += n;
        }

        thisWarp.sync();

        if (thisWarp.thread_rank() < warpsPerBlock)
        {
            pointsInCell[thisWarp.thread_rank() * 4 + warpId] = totalPointsCountPerCell;
        }
    }
    thisBlock.sync();

    // Calc endIds
    if (warpId == 0)
    {
        int itemsInCell = pointsInCell[(warpsPerBlock - 1) * 4 + 0];
        thisWarp.sync();

        for (int i = 1; i < 4; ++i)
        {
            int itemsCount = pointsInCell[(warpsPerBlock - 1) * 4 + i];
            thisWarp.sync();

            if (thisWarp.thread_rank() < warpsPerBlock)
            {
                pointsInCell[thisWarp.thread_rank() * 4 + i] += itemsInCell;
            }

            thisWarp.sync();

            itemsInCell += itemsCount;
        }
    }
    thisBlock.sync();

    int changeValue = 0;
    int changePointsId = 0;

    if (thisWarp.thread_rank() < warpsPerBlock)
    {
        // current global cell id
        changePointsId = thisWarp.thread_rank() * 4 + warpId;

        if (changePointsId != 0)
        {
            const int iddPrev = (changePointsId < 4) ? (warpsPerBlock - 1) * 4 + (changePointsId - 1) : (changePointsId - 4);

            changeValue = pointsInCell[iddPrev];
        }
        changeValue += subTree.startId;
    }

    thisBlock.sync();
    if (thisWarp.thread_rank() < warpsPerBlock)
    {
        pointsInCell[changePointsId] = changeValue;
    }
    thisBlock.sync();

    // 0-3 cell ids within a warp. i.e.: tl, tr, bl, br
    int offset1 = pointsInCell[warpId * 4 + 0];
    int offset2 = pointsInCell[warpId * 4 + 1];
    int offset3 = pointsInCell[warpId * 4 + 2];
    int offset4 = pointsInCell[warpId * 4 + 3];

    const int lane_mask_lt = (1 << laneId) - 1;
    thisBlock.sync();

    for (int i = startId + thisWarp.thread_rank();
             thisWarp.any(i < endId);
             i += thisWarp.size())
    {
        const auto isInRange = i < endId;

        const auto point = isInRange ? pointsExch[i].position : float3{};

        const auto isUpLeft = isInRange && point.x <= center.x && point.y > center.y;
        const auto mask1 = thisWarp.ballot(isUpLeft);
        const auto destId1 = offset1 + __popc(mask1 & lane_mask_lt);
        if (isUpLeft)
        {
            points[destId1] = pointsExch[i];
        }
        offset1 += thisWarp.shfl(__popc(mask1), 0);

        const auto isUpRight = isInRange && point.x > center.x && point.y > center.y;
        const auto mask2 = thisWarp.ballot(isUpRight);
        const auto destId2 = offset2 + __popc(mask2 & lane_mask_lt);
        if (isUpRight)
        {
            points[destId2] = pointsExch[i];
        }
        offset2 += thisWarp.shfl(__popc(mask2), 0);

        const auto isDownLeft = isInRange && point.x <= center.x && point.y <= center.y;
        const auto mask3 = thisWarp.ballot(isDownLeft);
        const auto destId3 = offset3 + __popc(mask3 & lane_mask_lt);
        if (isDownLeft)
        {
            points[destId3] = pointsExch[i];
        }
        offset3 += thisWarp.shfl(__popc(mask3), 0);

        const auto isDownRight = isInRange && point.x > center.x && point.y <= center.y;
        const auto mask4 = thisWarp.ballot(isDownRight);
        const auto destId4 = offset4 + __popc(mask4 & lane_mask_lt);
        if (isDownRight)
        {
            points[destId4] = pointsExch[i];
        }
        offset4 += thisWarp.shfl(__popc(mask4), 0);
    }
    thisBlock.sync();

    if (thisBlock.thread_rank() == thisBlock.size() - 1)
    {
        const auto count = getNumNodesInCurrentDepth<2>(depth) - (subTree.id & ~3);
        QuadTree* child = &tree[count];

        const auto treeIdNext = 4 * subTree.id;

        child[treeIdNext + 0].id = treeIdNext + 0;
        child[treeIdNext + 0].bounds.min = {subTree.bounds.min.x, center.y};
        child[treeIdNext + 0].bounds.max = {center.x, subTree.bounds.max.y};
        child[treeIdNext + 0].startId = subTree.startId;
        child[treeIdNext + 0].endId = offset1;

        child[treeIdNext + 1].id = treeIdNext + 1;
        child[treeIdNext + 1].bounds.min = center;
        child[treeIdNext + 1].bounds.max = subTree.bounds.max;
        child[treeIdNext + 1].startId = offset1;
        child[treeIdNext + 1].endId = offset2;

        child[treeIdNext + 2].id = treeIdNext + 2;
        child[treeIdNext + 2].bounds.min = subTree.bounds.min;
        child[treeIdNext + 2].bounds.max = center;
        child[treeIdNext + 2].startId = offset2;
        child[treeIdNext + 2].endId = offset3;

        child[treeIdNext + 3].id = treeIdNext + 3;
        child[treeIdNext + 3].bounds.min = {center.x, subTree.bounds.min.y};
        child[treeIdNext + 3].bounds.max = {subTree.bounds.max.x, center.y};
        child[treeIdNext + 3].startId = offset3;
        child[treeIdNext + 3].endId = offset4;

        QuadTreeKernel<<<4, thisBlock.size(), warpsPerBlock * 4 * sizeof(int)>>>(&child[treeIdNext],pointsExch,
            points,
            depth + 1, maxDepth, minPointsToDivide);
    }
}

#endif // QUADTREEBUILDERKERNEL_CUH
