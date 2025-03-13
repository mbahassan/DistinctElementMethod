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

/*__global__ void QuadTreeKernel(Particle* points, Particle* pointsExch,
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
        node.isLeaf = true;
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

    // if (thisWarp.thread_rank() == 0) {
    //     pointsInCell[0 * warpsPerBlock + warpId] = pointsInCellLocal[0];
    //     pointsInCell[1 * warpsPerBlock + warpId] = pointsInCellLocal[1];
    //     pointsInCell[2 * warpsPerBlock + warpId] = pointsInCellLocal[2];
    //     pointsInCell[3 * warpsPerBlock + warpId] = pointsInCellLocal[3];
    // }
    // Make sure warps have finished counting.
    thisBlock.sync();

    //
    // 3- Scan the warps' results to know the "global" numbers.
    //

    // First 4 warps scan the numbers of points per child (inclusive scan).
//     if (warpId < 4) {
//         int totalPointsCountPerCell = thisWarp.thread_rank() < warpsPerBlock
//                           ? pointsInCell[thisWarp.thread_rank() * 4 + warpId]
//                           : 0;
// #pragma unroll
//         for (int offset = 1; offset < warpsPerBlock; offset *= 2) {
//             int n = thisWarp.shfl_up(totalPointsCountPerCell, offset);
//
//             if (thisWarp.thread_rank() >= offset) totalPointsCountPerCell += n;
//         }
//
//         if (thisWarp.thread_rank() < warpsPerBlock)
//             pointsInCell[thisWarp.thread_rank() * 4 + warpId] = totalPointsCountPerCell;
//     }


    if (warpId < 4) {
        int totalPointsCountPerCell = (thisWarp.thread_rank() < warpsPerBlock)
                              ? pointsInCell[thisWarp.thread_rank() + warpId * warpsPerBlock]
                              : 0;
    #pragma unroll
        for (int offset = 1; offset < warpsPerBlock; offset *= 2) {
            int n = thisWarp.shfl_up(totalPointsCountPerCell, offset);
            if (thisWarp.thread_rank() >= offset)
                totalPointsCountPerCell += n;
        }
        if (thisWarp.thread_rank() < warpsPerBlock)
            pointsInCell[thisWarp.thread_rank() + warpId * warpsPerBlock] = totalPointsCountPerCell;
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
    // int offset1 = pointsInCell[warpId * 4 + 0];
    // int offset2 = pointsInCell[warpId * 4 + 1];
    // int offset3 = pointsInCell[warpId * 4 + 2];
    // int offset4 = pointsInCell[warpId * 4 + 3];
    thisBlock.sync();

    // ---- Begin replacement for exclusive scan conversion ----

    // Broadcast valid counts for each quadrant from warp0 to all warps.
    // For each quadrant q, the value at index (q * warpsPerBlock) is assumed to be valid.
    if (threadIdx.x < 4) {  // one thread per quadrant (from warp0)
        int q = threadIdx.x;            // quadrant index (0: top-left, 1: top-right, etc.)
        int base = q * warpsPerBlock;     // starting index in shared memory for quadrant q
        int finalCount = pointsInCell[base];  // valid count computed by warp0
        // Broadcast finalCount to all warps for this quadrant.
        for (int i = 1; i < warpsPerBlock; i++) {
            pointsInCell[base + i] = finalCount;
        }
    }
    __syncthreads();

    // Now, compute global offsets for each quadrant with a manual exclusive scan.
    // Read the total counts for each quadrant.
    int totalCount0 = pointsInCell[0];                   // Quadrant 0 count
    int totalCount1 = pointsInCell[1 * warpsPerBlock];     // Quadrant 1 count
    int totalCount2 = pointsInCell[2 * warpsPerBlock];     // Quadrant 2 count
    int totalCount3 = pointsInCell[3 * warpsPerBlock];     // Quadrant 3 count

    // Compute exclusive scan (global offsets):
    int globalOffset0 = node.startId;                   // Start offset for quadrant 0
    int globalOffset1 = globalOffset0 + totalCount0;      // Start offset for quadrant 1
    int globalOffset2 = globalOffset1 + totalCount1;      // Start offset for quadrant 2
    int globalOffset3 = globalOffset2 + totalCount2;      // Start offset for quadrant 3
    int globalOffset4 = globalOffset3 + totalCount3;      // End offset (should equal node.endId)

    // For the reordering phase, use local copies of these offsets.
    int offset0 = globalOffset0;
    int offset1 = globalOffset1;
    int offset2 = globalOffset2;
    int offset3 = globalOffset3;
    __syncthreads();
    // ---- End replacement for exclusive scan conversion ----

    //
    // 4- Move points.
    //

    // // Reorder points.
    // for (int i = startId + thisWarp.thread_rank(); thisWarp.any(i < endId); i += warpSize)
    // {
    //     // Is it still an active thread?
    //     bool isInRange = i < endId;
    //
    //     // Load the coordinates of the point
    //     const auto p = isInRange ? pointsExch[i].position : float3{};
    //
    //     /// Count top-left points.
    //     bool isTopLeft = isInRange && p.x < center.x && p.y >= center.y;
    //     const auto mask1 = thisWarp.ballot(isTopLeft);
    //     const auto destId1 = offset1 + __popc(mask1 & lane_mask_lt);
    //
    //     if (isTopLeft)
    //         points[destId1].position = p;
    //
    //     offset1 += thisWarp.shfl(__popc(mask1), 0);
    //
    //     /// Process top-right points
    //     const auto isTopRight = isInRange && p.x >= center.x && p.y >= center.y;
    //     const auto mask2 = thisWarp.ballot(isTopRight);
    //     const auto destId2 = offset2 + __popc(mask2 & lane_mask_lt);
    //
    //     if (isTopRight)
    //         points[destId2].position = p;
    //
    //     offset2 += thisWarp.shfl(__popc(mask2), 0);
    //
    //     /// Process bottom-left points
    //     const auto isBottomLeft = isInRange && p.x < center.x && p.y < center.y;
    //     const auto mask3 = thisWarp.ballot(isBottomLeft);
    //     const auto destId3 = offset3 + __popc(mask3 & lane_mask_lt);
    //
    //     if (isBottomLeft)
    //         points[destId3].position = p;
    //
    //     offset3 += thisWarp.shfl(__popc(mask3), 0);
    //
    //     /// Process bottom-right points
    //     const auto isBottomRight = isInRange && p.x >= center.x && p.y < center.y;
    //     const auto mask4 = thisWarp.ballot(isBottomRight);
    //     const auto destId4 = offset4 + __popc(mask4 & lane_mask_lt);
    //
    //     if (isBottomRight)
    //         points[destId4].position = p;
    //
    //     offset4 += thisWarp.shfl(__popc(mask4), 0);
    // }
    // cg::sync(thisBlock);

    // Reorder points.
    for (int i = startId + thisWarp.thread_rank(); thisWarp.any(i < endId); i += warpSize) {
        bool isInRange = i < endId;
        const auto p = isInRange ? pointsExch[i].position : float3{};

        // Process top-left (quadrant 0)
        bool isTopLeft = isInRange && p.x < center.x && p.y >= center.y;
        const auto mask1 = thisWarp.ballot(isTopLeft);
        const auto destId0 = offset0 + __popc(mask1 & lane_mask_lt);
        if (isTopLeft)
            points[destId0].position = p;
        offset0 += thisWarp.shfl(__popc(mask1), 0);

        // Process top-right (quadrant 1)
        bool isTopRight = isInRange && p.x >= center.x && p.y >= center.y;
        const auto mask2 = thisWarp.ballot(isTopRight);
        const auto destId1 = offset1 + __popc(mask2 & lane_mask_lt);
        if (isTopRight)
            points[destId1].position = p;
        offset1 += thisWarp.shfl(__popc(mask2), 0);

        // Process bottom-left (quadrant 2)
        bool isBottomLeft = isInRange && p.x < center.x && p.y < center.y;
        const auto mask3 = thisWarp.ballot(isBottomLeft);
        const auto destId2 = offset2 + __popc(mask3 & lane_mask_lt);
        if (isBottomLeft)
            points[destId2].position = p;
        offset2 += thisWarp.shfl(__popc(mask3), 0);

        // Process bottom-right (quadrant 3)
        bool isBottomRight = isInRange && p.x >= center.x && p.y < center.y;
        const auto mask4 = thisWarp.ballot(isBottomRight);
        const auto destId3 = offset3 + __popc(mask4 & lane_mask_lt);
        if (isBottomRight)
            points[destId3].position = p;
        offset3 += thisWarp.shfl(__popc(mask4), 0);
    }
    cg::sync(thisBlock);


    //
    // 5- Launch new blocks.
    //

    node.isLeaf = false;
    if ((depth < configTree.maxDepth && numPoints > configTree.minPointsToDivide))
    {
        // The last thread launches new blocks.
        if (threadIdx.x == configTree.threadsPerBlock - 1) {
            auto childOffset = getNumNodesInCurrentDepth<2>(depth);
            QuadTree* children = &tree[childOffset];
            int treeIdNext = 4 * node.id;

            // Child 0 (top-left)
            children[treeIdNext + 0].id = treeIdNext + 0;
            children[treeIdNext + 0].bounds.min = {aabb.min.x, center.y};
            children[treeIdNext + 0].bounds.max = {center.x, aabb.max.y};
            children[treeIdNext + 0].startId = globalOffset0;
            children[treeIdNext + 0].endId   = globalOffset1;

            // Child 1 (top-right)
            children[treeIdNext + 1].id = treeIdNext + 1;
            children[treeIdNext + 1].bounds.min = center;
            children[treeIdNext + 1].bounds.max = aabb.max;
            children[treeIdNext + 1].startId = globalOffset1;
            children[treeIdNext + 1].endId   = globalOffset2;

            // Child 2 (bottom-left)
            children[treeIdNext + 2].id = treeIdNext + 2;
            children[treeIdNext + 2].bounds.min = aabb.min;
            children[treeIdNext + 2].bounds.max = center;
            children[treeIdNext + 2].startId = globalOffset2;
            children[treeIdNext + 2].endId   = globalOffset3;

            // Child 3 (bottom-right)
            children[treeIdNext + 3].id = treeIdNext + 3;
            children[treeIdNext + 3].bounds.min = {center.x, aabb.min.y};
            children[treeIdNext + 3].bounds.max = {aabb.max.x, center.y};
            children[treeIdNext + 3].startId = globalOffset3;
            children[treeIdNext + 3].endId   = globalOffset4;

            // Launch 4 children.
            QuadTreeKernel<<<4, thisBlock.size(), warpsPerBlock * 4 * sizeof(int)>>>(
                pointsExch, points, &children[treeIdNext],
                depth + 1, configTree);
        }
    }
}*/


template<int NUM_THREADS_PER_BLOCK>
__global__ void QuadTreeKernel(QuadTree *nodes, Particle *points, Particle *pointsExch, TreeConfig params) {
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
    Particle *inputPoints = (params.depth % 2 == 0) ? points : pointsExch;
    Particle *outputPoints = (params.depth % 2 == 0) ? pointsExch : points;

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    // Compute the number of points.
    for (int range_it = range_begin + tile32.thread_rank();
         tile32.any(range_it < range_end); range_it += warpSize) {
        // Is it still an active thread?
        bool is_active = range_it < range_end;

        // Load the coordinates of the point.
        float3 p = is_active ? inputPoints[range_it].position : make_float3(0.0f, 0.0f,0.0f);

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

            // Load the coordinates of the point.
            float3 p = is_active ? inputPoints[range_it].position : make_float3(0.0f, 0.0f,0.0f);
            // Get the full particle data
            // Particle particle = is_active ? inputPoints[range_it] : Particle();
            float3 particle = is_active ? inputPoints[range_it].position : make_float3(0.0f, 0.0f,0.0f);

            // Count top-left points.
            bool pred = is_active && p.x < center.x && p.y >= center.y;
            int vote = tile32.ballot(pred);
            int dest = warp_cnts[0] + __popc(vote & lane_mask_lt);

            // if (pred) out_points.set_point(dest, p);
            if (pred) outputPoints[dest].position = particle;

            warp_cnts[0] += tile32.shfl(__popc(vote), 0);

            // Count top-right points.
            pred = is_active && p.x >= center.x && p.y >= center.y;
            vote = tile32.ballot(pred);
            dest = warp_cnts[1] + __popc(vote & lane_mask_lt);

            // if (pred) out_points.set_point(dest, p);
            if (pred) outputPoints[dest].position = particle;

            warp_cnts[1] += tile32.shfl(__popc(vote), 0);

            // Count bottom-left points.
            pred = is_active && p.x < center.x && p.y < center.y;
            vote = tile32.ballot(pred);
            dest = warp_cnts[2] + __popc(vote & lane_mask_lt);

            // if (pred) out_points.set_point(dest, p);
            if (pred) outputPoints[dest].position = particle;

            warp_cnts[2] += tile32.shfl(__popc(vote), 0);

            // Count bottom-right points.
            pred = is_active && p.x >= center.x && p.y < center.y;
            vote = tile32.ballot(pred);
            dest = warp_cnts[3] + __popc(vote & lane_mask_lt);

            // if (pred) out_points.set_point(dest, p);
            if (pred) outputPoints[dest].position = particle;

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
            QuadTreeKernel<NUM_THREADS_PER_BLOCK><<<
                    4, NUM_THREADS_PER_BLOCK, 4 * NUM_WARPS_PER_BLOCK * sizeof(int)>>>(
                        &children[child_offset], pointsExch, points, TreeConfig(params, true));
        }
    }
}


#endif //QUADTREEBUILDERKERNEL_CUH
