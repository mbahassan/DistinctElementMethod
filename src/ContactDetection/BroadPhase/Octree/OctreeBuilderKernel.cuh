//
// Created by iqraa on 28-2-25.
//

#ifndef OCTREEBUILDERKERNEL_CUH
#define OCTREEBUILDERKERNEL_CUH

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



template<int NUM_THREADS_PER_BLOCK>
__global__ void OctreeKernel(Octree *nodes, Spherical *points, Spherical *pointsExch, TreeConfig params) {
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
    Octree &node = nodes[blockIdx.x];

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
    // Define the child quadrants
    AxisAlignedBoundingBox<float2> childBounds[4];
    // Top-left
    childBounds[0].min = {bbox.min.x, center.y};
    childBounds[0].max = {center.x, bbox.max.y};
    // Top-right
    childBounds[1].min = {center.x, center.y};
    childBounds[1].max = {bbox.max.x, bbox.max.y};
    // Bottom-left
    childBounds[2].min = {bbox.min.x, bbox.min.y};
    childBounds[2].max = {center.x, center.y};
    // Bottom-right
    childBounds[3].min = {center.x, bbox.min.y};
    childBounds[3].max = {bbox.max.x, center.y};

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
    Spherical *inputPoints = (params.depth % 2 == 0) ? points : pointsExch;
    Spherical *outputPoints = (params.depth % 2 == 0) ? pointsExch : points;

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    // Compute the number of points.
    for (int range_it = range_begin + tile32.thread_rank();
         tile32.any(range_it < range_end); range_it += warpSize) {
        // Is it still an active thread?
        bool is_active = range_it < range_end;

        // Check if particle's bounding box overlaps with each quadrant
        if (is_active) {
            // Get the particle's bounding box
            const auto& particleBBox = inputPoints[range_it].boundingBox;

            // A particle is counted in a quadrant if its bounding box overlaps with that quadrant
            // Top-left quadrant
            bool overlapsTopLeft = !(particleBBox.max.x <= childBounds[0].min.x ||
                                     particleBBox.min.x >= childBounds[0].max.x ||
                                     particleBBox.max.y <= childBounds[0].min.y ||
                                     particleBBox.min.y >= childBounds[0].max.y);
            int num_pts = __popc(tile32.ballot(overlapsTopLeft));
            warp_cnts[0] += tile32.shfl(num_pts, 0);

            // Top-right quadrant
            bool overlapsTopRight = !(particleBBox.max.x <= childBounds[1].min.x ||
                                      particleBBox.min.x >= childBounds[1].max.x ||
                                      particleBBox.max.y <= childBounds[1].min.y ||
                                      particleBBox.min.y >= childBounds[1].max.y);
            num_pts = __popc(tile32.ballot(overlapsTopRight));
            warp_cnts[1] += tile32.shfl(num_pts, 0);

            // Bottom-left quadrant
            bool overlapsBottomLeft = !(particleBBox.max.x <= childBounds[2].min.x ||
                                        particleBBox.min.x >= childBounds[2].max.x ||
                                        particleBBox.max.y <= childBounds[2].min.y ||
                                        particleBBox.min.y >= childBounds[2].max.y);
            num_pts = __popc(tile32.ballot(overlapsBottomLeft));
            warp_cnts[2] += tile32.shfl(num_pts, 0);

            // Bottom-right quadrant
            bool overlapsBottomRight = !(particleBBox.max.x <= childBounds[3].min.x ||
                                         particleBBox.min.x >= childBounds[3].max.x ||
                                         particleBBox.max.y <= childBounds[3].min.y ||
                                         particleBBox.min.y >= childBounds[3].max.y);
            num_pts = __popc(tile32.ballot(overlapsBottomRight));
            warp_cnts[3] += tile32.shfl(num_pts, 0);
        }
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

            if (is_active) {
                Spherical particle = inputPoints[range_it];
                const auto& particleBBox = particle.boundingBox;

                // Check overlaps with each quadrant and place particle in all overlapping quadrants
                // Top-left quadrant
                bool overlapsTopLeft = !(particleBBox.max.x <= childBounds[0].min.x ||
                                         particleBBox.min.x >= childBounds[0].max.x ||
                                         particleBBox.max.y <= childBounds[0].min.y ||
                                         particleBBox.min.y >= childBounds[0].max.y);
                int vote = tile32.ballot(overlapsTopLeft);
                int dest = warp_cnts[0] + __popc(vote & lane_mask_lt);
                if (overlapsTopLeft) outputPoints[dest] = particle;
                warp_cnts[0] += tile32.shfl(__popc(vote), 0);

                // Top-right quadrant
                bool overlapsTopRight = !(particleBBox.max.x <= childBounds[1].min.x ||
                                          particleBBox.min.x >= childBounds[1].max.x ||
                                          particleBBox.max.y <= childBounds[1].min.y ||
                                          particleBBox.min.y >= childBounds[1].max.y);
                vote = tile32.ballot(overlapsTopRight);
                dest = warp_cnts[1] + __popc(vote & lane_mask_lt);
                if (overlapsTopRight) outputPoints[dest] = particle;
                warp_cnts[1] += tile32.shfl(__popc(vote), 0);

                // Bottom-left quadrant
                bool overlapsBottomLeft = !(particleBBox.max.x <= childBounds[2].min.x ||
                                            particleBBox.min.x >= childBounds[2].max.x ||
                                            particleBBox.max.y <= childBounds[2].min.y ||
                                            particleBBox.min.y >= childBounds[2].max.y);
                vote = tile32.ballot(overlapsBottomLeft);
                dest = warp_cnts[2] + __popc(vote & lane_mask_lt);
                if (overlapsBottomLeft) outputPoints[dest] = particle;
                warp_cnts[2] += tile32.shfl(__popc(vote), 0);

                // Bottom-right quadrant
                bool overlapsBottomRight = !(particleBBox.max.x <= childBounds[3].min.x ||
                                             particleBBox.min.x >= childBounds[3].max.x ||
                                             particleBBox.max.y <= childBounds[3].min.y ||
                                             particleBBox.min.y >= childBounds[3].max.y);
                vote = tile32.ballot(overlapsBottomRight);
                dest = warp_cnts[3] + __popc(vote & lane_mask_lt);
                if (overlapsBottomRight) outputPoints[dest] = particle;
                warp_cnts[3] += tile32.shfl(__popc(vote), 0);
            }
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
            Octree *children = &nodes[params.numNodesAtThisLevel - (node.id & ~3)];

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
            OctreeKernel<NUM_THREADS_PER_BLOCK><<<
                    4, NUM_THREADS_PER_BLOCK, 4 * NUM_WARPS_PER_BLOCK * sizeof(int)>>>(
                        &children[child_offset], pointsExch, points, TreeConfig(params, true));
        }
    }
}


#endif //OCTREEBUILDERKERNEL_CUH
