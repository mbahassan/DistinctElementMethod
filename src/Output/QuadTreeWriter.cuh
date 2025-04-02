
#ifndef QUADTREE_WRITER_H
#define QUADTREE_WRITER_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <ContactDetection/BroadPhase/QuadTree/QuadTree.h>
#include <Particle/Spherical.h>
#include <ContactDetection/BroadPhase/Config/TreeConfig.h>

class QuadTreeWriter {
public:
    static void writeQuadTree(
        const std::string& filename,
        const QuadTree* tree,
        const TreeConfig& config
        ){

        std::ofstream file(filename);
        if (!file.is_open())
            {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        // Store a copy of the original tree pointer
        const QuadTree* originalTree = tree;

        // Print debug info to understand the tree structure
        std::cout << "===== QuadTree Debug Info =====" << std::endl;
        std::cout << "Root ID: " << tree->id << std::endl;
        std::cout << "Root Bounds: (" << tree->bounds.min.x << "," << tree->bounds.min.y << ") to ("
                 << tree->bounds.max.x << "," << tree->bounds.max.y << ")" << std::endl;
        std::cout << "Root Particles: " << (tree->endId - tree->startId) << std::endl;
        std::cout << "Max Depth: " << config.maxDepth << std::endl;

        // Inspect first few nodes to understand the tree layout
        std::cout << "\nFlattened Tree Inspection:" << std::endl;
        std::cout << "Index\tID\tParticles\tBounds" << std::endl;

        // Make a temporary copy of the tree pointer for inspection
        const QuadTree* inspectionTree = tree;

        // Save node pointers at each depth for processing later
        std::vector<const QuadTree*> depthStartPointers;
        depthStartPointers.push_back(originalTree);

        for (int depth = 0; depth < config.maxDepth; depth++) {
            const auto leafs = getNumNodesInCurrentDepth<2>(depth);

            if (depth + 1 < config.maxDepth) {
                depthStartPointers.push_back(inspectionTree + leafs);
            }

            for (int leaf = 0; leaf < leafs; ++leaf) {
                const QuadTree* node = &inspectionTree[leaf];
                int particleCount = node->endId - node->startId;
                std::cout << leaf << "\t" << node->id << "\t" << particleCount << "\t\t("
                         << node->bounds.min.x << "," << node->bounds.min.y << ") to ("
                         << node->bounds.max.x << "," << node->bounds.max.y << ")" << std::endl;
            }
            inspectionTree += leafs;
        }
        std::cout << "=============================" << std::endl;

        // Data structures for collecting QuadTree information
        std::vector<float3> points;
        std::vector<int> cells;
        std::vector<int> offsets;
        std::vector<int> cellTypes;
        std::vector<int> leafDepths;
        std::vector<int> pointCounts;
        std::vector<int> nodeIds;

        // Process all nodes at each depth level
        for (int depth = 0; depth < config.maxDepth; depth++) {
            const auto nodesAtDepth = getNumNodesInCurrentDepth<2>(depth);
            const QuadTree* depthTree = depthStartPointers[depth];

            std::cout << "Processing depth " << depth << " with " << nodesAtDepth << " nodes" << std::endl;

            for (int nodeIdx = 0; nodeIdx < nodesAtDepth; nodeIdx++) {
                const QuadTree* node = &depthTree[nodeIdx];

                // Only process nodes with particles
                if (node->startId < node->endId) {
                    std::cout << "  Node ID " << node->id
                             << " with " << (node->endId - node->startId)
                             << " particles" << std::endl;

                    collectNodeData(node, depth, points, cells, offsets, cellTypes,
                                   leafDepths, pointCounts, nodeIds);
                }
            }
        }

        // Begin XML file
        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        file << "  <UnstructuredGrid>\n";
        file << "    <Piece NumberOfPoints=\"" << points.size() << "\" NumberOfCells=\"" << cellTypes.size() << "\">\n";

        // Write points
        file << "      <Points>\n";
        file << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (const auto& point : points) {
            file << "          " << point.x << " " << point.y << " 0.0\n";
        }
        file << "        </DataArray>\n";
        file << "      </Points>\n";

        // Write cells
        file << "      <Cells>\n";

        // Connectivity
        file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        file << "          ";
        for (const auto& cellPoint : cells) {
            file << cellPoint << " ";
        }
        file << "\n";
        file << "        </DataArray>\n";

        // Offsets
        file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        file << "          ";
        for (const auto& offset : offsets) {
            file << offset << " ";
        }
        file << "\n";
        file << "        </DataArray>\n";

        // Types
        file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        file << "          ";
        for (const auto& type : cellTypes) {
            file << type << " ";
        }
        file << "\n";
        file << "        </DataArray>\n";
        file << "      </Cells>\n";

        // Cell data
        file << "      <CellData Scalars=\"depth\">\n";

        // Depth data
        file << "        <DataArray type=\"Int32\" Name=\"depth\" format=\"ascii\">\n";
        file << "          ";
        for (const auto& depth : leafDepths) {
            file << depth << " ";
        }
        file << "\n";
        file << "        </DataArray>\n";

        // Point count data
        file << "        <DataArray type=\"Int32\" Name=\"point_count\" format=\"ascii\">\n";
        file << "          ";
        for (const auto& count : pointCounts) {
            file << count << " ";
        }
        file << "\n";
        file << "        </DataArray>\n";

        // Node ID data
        file << "        <DataArray type=\"Int32\" Name=\"node_id\" format=\"ascii\">\n";
        file << "          ";
        for (const auto& id : nodeIds) {
            file << id << " ";
        }
        file << "\n";
        file << "        </DataArray>\n";

        file << "      </CellData>\n";

        file << "    </Piece>\n";
        file << "  </UnstructuredGrid>\n";
        file << "</VTKFile>\n";

        file.close();

        std::cout << "VTK file written with " << cellTypes.size() << " cells." << std::endl;
    }

private:
    static void collectNodeData(
        const QuadTree* node,
        int depth,
        std::vector<float3>& points,
        std::vector<int>& cells,
        std::vector<int>& offsets,
        std::vector<int>& cellTypes,
        std::vector<int>& leafDepths,
        std::vector<int>& pointCounts,
        std::vector<int>& nodeIds
    ) {
        // Add the current cell (quad)
        size_t pointStartIndex = points.size();

        // Validate the bounds
        float minX = node->bounds.min.x;
        float minY = node->bounds.min.y;
        float maxX = node->bounds.max.x;
        float maxY = node->bounds.max.y;

        // Add the four corners of the cell (bottom-left, bottom-right, top-right, top-left)
        points.push_back({minX, minY, 0.0f});  // Bottom-left
        points.push_back({maxX, minY, 0.0f});  // Bottom-right
        points.push_back({maxX, maxY, 0.0f});  // Top-right
        points.push_back({minX, maxY, 0.0f});  // Top-left

        // Add cell connectivity
        for (int i = 0; i < 4; i++) {
            cells.push_back(pointStartIndex + i);
        }

        // Update offsets
        offsets.push_back(cells.size());

        // Add cell type - 9 is VTK_QUAD
        cellTypes.push_back(9);

        // Add metadata
        leafDepths.push_back(depth);

        // Calculate particle count directly
        int particleCount = node->endId - node->startId;
        pointCounts.push_back(particleCount);

        // Add node ID
        nodeIds.push_back(node->id);
    }
};


#endif //QUADTREE_WRITER_H