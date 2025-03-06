#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <memory>
#include <ContactDetection/BroadPhase/QuadTree/QuadTree.h>
#include <Particle/Particle.hpp>
#include <ContactDetection/BroadPhase/Config/TreeConfig.h>

class QuadTreeWriter {
public:
    static void writeQuadTree(const std::string& filename, const QuadTree* rootTree, const TreeConfig& config) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        // Data structures for collecting QuadTree information
        std::vector<float2> points;
        std::vector<int> cells;
        std::vector<int> offsets;
        std::vector<int> cellTypes;
        std::vector<int> leafDepths;
        std::vector<int> pointCounts;
        std::vector<int> nodeIds;

        // Process nodes level by level
        const QuadTree* treeLevel = rootTree;
        
        // First, collect node information at each level
        for (int depth = 0; depth < config.maxDepth; ++depth) {
            const int nodesAtDepth = getNumNodesInCurrentDepth<2>(depth);
            
            for (int nodeIdx = 0; nodeIdx < nodesAtDepth; ++nodeIdx) {
                const QuadTree* node = treeLevel + nodeIdx;
                
                // Skip empty nodes
                if (node->startId == node->endId) {
                    continue;
                }
                
                // Non-empty nodes are added to the visualization
                collectNodeData(node, depth, points, cells, offsets, cellTypes, 
                               leafDepths, pointCounts, nodeIds);
            }
            
            // Move to the next level in the tree
            treeLevel += nodesAtDepth;
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
    }

private:
    static void collectNodeData(
        const QuadTree* node,
        int depth,
        std::vector<float2>& points,
        std::vector<int>& cells,
        std::vector<int>& offsets,
        std::vector<int>& cellTypes,
        std::vector<int>& leafDepths,
        std::vector<int>& pointCounts,
        std::vector<int>& nodeIds
    ) {
        // Add the current cell (quad)
        size_t pointStartIndex = points.size();
        
        // Add the four corners of the cell (bottom-left, bottom-right, top-right, top-left)
        points.push_back({node->bounds.min.x, node->bounds.min.y});                                  // Bottom-left
        points.push_back({node->bounds.max.x, node->bounds.min.y});          // Bottom-right
        points.push_back({node->bounds.max.x, node->bounds.max.y});                                  // Top-right
        points.push_back({node->bounds.min.x, node->bounds.max.y});          // Top-left
        
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
        pointCounts.push_back(node->maxParticlesPerNode());
        nodeIds.push_back(node->id);
    }
};