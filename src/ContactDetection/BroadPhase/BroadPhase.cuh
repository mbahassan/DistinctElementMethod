//
// Created by mbahassan on 3/13/25.
//

#ifndef BROADPHASE_CUH
#define BROADPHASE_CUH

#include <vector>
#include <unordered_set>
#include <ContactDetection/BroadPhase/QuadTree/QuadTreeBuilder.cuh>
#include <ContactDetection/BroadPhase/Config/TreeType.h>
#include <Output/QuadTreeWriter.cuh>
#include <Particle/Spherical.hpp>

#include "Tools/CudaHelper.hpp"

struct PotentialContact
{
    int particleIdI;
    int particleIdJ;
};


template<class ParticleType>
class BroadPhase {
public:

    explicit BroadPhase(const std::string& path)
    {
        const nlohmann::json data = Parser::readJson(path);

        treeType_ = data["ContactDetection"]["BroadPhase"]["treeType"];
        const int maxDepth = data["ContactDetection"]["BroadPhase"]["maxDepth"];
        const int minPointsPerNode = data["ContactDetection"]["BroadPhase"]["minPointsPerNode"];
        treeConfig_ = TreeConfig(maxDepth, minPointsPerNode);
    }

    explicit BroadPhase(const TreeType treeType)
    : treeConfig_(2, 1){
        treeType_ = treeType;
    }


    void initialize(std::vector<ParticleType>& particles)
    {
        if (treeType_ == QUADTREE)
        {
            size_t particlesCount = particles.size();
            ParticleType* pointsHost = particles.data();
            std::cout << "BroadPhase::initialize(): " << particlesCount << " particles\n";

            ParticleType* points;
            hostToDevice(pointsHost, particlesCount, &points);

            treeBuilder = std::make_unique<QuadTreeBuilder<ParticleType>>(treeConfig_);
            treeBuilder->initialize(particlesCount);
            treeBuilder->build(points, particlesCount);

            QuadTreeWriter::writeQuadTree("./results/quadtree_000000.vtu", &treeBuilder->getTree(), treeConfig_);
            deviceToHost(points, particlesCount, &pointsHost);
        }
    }


    std::vector<PotentialContact> getPotentialContacts(std::vector<ParticleType>& points) const
    {
        std::vector<PotentialContact> potentialContacts;
        std::unordered_set<uint64_t> processedPairs; // To avoid duplicate checks

        // Store all leaf nodes for easier neighbor processing
        std::vector<std::pair<const QuadTree*, int>> leafNodes;

        // First pass: collect all leaf nodes
        const QuadTree* currentTree = &treeBuilder->getTree();
        for (int depth = 0; depth < treeConfig_.maxDepth; ++depth) {
            const auto numNodesAtDepth = getNumNodesInCurrentDepth<2>(depth);

            for (int nodeIdx = 0; nodeIdx < numNodesAtDepth; ++nodeIdx) {
                const QuadTree* subTree = &currentTree[nodeIdx];

                // Check if this is a leaf node with particles
                if ((subTree->particlesCountInNode() < treeConfig_.minPointsPerNode ||
                    depth == treeConfig_.maxDepth - 1) && subTree->particlesCountInNode() > 0) {
                    leafNodes.push_back(std::make_pair(subTree, nodeIdx));
                }
            }

            currentTree += numNodesAtDepth;
        }

        // Second pass: check contacts between particles in each leaf and with particles in neighboring leaves
        for (size_t i = 0; i < leafNodes.size(); ++i)
        {
            const QuadTree* leafA = leafNodes[i].first;
            int nodeIdxA = leafNodes[i].second;

            // First, check particles within the same leaf
            checkContactsInLeaf(leafA, points, potentialContacts);

            // Then check with neighboring leaves
            for (size_t j = i + 1; j < leafNodes.size(); ++j) {
                const QuadTree* leafB = leafNodes[j].first;

                // Check if these leaves are neighbors
                if (areNeighboringNodes(leafA, leafB)) {
                    checkContactsBetweenLeaves(leafA, leafB, points, potentialContacts, processedPairs);
                }
            }
        }

        std::cout << "Found " << potentialContacts.size() << " potential contacts" << std::endl;
        return potentialContacts;
    }

private:

    static void checkContactsInLeaf(
        const QuadTree* leaf,
        std::vector<ParticleType>& points,
        std::vector<PotentialContact>& contacts) {

        const int start = leaf->startId;
        const int end = leaf->endId;

        // Check for potential contacts between particles in this leaf
        for (int i = start; i < end; ++i) {
            for (int j = i + 1; j < end; ++j) {
                ParticleType& p1 = points[i];
                ParticleType& p2 = points[j];

                if (p1.boundingBox.Check(p2.boundingBox.min) ||
                    p1.boundingBox.Check(p2.boundingBox.max)) {
                    // Create a contact pair and add to our vector
                    PotentialContact contact;
                    contact.particleIdI = p1.getId();
                    contact.particleIdJ = p2.getId();
                    contacts.push_back(contact);
                }
            }
        }
    }


    static void checkContactsBetweenLeaves(
        const QuadTree* leafA,
        const QuadTree* leafB,
        std::vector<ParticleType>& points,
        std::vector<PotentialContact>& contacts,
        std::unordered_set<uint64_t>& processedPairs) {

        const int startA = leafA->startId;
        const int endA = leafA->endId;
        const int startB = leafB->startId;
        const int endB = leafB->endId;

        // Check for potential contacts between particles in the two leaves
        for (int i = startA; i < endA; ++i) {
            for (int j = startB; j < endB; ++j) {
                ParticleType& p1 = points[i];
                ParticleType& p2 = points[j];

                // Create a unique pair ID to avoid duplicate checks
                // (using min/max to ensure order doesn't matter)
                int minId = std::min(p1.getId(), p2.getId());
                int maxId = std::max(p1.getId(), p2.getId());
                uint64_t pairId = (static_cast<uint64_t>(minId) << 32) | maxId;

                // Skip if we've already processed this pair
                if (processedPairs.find(pairId) != processedPairs.end()) {
                    continue;
                }

                processedPairs.insert(pairId);
                if (p1.boundingBox.Check(p2.boundingBox.min) ||
                    p1.boundingBox.Check(p2.boundingBox.max)) {
                    // Create a contact pair and add to our vector
                    PotentialContact contact;
                    contact.particleIdI = p1.getId();
                    contact.particleIdJ = p2.getId();
                    contacts.push_back(contact);
                }
            }
        }
    }

    static bool areNeighboringNodes(const QuadTree* nodeA, const QuadTree* nodeB) {
        float maxContactDistance = 0.2;
        // maxContactDistance should be at least 2 * (maximum particle radius)

        // Check if the two nodes are separated along x or y-axis
        bool separatedX = nodeA->bounds.max.x < nodeB->bounds.min.x - maxContactDistance ||
                         nodeB->bounds.max.x < nodeA->bounds.min.x - maxContactDistance;

        bool separatedY = nodeA->bounds.max.y < nodeB->bounds.min.y - maxContactDistance ||
                         nodeB->bounds.max.y < nodeA->bounds.min.y - maxContactDistance;

        // If the nodes are not separated along either axis, they are neighbors
        return !(separatedX || separatedY);
    }


    TreeType  treeType_;
    TreeConfig treeConfig_;
    std::unique_ptr<QuadTreeBuilder<ParticleType>> treeBuilder;

};


#endif // BROADPHASE_CUH