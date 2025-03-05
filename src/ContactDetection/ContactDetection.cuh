//
// Created by mbahassan on 2/28/25.
//

#ifndef CONTACTDETECTION_CUH
#define CONTACTDETECTION_CUH

#include "BroadPhase/QuadTree/QuadTreeBuilder.cuh"
#include "ContactDetection/BroadPhase/Config/TreeType.h"
#include "Particle/Particle.hpp"
#include "Tools/CudaHelper.hpp"

struct PotentialContact {
    int nodeId;

};
struct Contact {
    Particle pi;
    Particle pj;
    float3 normal;
    float3 contactPoint;
};

class ContactDetection {
    public:
    explicit ContactDetection(const TreeType treeType){
        treeType_ = treeType;
    } ;

    std::vector<PotentialContact> broadPhase(std::vector<Particle>& particles) {
        detectContacts(particles);
        return {};
    }

    std::vector<Contact> narrowPhase(std::vector<PotentialContact>& potentialContact) {
        return {};
    }

    void detectContacts(std::vector<Particle>& particles) {
        if (treeType_ == QUADTREE)
        {
            treeConfig_.origin = {0,0,0};
            treeConfig_.size = {1,1,1};

            auto pointsCount = particles.size();
            Particle* pointsHost = particles.data();
            std::cout << "RunAppOctree(): " << pointsCount << "\n";
            Particle* points;
            hostToDevice(pointsHost, pointsCount, &points);

            auto treeBuilder = std::make_unique<QuadTreeBuilder>(treeConfig_);
            treeBuilder->initialize(pointsCount);
            treeBuilder->build(points, pointsCount);

            deviceToHost(points, pointsCount, &pointsHost);

            const QuadTree* tree2 = &treeBuilder->getTree();
            int totalCount = 0;

            for (int depth = 0; depth < treeConfig_.maxDepth; ++depth)
            {
                const auto leafs = getNumNodesInCurrentDepth<2>(depth);
                for (int leaf = 0; leaf < leafs; ++leaf)
                {
                    const QuadTree* subTree = &tree2[leaf];
                    std::cout<< "Tree id: " << subTree->id << " bounds: ("
                    <<subTree->bounds.min.x<<" "<<subTree->bounds.min.y<<") ("<<subTree->bounds.max.x<<" "
                    <<subTree->bounds.max.y<<" ) startId: "<<subTree->startId << " endId: " << subTree->endId <<std::endl;
                    if ((subTree->maxParticlesPerNode() < treeConfig_.minPointsToDivide ||
                        depth == treeConfig_.maxDepth - 1) && subTree->maxParticlesPerNode() > 0)
                    {
                        totalCount += subTree->maxParticlesPerNode();
                    }
                }

                tree2 += leafs;
            }
        }
    }

private:
    TreeType treeType_;
    TreeConfig treeConfig_ {};
};



#endif //CONTACTDETECTION_CUH
