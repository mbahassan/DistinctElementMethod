//
// Created by mbahassan on 2/28/25.
//

#ifndef CONTACTDETECTION_CUH
#define CONTACTDETECTION_CUH

#include "BroadPhase/QuadTree/QuadTreeBuilder.cuh"
#include "ContactDetection/BroadPhase/Config/TreeType.h"
#include "Particle/Particle.hpp"
#include "ContactDetection/BroadPhase/QuadTree/QuadTree.h"
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

    ContactDetection(TreeType treeType) {
        treeType_ = treeType;
    };

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
        }
    }

private:
    TreeType treeType_;
    TreeConfig treeConfig_;
};



#endif //CONTACTDETECTION_CUH
