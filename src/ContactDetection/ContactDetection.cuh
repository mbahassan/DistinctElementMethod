//
// Created by mbahassan on 2/28/25.
//

#ifndef CONTACTDETECTION_CUH
#define CONTACTDETECTION_CUH

#include "ContactDetection/BroadPhase/Config/TreeType.h"
#include "Particle/Particle.hpp"
#include "ContactDetection/BroadPhase/QuadTree/QuadTree.h"

struct PotentialPair {
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

    PotentialPair broadPhase(std::vector<Particle>& particles) {

        return {};
    }

    void detectContacts() {
        if (treeType_ == QUADTREE) {

        }
    }

private:
    TreeType treeType_;
};



#endif //CONTACTDETECTION_CUH
