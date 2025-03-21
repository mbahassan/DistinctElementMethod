//
// Created by mbahassan on 3/13/25.
//

#ifndef CONTACT_DETECTION_CUH
#define CONTACT_DETECTION_CUH

#include <vector>
#include "BroadPhase/BroadPhase.cuh"
#include "NarrowPhase/NarrowPhase.cuh"
#include "ContactDetection/BroadPhase/Config/TreeType.h"
#include "Particle/Spherical.hpp"
#include "ContactConfig.h"


class ContactDetection : ContactConfig
{
public:
    explicit ContactDetection(const TreeType treeType): broadPhase_(treeType) {}

    explicit ContactDetection(const std::string& path): broadPhase_(path) {}

    // Run the complete contact detection pipeline
    std::vector<EPA::Contact> detectContacts(std::vector<Spherical>& particles) {
        // Initialize the broad phase (build the spatial data structure)
        broadPhase_.initialize(particles);

        // Run broad phase to get potential contacts
        std::vector<PotentialContact> potentialContacts = broadPhase_.findPotentialContacts(particles);

        // Run narrow phase to get actual contacts
        return narrowPhase_.detectCollisions(particles, potentialContacts);
    }

    // You can also provide separate methods if needed
    std::vector<PotentialContact> runBroadPhase(std::vector<Spherical>& particles) {
        broadPhase_.initialize(particles);
        return broadPhase_.findPotentialContacts(particles);
    }

    std::vector<EPA::Contact> runNarrowPhase(
        const std::vector<Spherical>& particles,
        const std::vector<PotentialContact>& potentialContacts) {
        return narrowPhase_.detectCollisions(particles, potentialContacts);
    }

private:
    BroadPhase broadPhase_;

    NarrowPhase narrowPhase_;
};

#endif // CONTACT_DETECTION_CUH