//
// Created by mbahassan on 3/13/25.
//

#ifndef CONTACT_DETECTION_CUH
#define CONTACT_DETECTION_CUH

#include <vector>
#include "BroadPhase/BroadPhase.cuh"
#include "NarrowPhase/NarrowPhase.cuh"
#include "ContactDetection/BroadPhase/Config/TreeType.h"
#include <Particle/Spherical.hpp>
#include <Particle/Polyhedral.hpp>
#include "ContactConfig.h"

template<class ParticleType>
class ContactDetection : ContactConfig
{
public:
    explicit ContactDetection(const TreeType treeType): broadPhase_(treeType) {}

    explicit ContactDetection(const std::string& path): broadPhase_(path) {}

    // Run the complete contact detection pipeline
    std::vector<EPA::Contact> detectContacts(std::vector<ParticleType>& particles) {
        // Initialize the broad phase (build the spatial data structure)
        broadPhase_.initialize(particles);

        // Run broad phase to get potential contacts
        std::vector<PotentialContact> potentialContacts = broadPhase_.getPotentialContacts(particles);

        // Run narrow phase to get actual contacts
        return narrowPhase_.detectCollisions(particles, potentialContacts);
    }

    // You can also provide separate methods if needed
    std::vector<PotentialContact> broadPhase(std::vector<ParticleType>& particles)
    {
        broadPhase_.initialize(particles);
        return broadPhase_.getPotentialContacts(particles);
    }

    std::vector<EPA::Contact> narrowPhase(
        const std::vector<ParticleType>& particles,
        const std::vector<PotentialContact>& potentialContacts) {
        return narrowPhase_.detectCollisions(particles, potentialContacts);
    }

private:
    BroadPhase<ParticleType> broadPhase_;

    NarrowPhase narrowPhase_;
};

#endif // CONTACT_DETECTION_CUH