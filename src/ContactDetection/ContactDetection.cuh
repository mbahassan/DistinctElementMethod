//
// Created by mbahassan on 3/13/25.
//

#ifndef CONTACT_DETECTION_CUH
#define CONTACT_DETECTION_CUH

#include <vector>
#include "BroadPhase/BroadPhase.h"
#include "NarrowPhase/NarrowPhase.cuh"
#include "ContactDetection/BroadPhase/Config/TreeType.h"
#include "ContactConfig.h"

template<class ParticleType>
class ContactDetection : public ContactConfig , public BroadPhase<ParticleType>
{
public:
    explicit ContactDetection(const TreeType treeType): BroadPhase<ParticleType>(treeType) {}

    explicit ContactDetection(const std::string& path): BroadPhase<ParticleType>(path){}

    // Run the complete contact detection pipeline
    std::vector<Contact> detectContacts(std::vector<ParticleType>& particles)
    {

        // Initialize the broad phase (build the spatial data structure)
        this->initialization(particles);

        // Get potential contacts
        std::vector<PotentialContact> potentialContacts = this->getPotentialContacts(particles);

        // Run narrow phase to get actual contacts
        return narrowPhase_.detectCollisions(particles, potentialContacts);
    }

    // You can also provide separate methods if needed
    std::vector<PotentialContact> broadPhase(std::vector<ParticleType>& particles)
    {
        size_t particlesCount = particles.size();
        ParticleType* particlesHost = particles.data();

        this->initialization(particlesHost, particlesCount);
        return this->getPotentialContacts(particles);
    }

    std::vector<Contact> narrowPhase(
        const std::vector<ParticleType>& particles,
        const std::vector<PotentialContact>& potentialContacts) {
        return narrowPhase_.detectCollisions(particles, potentialContacts);
    }


private:

    NarrowPhase narrowPhase_;
};

#endif // CONTACT_DETECTION_CUH