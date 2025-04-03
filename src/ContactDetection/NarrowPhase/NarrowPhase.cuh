//
// Created by iqraa on 13-3-25.
//

#ifndef NARROWPHASE_CUH
#define NARROWPHASE_CUH

#include <vector>
#include <ContactDetection/BroadPhase/BroadPhase.h>

#include "ContactDetection/NarrowPhase/GJK/GJK.h"
#include "ContactDetection/NarrowPhase/EPA/EPA.h"



class NarrowPhase
{
public:
    NarrowPhase() = default;

    // Detect collisions using GJK + EPA
    template<typename ParticleType>
    std::vector<Contact> detectCollisions(
        const std::vector<ParticleType>& particles,
        const std::vector<PotentialContact>& pContacts) {

        std::vector<Contact> contacts;

        for (const auto& pContact : pContacts) {
            const ParticleType& particleA = particles[pContact.particleIdI];
            const ParticleType& particleB = particles[pContact.particleIdJ];

            // Skip if the particles are the same
            if (particleA.getId() == particleB.getId()) {
                continue;
            }

            // Use GJK to check for overlap and get simplex
            Simplex simplex;
            if (gjk.gjkOverlapWithSimplex(particleA, particleB, simplex)) {
                // Use EPA to compute contact information
                Contact contact = EPA<ParticleType>::computeContactEPA(particleA, particleB, simplex);
                contacts.push_back(contact);
            }
        }

        return contacts;
    }

private:
    GJK gjk;
};

#endif // NARROWPHASE_CUH