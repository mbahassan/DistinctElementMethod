//
// Created by iqraa on 13-3-25.
//

#ifndef NARROWPHASE_CUH
#define NARROWPHASE_CUH

#include <vector>
#include <ContactDetection/BroadPhase/BroadPhase.h>

#include "Particle/Spherical.hpp"
#include "ContactDetection/NarrowPhase/GJK/GJK.h"
#include "ContactDetection/NarrowPhase/EPA/EPA.h"

// Forward declaration to avoid circular dependency
struct PotentialContact;

class NarrowPhase {
public:
    NarrowPhase() = default;

    // Detect collisions using GJK + EPA
    template<typename ParticleType>
    std::vector<EPA::Contact> detectCollisions(
        const std::vector<ParticleType>& particles,
        const std::vector<PotentialContact>& potentialContacts) {

        std::vector<EPA::Contact> contacts;

        for (const auto& potentialContact : potentialContacts) {
            const ParticleType& particleA = particles[potentialContact.particleIdI];
            const ParticleType& particleB = particles[potentialContact.particleIdJ];

            // Skip if the particles are the same
            if (particleA.id == particleB.id) {
                continue;
            }

            // Use GJK to check for overlap and get simplex
            Simplex simplex;
            if (gjk.gjkOverlapWithSimplex(particleA, particleB, simplex)) {
                // Use EPA to compute contact information
                EPA::Contact contact = EPA::computeContactEPA(particleA, particleB, simplex);
                contacts.push_back(contact);
            }
        }

        return contacts;
    }

private:
    GJK gjk;
};

#endif // NARROWPHASE_CUH