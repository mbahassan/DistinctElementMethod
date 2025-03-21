//
// Created by iqraa on 13-3-25.
//

#ifndef NARROWPHASE_CUH
#define NARROWPHASE_CUH

#include <vector>
#include <ContactDetection/BroadPhase/BroadPhase.cuh>

#include "Particle/Spherical.hpp"
#include "ContactDetection/NarrowPhase/GJK/GJK.cuh"
#include "ContactDetection/NarrowPhase/EPA/EPA.cuh"

// Forward declaration to avoid circular dependency
struct PotentialContact;

class NarrowPhase {
public:
    NarrowPhase() = default;

    // Detect collisions using GJK + EPA
    std::vector<EPA::Contact> detectCollisions(
        const std::vector<Spherical>& particles,
        const std::vector<PotentialContact>& potentialContacts) {

        std::vector<EPA::Contact> contacts;

        for (const auto& potentialContact : potentialContacts) {
            const Spherical& particleA = particles[potentialContact.particleIdI];
            const Spherical& particleB = particles[potentialContact.particleIdJ];

            // Skip if the particles are the same
            if (particleA.getId() == particleB.getId()) {
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