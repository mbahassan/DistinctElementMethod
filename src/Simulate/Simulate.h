//
// Created by iqraa on 1-4-25.
//

#ifndef SIMULATE_H
#define SIMULATE_H

#include <ContactDetection/ContactDetection.cuh>
#include <ContactDetection/BroadPhase/Config/TreeType.h>
#include <TimeIntegrator/TimeIntegrator.h>
#include <ForceModels/ForceModel.cuh>

template<typename ParticleType>
class Simulate
{
public:
    explicit Simulate(const float dt = 0.001,
                      const Model forceModel = HertzMindlin,
                      const Integrator integrator = Euler,
                      const TreeType treeType = QUADTREE)
    {
        dt_ = dt;
        // forceModel_ = forceModel;

        integrator_ = integrator;
        treeType_ = treeType;

        contactDetection_ = ContactDetection<ParticleType>(treeType);
        forceModel_ = ForceModel<ParticleType>(forceModel);
        timeIntegrator_ = TimeIntegrator<ParticleType>(integrator);
    }

    /// Advance the simulation by one timestep
    void step() {
        // Reset forces
        for (auto &p: particles)
        {
            p.force = 0;
            p.torque = 0;
        }

        // Apply Gravity Forces (gravity, etc.)
        // applyGravityForces();

        /// Detect contacts
        auto contacts = contactDetection_.detectContacts(particles);

        /// Calculate contact forces
        for (auto &contact: contacts) {
            // force = self.contact_force_model.calculate_contact_force(contact);

            //  Apply forces to particles
            // p1 = contact["particle1"];
            // p2 = contact["particle2"];


            // p1.force[i] -= force; // Action
            // p2.force[i] += force; // Reaction

            // # Calculate and apply torques
            // # (Simplified - would need more details for non-central collisions)
        }

        //  Integrate motion
        timeIntegrator_.step(particles, dt_);

        // Update the time
        time_ += dt_;
    }

    /// Run the simulation until the specified end time
    void run(const float endTime)
    {
        while(time_ < endTime)
        {
            step();

            //  Here you would add code for:
            //  - Saving data at specific intervals
            //  - Visualization
            //  - Checking termination conditions
        }
    }

private:
    float time_ = 0.0f;
    float dt_ = 0.0001;
    // ForceModels forceModel_ = HertzMindelin;
    Integrator integrator_ = Euler;
    TreeType treeType_ = QUADTREE;

    /// Main Components
    ContactDetection<ParticleType> contactDetection_;
    TimeIntegrator<ParticleType> timeIntegrator_;
    ForceModel<ParticleType> forceModel_;
    std::vector<ParticleType> particles;
};


#endif //SIMULATE_H
