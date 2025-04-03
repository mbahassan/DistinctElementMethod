//
// Created by iqraa on 1-4-25.
//

#ifndef SIMULATE_H
#define SIMULATE_H

#include <ContactDetection/ContactDetection.cuh>
#include <TimeIntegrator/TimeIntegrator.cuh>
#include <ForceModels/ForceModel.cuh>
#include <Output/Output.cuh>

template<typename ParticleType>
class Simulate {
public:
    explicit Simulate(const float dt = 0.001,
                      const Model forceModel = HertzMindlin,
                      const Integrator integrator = Euler,
                      const TreeType treeType = Quadtree,
                      const std::string &outputDir = "Results"): cd_(treeType),
                                                                 ti_(integrator),
                                                                 fm_(forceModel),
                                                                 output_(outputDir) {
        dt_ = dt;
    }

    explicit Simulate(const float dt = 0.001,
                      const Model forceModel = HertzMindlin,
                      const Integrator integrator = Euler,
                      const std::string &treefile = "input.json",
                      const std::string &outputDir = "Results"): cd_(treefile),
                                                                 ti_(integrator),
                                                                 fm_(forceModel),
                                                                 output_(outputDir) {
        dt_ = dt;
    }

    void addParticles(std::vector<ParticleType> &particles) { particles_ = particles; }

    /// Run the simulation until the specified end time
    void solve(const float endTime) {
        tend = endTime;
        unsigned int counter = 0;
        float writeInterval = 0.1;
        float writeTime = 0.0f;

        /// write 0 timestep
        output_.writeParticles(particles_, counter);

        // Detect contacts
        auto contacts = cd_.detectContacts(particles_);
        output_.writeTree(&cd_.getTreeBuilder()->getTree(), counter);

        /// Time-Loop
        while (time_ < tend)
        {
            /// perform one time step
            oneTimeStep();

            //  Here you would add code for:
            if (time_ >= writeTime)
            {
                //  - Saving data at specific intervals

                ///  - Visualization
                output_.writeParticles(particles_, counter);
                output_.writeTree(&cd_.getTreeBuilder()->getTree(), counter);

                ///  - Checking termination conditions

                /// increment counter
                counter++;

                // Update next output time
                writeTime = time_ + writeInterval;
            }


            // Update the time
            time_ += dt_;
        }
    }

private:
    float dt_ = 0.0001;

    float time_ = dt_;
    float tend = time_;

    float3 gravity_{0.f, -9.81f, 0.f};

    /// Main Components
    ContactDetection<ParticleType> cd_;

    TimeIntegrator<ParticleType> ti_;

    ForceModel fm_;

    std::vector<ParticleType> particles_;

    Output output_;

    /// Advance the simulation by one timestep
    void oneTimeStep() {
        // Reset forces
        for (auto &p: particles_) {
            p.force = {0.f, 0.f, 0.f};
            p.torque = {0.f, 0.f, 0.f};
        }

        // Apply Gravity Forces (gravity, etc.)
        applyGravityForces();

        /// Detect contacts
        auto contacts = cd_.detectContacts(particles_);

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

        //  Time Integrate
        ti_.integrate(particles_, dt_);
    }

    void applyGravityForces() {
        for (auto &p: particles_) {
            p.force += gravity_ * p.mass;
        }
    }
};


#endif //SIMULATE_H
