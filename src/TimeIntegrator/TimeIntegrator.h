//
// Created by iqraa on 1-4-25.
//

#ifndef TIMEINTEGRATOR_H
#define TIMEINTEGRATOR_H

#include "EulerIntegrator/EulerIntegrator.cuh"
#include "VerletIntegrator/VerletIntegrator.h"

enum Integrator
{
    Euler,
    Verlet
};

class TimeIntegrator {
public:
    explicit TimeIntegrator(const Integrator method = Euler): method_(method) {
    }

    template<typename ParticleType>
    void step(const std::vector<ParticleType> &particles, const double dt)
    {
        if (method_ == Euler) {
            eulerIntegrator_ = std::make_unique<EulerIntegrator>();
            eulerIntegrator_->eulerStep(particles, dt);
        } else if (method_ == Verlet) {
            verletIntegrator_->verletStep(particles, dt);
        }
    }

private:
    std::unique_ptr<EulerIntegrator> eulerIntegrator_;
    std::unique_ptr<VerletIntegrator> verletIntegrator_;
    Integrator method_;
};


#endif //TIMEINTEGRATOR_H
