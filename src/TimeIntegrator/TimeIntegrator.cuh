//
// Created by iqraa on 1-4-25.
//

#ifndef TIME_INTEGRATOR_H
#define TIME_INTEGRATOR_H

#include "EulerIntegrator/EulerIntegrator.cuh"
#include "VerletIntegrator/VerletIntegrator.cuh"

enum Integrator
{
    Euler,
    Verlet
};


template<typename ParticleType>
class TimeIntegrator : public EulerIntegrator<ParticleType>
{
public:
    explicit TimeIntegrator(const Integrator method = Euler): method_(method) {}

    void integrate(std::vector<ParticleType> &particles, const double dt)
    {
        size_t particlesCount = particles.size();
        ParticleType* particlesHost = particles.data();
        if (method_ == Euler)
        {
            eulerIntegrator_ = std::make_unique<EulerIntegrator<ParticleType>>();
            eulerIntegrator_->eulerStep(particlesHost, particlesCount, dt);
        } else if (method_ == Verlet)
        {
            verletIntegrator_ = std::make_unique<VerletIntegrator<ParticleType>>();
            verletIntegrator_->verletStep(particlesHost, particlesCount, dt);
        }
    }

private:
    std::unique_ptr<EulerIntegrator<ParticleType>> eulerIntegrator_;
    std::unique_ptr<VerletIntegrator<ParticleType>> verletIntegrator_;
    Integrator method_;

    
};


#endif //TIME_INTEGRATOR_H
