//
// Created by iqraa on 1-4-25.
//

#ifndef VERLETINTEGRATOR_H
#define VERLETINTEGRATOR_H

#include <Particle/Polyhedral.h>
#include <Simulate/Base/Base.cuh>

#include "VerletIntegratorKernel.cuh"


template<typename ParticleType>
class VerletIntegrator: public virtual Base<ParticleType>
{
public:
    VerletIntegrator() = default;

    ~VerletIntegrator();

    void verletStep(ParticleType* devParticlesPtr_, size_t particlesCount, float dt);

};

template class VerletIntegrator<Spherical>;
template class VerletIntegrator<Polyhedral>;

#endif //VERLETINTEGRATOR_H
