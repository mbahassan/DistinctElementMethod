//
// Created by iqraa on 1-4-25.
//

#ifndef VERLETINTEGRATOR_H
#define VERLETINTEGRATOR_H

#include <Particle/Polyhedral.h>
#include <Tools/CudaHelper.hpp>
#include <Simulate/Base/Base.cuh>

#include "VerletIntegratorKernel.cuh"


template<typename ParticleType>
class VerletIntegrator: public Base
{
public:
    VerletIntegrator() = default;

    ~VerletIntegrator();

    void verletStep(std::vector<ParticleType>& particles, float dt);

private:
    ParticleType* devParticle = nullptr;
};

template class VerletIntegrator<Spherical>;
template class VerletIntegrator<Polyhedral>;

#endif //VERLETINTEGRATOR_H
