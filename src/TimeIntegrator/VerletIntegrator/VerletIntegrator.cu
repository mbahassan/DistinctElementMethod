//
// Created by iqraa on 1-4-25.
//

#include "VerletIntegrator.h"


template<typename ParticleType>
VerletIntegrator<ParticleType>::~VerletIntegrator()
{
    cudaFree(devParticle);
}