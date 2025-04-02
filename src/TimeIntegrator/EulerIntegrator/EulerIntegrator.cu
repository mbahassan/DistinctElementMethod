//
// Created by iqraa on 27-2-25.
//

#include "EulerIntegrator.h"


template<typename ParticleType>
EulerIntegrator<ParticleType>::~EulerIntegrator()
{
  cudaFree(devParticle);
}