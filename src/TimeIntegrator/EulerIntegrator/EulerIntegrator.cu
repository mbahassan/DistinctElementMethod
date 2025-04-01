//
// Created by iqraa on 27-2-25.
//

#include "EulerIntegrator.cuh"
#include "Tools/CudaHelper.hpp"
#include "EulerIntegratorKernel.cuh"


template<typename ParticleType>
EulerIntegrator<ParticleType>::~EulerIntegrator()
{
  cudaFree(devParticle);
}