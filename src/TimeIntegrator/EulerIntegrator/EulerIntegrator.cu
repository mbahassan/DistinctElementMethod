//
// Created by iqraa on 27-2-25.
//

#include "EulerIntegrator.cuh"
#include "Tools/CudaHelper.hpp"
#include "EulerIntegratorKernel.cuh"

EulerIntegrator::EulerIntegrator(const Particle* particle, float dt, const int size)
{
    hostToDevice(particle, size_, &devParticle);

    EulerIntegratorKernel<<<2, threadsPerBlock>>>(devParticle, dt, size);


}

EulerIntegrator::~EulerIntegrator()
{
  cudaFree(devParticle);
}