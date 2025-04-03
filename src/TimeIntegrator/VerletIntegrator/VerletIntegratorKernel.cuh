//
// Created by iqraa on 27-2-25.
//

#ifndef VERLET_INTEGRATOR_KERNEL_CUH
#define VERLET_INTEGRATOR_KERNEL_CUH

#include <device_launch_parameters.h>

#include "Particle/Spherical.h"
#include "Tools/ArthmiticOperator/MathOperators.hpp"

template<typename ParticleType>
__global__ void verletIntegratorKernel(ParticleType *particle, const int size_, const float dt)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= size_) return;

  // Update position
  particle[idx].position += particle[idx].velocity * dt;

  // Update velocity
  // particle[idx].velocity = particle[idx].force / particle[idx].mass *dt;

  // Update Angular velocity
  // particle[idx].angularVel += particle[idx].torque / particle[idx].inertia * dt;
}

#endif //VERLET_INTEGRATOR_KERNEL_CUH
