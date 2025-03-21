//
// Created by iqraa on 27-2-25.
//

#ifndef EULERINTEGRATORKERNEL_CUH
#define EULERINTEGRATORKERNEL_CUH

#include "Particle/Spherical.hpp"
#include "Tools/ArthmiticOperator/MathOperators.hpp"

__global__ void EulerIntegratorKernel(Spherical *particle, const float dt, const int size_)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= size_) return;

  // Update position
  particle[idx].position += particle[idx].velocity * dt;

  // Update velocity
  particle[idx].velocity = particle[idx].force / particle[idx].mass *dt;

  // Update Angular velocity
  particle[idx].angularVel += particle[idx].torque / particle[idx].inertia * dt;
}

#endif //EULERINTEGRATORKERNEL_CUH
