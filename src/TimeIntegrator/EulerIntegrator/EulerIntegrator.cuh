//
// Created by iqraa on 27-2-25.
//

#ifndef EULER_INTEGRATOR_H
#define EULER_INTEGRATOR_H

#include <Particle/Polyhedral.h>
#include <Particle/Spherical.h>
#include <Simulate/Base/Base.cuh>


template<typename ParticleType>
class EulerIntegrator: public virtual Base<ParticleType>
{
public:
  EulerIntegrator() = default;

  ~EulerIntegrator()
  {
    // cudaFree(this->devParticlesPtr_);
  }

  void eulerStep(ParticleType* devParticlesPtr_, size_t particlesCount,  float dt);

};


template class EulerIntegrator<Spherical>;
template class EulerIntegrator<Polyhedral>;

#endif //EULER_INTEGRATOR_H
