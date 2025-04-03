//
// Created by iqraa on 27-2-25.
//

#ifndef EULER_INTEGRATOR_H
#define EULER_INTEGRATOR_H

#include <Particle/Polyhedral.h>
#include <Particle/Spherical.h>
#include <Simulate/Base/Base.cuh>


template<typename ParticleType>
class EulerIntegrator: public Base
{
public:
  EulerIntegrator() = default;

  ~EulerIntegrator()
  {
    cudaFree(devParticle);
  }

  void eulerStep(std::vector<ParticleType>& particles, float dt);


private:
  ParticleType* devParticle = nullptr;

};


template class EulerIntegrator<Spherical>;
template class EulerIntegrator<Polyhedral>;

#endif //EULER_INTEGRATOR_H
