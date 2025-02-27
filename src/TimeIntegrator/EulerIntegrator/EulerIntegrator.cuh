//
// Created by iqraa on 27-2-25.
//

#ifndef EULERINTEGRATOR_CUH
#define EULERINTEGRATOR_CUH

#include <Particle/Particle.hpp>
#include <Base/Base.cuh>

class EulerIntegrator :public Base
{
public:
  EulerIntegrator(const Particle* particle, float dt, int size);

  ~EulerIntegrator();

  void eulerIntegratorKernel() const ;

  private:
  Particle* devParticle = nullptr;

  int size_;
};



#endif //EULERINTEGRATOR_CUH
