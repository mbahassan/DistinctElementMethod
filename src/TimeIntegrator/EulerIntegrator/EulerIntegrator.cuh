//
// Created by iqraa on 27-2-25.
//

#ifndef EULERINTEGRATOR_CUH
#define EULERINTEGRATOR_CUH

#include <Particle/Spherical.hpp>
#include <Base/Base.cuh>

class EulerIntegrator :public Base
{
public:
  EulerIntegrator(const Spherical* particle, float dt, int size);

  ~EulerIntegrator();

  void eulerIntegratorKernel() const ;

  private:
  Spherical* devParticle = nullptr;

  int size_;
};


#endif //EULERINTEGRATOR_CUH
