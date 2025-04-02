//
// Created by iqraa on 27-2-25.
//

#ifndef EULER_INTEGRATOR_H
#define EULER_INTEGRATOR_H

#include <Particle/Polyhedral.h>
#include <Particle/Spherical.h>
#include <Tools/CudaHelper.hpp>
#include <Simulate/Base/Base.cuh>
#include "EulerIntegratorKernel.cu"

template<typename ParticleType>
class EulerIntegrator: public Base
{
public:
  EulerIntegrator() = default;

  ~EulerIntegrator();


  void eulerStep(std::vector<ParticleType>& particles, const float dt)
  {
    size_t particlesCount = particles.size();
    ParticleType* particlesHost = particles.data();
    std::cout << "Euler Integrator() " << particlesCount << " particles\n";


    hostToDevice(particlesHost, particlesCount, &devParticle);

    auto startTime = std::chrono::high_resolution_clock::now();

    // Run Euler Integrator kernel
    eulerIntegratorKernel<ParticleType> <<<10, threadsPerBlock>>>(devParticle, particlesCount, dt);
    GET_CUDA_ERROR("EulerIntegratorKernelError");

    cudaDeviceSynchronize();
    GET_CUDA_ERROR("EulerIntegratorKernelSyncError");

    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "Euler Integrator Kernel() duration: " <<
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()) << std::endl;
  }



private:
  ParticleType* devParticle = nullptr;

};


template class EulerIntegrator<Spherical>;
template class EulerIntegrator<Polyhedral>;

#endif //EULER_INTEGRATOR_H
