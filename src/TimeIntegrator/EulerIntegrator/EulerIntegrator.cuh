//
// Created by iqraa on 27-2-25.
//

#ifndef EULERINTEGRATOR_CUH
#define EULERINTEGRATOR_CUH

#include <Particle/Spherical.hpp>
#include "EulerIntegratorKernel.cuh"
#include <Simulate/Base/Base.cuh>
#include <Tools/CudaHelper.hpp>


template<typename ParticleType>
class EulerIntegrator :public Base
{
public:
  EulerIntegrator() = default;

  ~EulerIntegrator();


  void eulerStep(std::vector<ParticleType>& particles, float dt)
  {
    size_t particlesCount = particles.size();
    ParticleType* particlesHost = particles.data();
    std::cout << "Euler Integrator() " << particlesCount << " particles\n";


    hostToDevice(particlesHost, particlesCount, &devParticle);

    auto startTime = std::chrono::high_resolution_clock::now();

    // Run Euler Integrator kernel
    EulerIntegratorKernel<<<1, threadsPerBlock>>>(devParticle, particlesCount, dt);
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


#endif //EULERINTEGRATOR_CUH
