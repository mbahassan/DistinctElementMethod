//
// Created by iqraa on 27-2-25.
//

#include "EulerIntegrator.cuh"
#include "EulerIntegratorKernel.cuh"

#include <Tools/CudaHelper.hpp>


template<typename ParticleType>
void EulerIntegrator<ParticleType>::eulerStep(ParticleType* particlesHost, size_t particlesCount, const float dt)
{
    // size_t particlesCount = particles.size();
    // ParticleType* particlesHost = particles.data();
    std::cout << "\nEuler Integrator: \n";


    hostToDevice(particlesHost, particlesCount, &devParticle);

    auto startTime = std::chrono::high_resolution_clock::now();

    // Run Euler Integrator kernel
    eulerIntegratorKernel<ParticleType> <<<10, threadsPerBlock>>>(devParticle, particlesCount, dt);
    GET_CUDA_ERROR("EulerIntegratorKernelError");

    cudaDeviceSynchronize();
    GET_CUDA_ERROR("EulerIntegratorKernelSyncError");

    deviceToHost(devParticle, particlesCount, particlesHost);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "- Euler Kernel() duration: " <<
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()) << std::endl;
}
