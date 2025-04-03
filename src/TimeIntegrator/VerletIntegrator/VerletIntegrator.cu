//
// Created by iqraa on 1-4-25.
//

#include "VerletIntegrator.cuh"
#include <Tools/CudaHelper.hpp>

template<typename ParticleType>
VerletIntegrator<ParticleType>::~VerletIntegrator()
{
    cudaFree(devParticle);
}


template<typename ParticleType>
void VerletIntegrator<ParticleType>::verletStep(ParticleType* particlesHost,size_t particlesCount, const float dt)
{
//    size_t particlesCount = particles.size();
//    ParticleType* particlesHost = particles.data();
    std::cout << "Verlet Integrator() " << particlesCount << " particles\n";


    hostToDevice(particlesHost, particlesCount, &devParticle);

    auto startTime = std::chrono::high_resolution_clock::now();

    // Run Euler Integrator kernel
    verletIntegratorKernel<<<10, threadsPerBlock>>>(devParticle, particlesCount, dt);
    GET_CUDA_ERROR("VerletIntegratorKernelError");

    cudaDeviceSynchronize();
    GET_CUDA_ERROR("VerletIntegratorKernelSyncError");

    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "Verlet Integrator Kernel() duration: " <<
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()) << std::endl;

    deviceToHost(devParticle, particlesCount, particlesHost);
}