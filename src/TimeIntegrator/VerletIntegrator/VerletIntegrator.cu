//
// Created by iqraa on 1-4-25.
//

#include "VerletIntegrator.cuh"
#include <Tools/CudaHelper.hpp>

template<typename ParticleType>
VerletIntegrator<ParticleType>::~VerletIntegrator()
{
    // cudaFree(devParticle);
}


template<typename ParticleType>
void VerletIntegrator<ParticleType>::verletStep(ParticleType* devParticlesPtr_, size_t particlesCount, const float dt)
{
//    size_t particlesCount = particles.size();
//    ParticleType* particlesHost = particles.data();
    std::cout << "- Integrator: Verlet\n";


    // hostToDevice(particlesHost, particlesCount, &devParticle);

    auto startTime = std::chrono::high_resolution_clock::now();

    // Run Euler Integrator kernel
    verletIntegratorKernel<<<10, this->threadsPerBlock>>>(devParticlesPtr_, particlesCount, dt);
    GET_CUDA_ERROR("Verlet Integrator Kernel Error");

    cudaDeviceSynchronize();
    GET_CUDA_ERROR("Verlet Integrator Kernel Sync Error");

    // auto endTime = std::chrono::high_resolution_clock::now();
    // std::cout << "Verlet Integrator Kernel() duration: " <<
        // (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()) << std::endl;

    // deviceToHost(devParticle, particlesCount, particlesHost);
}