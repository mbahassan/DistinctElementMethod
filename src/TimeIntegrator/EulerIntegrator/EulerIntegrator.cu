//
// Created by iqraa on 27-2-25.
//

#include "EulerIntegrator.cuh"
#include "EulerIntegratorKernel.cuh"

#include <Tools/CudaHelper.hpp>


template<typename ParticleType>
void EulerIntegrator<ParticleType>::eulerStep(ParticleType* devParticlesPtr_,size_t  particlesCount, const float dt)
{
    // size_t particlesCount = particles.size();
    // ParticleType* particlesHost = particles.data();
    std::cout << "\n- Integrator: Euler\n";

    // ParticleType* devParticle;
    // hostToDevice(particlesHost, particlesCount, &devParticle);

    int blocksPerGrid = (particlesCount + this->threadsPerBlock - 1) / this->threadsPerBlock;

    // Run Euler Integrator kernel
    eulerIntegratorKernel<ParticleType> <<<blocksPerGrid, this->threadsPerBlock>>>(devParticlesPtr_, particlesCount, dt);
    GET_CUDA_ERROR("EulerIntegratorKernelError");

    cudaDeviceSynchronize();
    GET_CUDA_ERROR("EulerIntegratorKernelSyncError");

    // deviceToHost(devParticle, particlesCount, particles);

}
