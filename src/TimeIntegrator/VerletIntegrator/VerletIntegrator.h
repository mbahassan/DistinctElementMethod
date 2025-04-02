//
// Created by iqraa on 1-4-25.
//

#ifndef VERLETINTEGRATOR_H
#define VERLETINTEGRATOR_H

#include <Particle/Polyhedral.h>
#include <Tools/CudaHelper.hpp>
#include <Simulate/Base/Base.cuh>

#include "VerletIntegratorKernel.cu"


template<typename ParticleType>
class VerletIntegrator: public Base
{
public:
    VerletIntegrator() = default;

    ~VerletIntegrator();

    void verletStep(std::vector<ParticleType>& particles, const float dt)
    {
        size_t particlesCount = particles.size();
        ParticleType* particlesHost = particles.data();
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

    }
private:
    ParticleType* devParticle = nullptr;
};

template class VerletIntegrator<Spherical>;
template class VerletIntegrator<Polyhedral>;

#endif //VERLETINTEGRATOR_H
