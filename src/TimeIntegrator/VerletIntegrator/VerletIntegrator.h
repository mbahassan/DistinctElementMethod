//
// Created by iqraa on 1-4-25.
//

#ifndef VERLETINTEGRATOR_H
#define VERLETINTEGRATOR_H

#include <Tools/CudaHelper.hpp>
#include <Simulate/Base/Base.cuh>

#include "VerletIntegratorKernel.cuh"


template<typename ParticleType>
class VerletIntegrator: public Base
{
public:
    VerletIntegrator() = default;

    ~VerletIntegrator();

    void verletStep(std::vector<ParticleType>& particles, float dt)
    {
        size_t particlesCount = particles.size();
        ParticleType* particlesHost = particles.data();
        std::cout << "Euler Integrator() " << particlesCount << " particles\n";


        hostToDevice(particlesHost, particlesCount, &devParticle);

        auto startTime = std::chrono::high_resolution_clock::now();

        // Run Euler Integrator kernel
        VerletIntegratorKernel<<<1, threadsPerBlock>>>(devParticle, particlesCount, dt);
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



#endif //VERLETINTEGRATOR_H
