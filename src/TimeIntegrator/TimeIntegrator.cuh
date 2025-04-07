//
// Created by iqraa on 1-4-25.
//

#ifndef TIME_INTEGRATOR_H
#define TIME_INTEGRATOR_H

#include <Tools/CudaHelper.hpp>

#include "EulerIntegrator/EulerIntegrator.cuh"
#include "VerletIntegrator/VerletIntegrator.cuh"

enum Integrator
{
    Euler,
    Verlet
};


template<typename ParticleType>
class TimeIntegrator : public EulerIntegrator<ParticleType>, public VerletIntegrator<ParticleType>
{
public:
    explicit TimeIntegrator(const Integrator method = Euler): Base<ParticleType>(),
                                                              method_(method) {
    }

    void integrate(std::vector<ParticleType>& particles, const double dt)
    {
        // Time the integration
        startTime = std::chrono::high_resolution_clock::now();

        size_t particlesCount = particles.size();
        ParticleType* particlesHost = particles.data();

        // Copy particles vector to devParticles for the kernel
        hostToDevice(particlesHost, particlesCount, &devParticlesPtr_);

        if (method_ == Euler)
        {
            this->eulerStep(devParticlesPtr_, particlesCount, dt);
        } else if (method_ == Verlet)
        {
            this->verletStep(devParticlesPtr_, particlesCount, dt);
        }

        // copy back to host
        deviceToHost(devParticlesPtr_, particlesCount, particlesHost);

        endTime = std::chrono::high_resolution_clock::now();
        std::cout << "- Integrator Kernel() duration: " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;
    }

    ~TimeIntegrator()
    {
        cudaFree(devParticlesPtr_);
    }

private:

    std::chrono::high_resolution_clock::time_point startTime ;

    std::chrono::high_resolution_clock::time_point endTime ;

    ParticleType* devParticlesPtr_ = nullptr;

    Integrator method_;
};


#endif //TIME_INTEGRATOR_H
