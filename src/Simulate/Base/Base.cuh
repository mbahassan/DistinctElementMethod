//
// Created by iqraa on 21-2-25.
//

#ifndef BASE_CUH
#define BASE_CUH
#include <Particle/Polyhedral.h>
#include <Particle/Spherical.h>


template<typename ParticleType>
class Base
{
public:
    Base();

    void printDeviceInfo() const;

    int threadsPerBlock = 1024;

    int numberOfBlocks = 256;

    ParticleType* devParticlesPtr_ = nullptr;

private:

    cudaDeviceProp deviceProperties{};
};


template class Base<Spherical>;
template class Base<Polyhedral>;

#endif //BASE_CUH
