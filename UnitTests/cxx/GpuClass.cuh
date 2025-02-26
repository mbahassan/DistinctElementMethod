//
// Created by mbahassan on 2/21/25.
//

#ifndef GPUCLASS_H
#define GPUCLASS_H

#include <Particle/Particle.hpp>
#include <Base/Base.cuh>

class GpuClass: public Base
{
public:
    GpuClass(const Particle* particle, int size);
    ~GpuClass();

    void printHellow() const ;

private:
    Particle* devParticle = nullptr;
    int size_;
};



#endif //GPUCLASS_H
