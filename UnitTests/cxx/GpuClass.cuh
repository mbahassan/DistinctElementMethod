//
// Created by mbahassan on 2/21/25.
//

#ifndef GPUCLASS_H
#define GPUCLASS_H

#include <Particle/Spherical.hpp>
#include <Simulate/Base/Base.cuh>

class GpuClass: public Base
{
public:
    GpuClass(const Spherical* particle, int size);

    ~GpuClass();

    void printHellow() const ;

private:
    Spherical* devParticle = nullptr;
    int size_;
};



#endif //GPUCLASS_H
