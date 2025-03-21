//
// Created by mbahassan on 2/21/25.
//

#include "GpuClass.cuh"
#include "GpuClassKernel.cuh"
#include <Tools/CudaHelper.hpp>

GpuClass::GpuClass(const Spherical* particle, const int size):
size_(size)
{
    hostToDevice(particle, size_, &devParticle);
}


GpuClass::~GpuClass()
{
    cudaFree(devParticle);
}

void GpuClass::printHellow() const
{
    kernel<<<2,threadsPerBlock>>>(devParticle, size_);
    cudaDeviceSynchronize();

}
