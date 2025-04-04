//
// Created by iqraa on 14-2-25.
//

#ifndef CUDAHELPER_H
#define CUDAHELPER_H


#include <cuda_runtime_api.h>

void getLastCudaError(const char *errorMessage, const char *file, int line);
#define GET_CUDA_ERROR(msg) getLastCudaError(msg, __FILE__, __LINE__);

template<typename T>
void deviceToHost(const T* devPtr, const size_t size, T** hostPtr)
{
    *hostPtr = (T*)malloc(size * sizeof(T));

    cudaMemcpy(*hostPtr, devPtr, size * sizeof(T), cudaMemcpyDeviceToHost);
    GET_CUDA_ERROR("cudaMemcpy() deviceToHost");
    cudaDeviceSynchronize();
    GET_CUDA_ERROR("cudaDeviceSynchronize() deviceToHost");
}

template<typename T>
void hostToDevice(const T* hostPtr, const size_t size, T** devPtr)
{
    cudaMalloc((void**)devPtr, size * sizeof(T));

    cudaMemcpy(*devPtr, hostPtr, size * sizeof(T), cudaMemcpyHostToDevice);
    GET_CUDA_ERROR("cudaMemcpy() hostToDevice");
    cudaDeviceSynchronize();
    GET_CUDA_ERROR("cudaDeviceSynchronize() hostToDevice");
}

template<typename T>
void deviceToHost(const T* devPtr,  const size_t size, T* hostPtr)
{
    cudaMemcpy(hostPtr, devPtr, size * sizeof(T), cudaMemcpyDeviceToHost);
    GET_CUDA_ERROR("cudaMemcpy() deviceToHost");
    cudaDeviceSynchronize();
    GET_CUDA_ERROR("cudaDeviceSynchronize() deviceToHost");
}

#endif //CUDAHELPER_H
