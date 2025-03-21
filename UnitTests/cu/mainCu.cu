#include <iostream>
#include <vector>
#include "Tools/CudaHelper.hpp"
#include <Particle/Particle.hpp>

__global__ void kernel( Spherical* particle, const int N)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        particle[idx].setId(idx);
        // printf("i = %d\n",idx);
    }
}

int main() {
    bool active = true;
    int N = 5;
    std::vector<Spherical> particles(N);

    for (int i = 0; i < N; i++) {
        particles[i].setId(i);
    }

    auto lang1 = "C++";
    std::cout << "Hello and welcome to " << lang1 << "!\n";

    for (int i = 0; i < N; i++) {

        std::cout << "i = " << i << " Id: " <<particles[i].getId() << std::endl;
    }

    Spherical* devParticles = nullptr;
    hostToDevice(particles.data(), N, &devParticles);

    kernel<<<1, N>>>(devParticles, N);
    cudaDeviceSynchronize();
    Spherical* hostParticles = nullptr;
    deviceToHost(devParticles, N, &hostParticles);

    for (int i = 0; i < N; i++) {

        std::cout << "i = " << i << " Id: " <<hostParticles[i].getId() << std::endl;
    }

    cudaFree(devParticles);
    free(hostParticles);

    cudaDeviceReset();

    return 0;
}

