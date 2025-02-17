#include <iostream>
#include <vector>
#include "Tools/CudaHelper.hpp"
#include "Particle/Particle.hpp"

__global__ void kernel(const Particle* particle, const int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        printf("i = %d, Particle id: %d\n",idx, particle[idx].setId(idx));
    }
}

int main() {
    int N = 5;
    std::pmr::vector<Particle> particles(N);

    for (int i = 0; i < N; i++) {
        particles[i].setId(i);
    }

    auto lang1 = "C++";
    std::cout << "Hello and welcome to " << lang1 << "!\n";

    // for (int i = 1; i <= N; i++) {
    //
    //     std::cout << "i = " << i << std::endl;
    // }
    Particle* devParticles = nullptr;
    hostToDevice(particles.data(), N, &devParticles);

    kernel<<<1, N>>>(devParticles, N);
    cudaDeviceSynchronize();

    return 0;
}

