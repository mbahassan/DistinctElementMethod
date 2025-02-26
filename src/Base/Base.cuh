//
// Created by iqraa on 21-2-25.
//

#ifndef BASE_CUH
#define BASE_CUH



class Base {
public:
    Base();

    void printDeviceInfo() const;

    int threadsPerBlock = 1024;
    int numberOfBlocks = 256;
private:
    cudaDeviceProp deviceProperties{};
};



#endif //BASE_CUH
