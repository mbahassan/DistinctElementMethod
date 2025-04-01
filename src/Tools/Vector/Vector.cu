//
// Created by iqraa on 25-3-25.
//

#include "Vector.h"

// VectorD.cu (or .cpp)
// Implementation of Vector using Thrust for CUDA
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
//#include "cstone/cuda/errorcheck.cuh"
#include "Tools/Thrust/NoInitThrust.cuh"
#include "Tools/CudaHelper.hpp"
#include <cuda_runtime.h>


#define GPU_SYMBOL(x) x


template<class T>
class Vector<T>::Impl
{
public:
    Impl() = default;

    T* data() { return thrust::raw_pointer_cast(data_.data()); }
    const T* data() const { return thrust::raw_pointer_cast(data_.data()); }

    void resize(std::size_t size) { data_.resize(size); }
    void reserve(std::size_t size) { data_.reserve(size); }
    void push_back(const T& value) { data_.push_back(value); }
    std::size_t size() const { return data_.size(); }
    std::size_t capacity() const { return data_.capacity(); }

    Impl& operator=(const std::vector<T>& rhs)
    {
        data_ = rhs;
        return *this;
    }

private:
    friend bool operator==(const Impl& lhs, const Impl& rhs)
    {
        return lhs.data_ == rhs.data_;
    }
    thrust::device_vector<T, uninitialized_allocator<T>> data_;
};

template<class T>
Vector<T>::Vector() : impl_(new Impl()) {}

template<class T>
Vector<T>::Vector(std::size_t size) : impl_(new Impl())
{
    impl_->resize(size);
}

template<class T>
Vector<T>::Vector(std::size_t size, T init) : impl_(new Impl())
{
    impl_->resize(size);
    thrust::fill(thrust::device, impl_->data(), impl_->data() + impl_->size(), init);
}

template<class T>
Vector<T>::Vector(const Vector<T>& other) : impl_(new Impl())
{
    *impl_ = *other.impl_;
}

template<class T>
Vector<T>::Vector(const std::vector<T>& rhs) : impl_(new Impl())
{
    *impl_ = rhs;
}

template<class T>
Vector<T>::Vector(const T* first, const T* last) : impl_(new Impl())
{
    auto size = last - first;
    impl_->resize(size);
    checkGpuErrors(cudaMemcpy(impl_->data(), first, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
Vector<T>::~Vector() = default;

template<class T>
T* Vector<T>::data() { return impl_->data(); }

template<class T>
const T* Vector<T>::data() const { return impl_->data(); }

template<class T>
void Vector<T>::resize(std::size_t size) { impl_->resize(size); }

template<class T>
void Vector<T>::reserve(std::size_t size) { impl_->reserve(size); }

template<class T>
void Vector<T>::push_back(const T& value) { impl_->push_back(value); }

template<class T>
std::size_t Vector<T>::size() const { return impl_->size(); }

template<class T>
bool Vector<T>::empty() const { return impl_->size() == 0; }

template<class T>
std::size_t Vector<T>::capacity() const { return impl_->capacity(); }

template<class T>
Vector<T>& Vector<T>::swap(Vector<T>& rhs)
{
    std::swap(impl_, rhs.impl_);
    return *this;
}

template<class T>
Vector<T>& Vector<T>::operator=(Vector<T> rhs)
{
    swap(rhs);
    return *this;
}

template<class T>
Vector<T>& Vector<T>::operator=(const std::vector<T>& rhs)
{
    *impl_ = rhs;
    return *this;
}

template<class T>
bool operator==(const Vector<T>& lhs, const Vector<T>& rhs)
{
    return *lhs.impl_ == *rhs.impl_;
}

#define DEVICE_VECTOR(T)                                    \
    template class Vector<T>;                         \
    template bool operator==(const Vector<T>&, const Vector<T>&);

DEVICE_VECTOR(char);
DEVICE_VECTOR(uint8_t);
DEVICE_VECTOR(int);
DEVICE_VECTOR(unsigned);
DEVICE_VECTOR(uint64_t);
DEVICE_VECTOR(float);
DEVICE_VECTOR(double);

template class Vector<util::array<int, 2>>;
template class Vector<util::array<int, 3>>;
template class Vector<util::array<unsigned, 1>>;
template class Vector<util::array<uint64_t, 1>>;
template class Vector<util::array<uint64_t, 2>>;
template class Vector<util::array<unsigned, 2>>;
template class Vector<util::array<float, 3>>;
template class Vector<util::array<float, 4>>;
template class Vector<util::array<float, 8>>;
template class Vector<util::array<float, 12>>;
template class Vector<util::array<double, 3>>;
template class Vector<util::array<double, 4>>;
template class Vector<util::array<double, 8>>;
template class Vector<util::array<double, 12>>;


