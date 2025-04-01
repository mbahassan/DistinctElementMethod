//
// Created by iqraa on 25-3-25.
//

#ifndef VECTOR_CUH
#define VECTOR_CUH


#include <memory>
#include <vector>
#include "../Array/Array.h"
#include <Particle/Spherical.hpp>

/** Encapsulation for a GPU-resident device vector with a minimal interface.
 *  Hides any GPU/CUDA headers from CPU code.
 */
template<class T>
class Vector
{
public:
    using value_type = T;

    Vector();
    Vector(std::size_t size);
    Vector(std::size_t size, T init);
    Vector(const Vector&);
    Vector(const std::vector<T>&);    // Upload from host vector
    Vector(const T* first, const T* last); // Upload from host pointers

    ~Vector();

    T* data();
    const T* data() const;

    void resize(std::size_t size);
    void reserve(std::size_t size);
    void push_back(const T& value);

    std::size_t size() const;
    bool empty() const;
    std::size_t capacity() const;

    Vector& swap(Vector<T>& rhs);
    Vector& operator=(const std::vector<T>& rhs);
    Vector& operator=(Vector<T> rhs);

private:
    friend void swap(Vector& lhs, Vector& rhs) { lhs.swap(rhs); }
    template<class S>
    friend bool operator==(const Vector<S>& lhs, const Vector<S>& rhs);

    class Impl;
    std::unique_ptr<Impl> impl_;
};

template<class T>
T* rawPtr(Vector<T>& p) { return p.data(); }

template<class T>
const T* rawPtr(const Vector<T>& p) { return p.data(); }

// Explicit instantiations
extern template class Vector<char>;
extern template class Vector<uint8_t>;
extern template class Vector<int>;
extern template class Vector<unsigned>;
extern template class Vector<uint64_t>;
extern template class Vector<float>;
extern template class Vector<double>;
extern template class Vector<util::array<int, 2>>;
extern template class Vector<util::array<int, 3>>;
extern template class Vector<util::array<unsigned, 1>>;
extern template class Vector<util::array<unsigned, 2>>;
extern template class Vector<util::array<float, 3>>;
extern template class Vector<util::array<double, 3>>;
extern template class Vector<util::array<float, 4>>;
extern template class Vector<util::array<double, 4>>;

template class Vector<float3>;
template class Vector<Spherical>;




#endif //VECTOR_CUH
