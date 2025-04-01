//
// Created by iqraa on 25-3-25.
//

#ifndef ARRAY_H
#define ARRAY_H



#include <cmath>
#include <utility>
#include <iterator>
#include <initializer_list>
#include <cuda_runtime.h>

namespace util
{

// Determine alignment based on element type and count.
template<class T>
constexpr int determineAlignment(int n)
{
    return (sizeof(T) * n) % 16 == 0 ? 16 :
           ((sizeof(T) * n) % 8 == 0 ? 8 : alignof(T));
}

/** A std::array-like compile-time fixed-size array usable on host and device.
 */
template<class T, std::size_t N>
struct alignas(determineAlignment<T>(N)) array
{
    using value_type             = T;
    using pointer                = T*;
    using const_pointer          = const T*;
    using reference              = T&;
    using const_reference        = const T&;
    using iterator               = T*;
    using const_iterator         = const T*;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    T data_[N];

    __host__ __device__ constexpr iterator begin() noexcept { return data(); }
    __host__ __device__ constexpr const_iterator begin() const noexcept { return data(); }
    __host__ __device__ constexpr iterator end() noexcept { return data_ + N; }
    __host__ __device__ constexpr const_iterator end() const noexcept { return data_ + N; }
    __host__ __device__ constexpr reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    __host__ __device__ constexpr const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
    __host__ __device__ constexpr reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    __host__ __device__ constexpr const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
    __host__ __device__ constexpr const_iterator cbegin() const noexcept { return data(); }
    __host__ __device__ constexpr const_iterator cend() const noexcept { return data_ + N; }
    __host__ __device__ constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
    __host__ __device__ constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }
    __host__ __device__ constexpr size_type size() const noexcept { return N; }
    __host__ __device__ constexpr size_type max_size() const noexcept { return N; }
    [[nodiscard]] __host__ __device__ constexpr bool empty() const noexcept { return N == 0; }

    // Element access.
    __host__ __device__ constexpr reference operator[](size_type n) noexcept { return data_[n]; }
    __host__ __device__ constexpr const_reference operator[](size_type n) const noexcept { return data_[n]; }
    __host__ __device__ constexpr reference front() noexcept { return data_[0]; }
    __host__ __device__ constexpr const_reference front() const noexcept { return data_[0]; }
    __host__ __device__ constexpr reference back() noexcept { return data_[N - 1]; }
    __host__ __device__ constexpr const_reference back() const noexcept { return data_[N - 1]; }
    __host__ __device__ constexpr pointer data() noexcept { return data_; }
    __host__ __device__ constexpr const_pointer data() const noexcept { return data_; }

    // Assignment and arithmetic operators.
    __host__ __device__ constexpr array<T, N>& operator=(const value_type& rhs) noexcept
    {
        assignImpl(data(), rhs, std::make_index_sequence<N>{});
        return *this;
    }

    __host__ __device__ constexpr array<T, N>& operator+=(const array<T, N>& rhs) noexcept
    {
        assignImpl(data(), rhs.data(), std::make_index_sequence<N>{});
        return *this;
    }

    __host__ __device__ constexpr array<T, N>& operator-=(const array<T, N>& rhs) noexcept
    {
        assignImpl(data(), rhs.data(), std::make_index_sequence<N>{}, [](T a, T b) { return a - b; });
        return *this;
    }

    __host__ __device__ constexpr array<T, N>& operator*=(const value_type& rhs) noexcept
    {
        assignImpl(data(), rhs, std::make_index_sequence<N>{}, [](T a, T b) { return a * b; });
        return *this;
    }

    __host__ __device__ constexpr array<T, N>& operator/=(const value_type& rhs) noexcept
    {
        assignImpl(data(), rhs, std::make_index_sequence<N>{}, [](T a, T b) { return a / b; });
        return *this;
    }

private:
    // Simple assignment implementations.
    template<std::size_t... Is>
    __host__ __device__ constexpr static void assignImpl(T* a, const T& b, std::index_sequence<Is...>) noexcept
    {
        (void)std::initializer_list<int>{ ((a[Is] = b), 0)... };
    }

    template<std::size_t... Is>
    __host__ __device__ constexpr static void assignImpl(T* a, const T* b, std::index_sequence<Is...>) noexcept
    {
        (void)std::initializer_list<int>{ ((a[Is] = b[Is]), 0)... };
    }

    template<std::size_t... Is, class F>
    __host__ __device__ constexpr static void assignImpl(T* a, const T* b, std::index_sequence<Is...>, F f) noexcept
    {
        (void)std::initializer_list<int>{ ((a[Is] = f(a[Is], b[Is])), 0)... };
    }

    template<std::size_t... Is, class F>
    __host__ __device__ constexpr static void assignImpl(T* a, const T& b, std::index_sequence<Is...>, F f) noexcept
    {
        (void)std::initializer_list<int>{ ((a[Is] = f(a[Is], b)), 0)... };
    }
};

// Free functions for structured binding support.
template<std::size_t I, class T, std::size_t N>
__host__ __device__ constexpr T& get(array<T, N>& a) { return a[I]; }

template<std::size_t I, class T, std::size_t N>
__host__ __device__ constexpr const T& get(const array<T, N>& a) { return a[I]; }

template<std::size_t I, class T, std::size_t N>
__host__ __device__ constexpr T&& get(array<T, N>&& a) { return std::move(a[I]); }

template<std::size_t I, class T, std::size_t N>
__host__ __device__ constexpr const T&& get(const array<T, N>&& a) { return std::move(a[I]); }

// Arithmetic operators.
template<class T, std::size_t N>
__host__ __device__ constexpr array<T, N> operator+(const array<T, N>& a, const array<T, N>& b)
{
    array<T, N> ret = a;
    ret += b;
    return ret;
}

namespace detail
{

template<class T, std::size_t... Is>
__host__ __device__ constexpr array<T, sizeof...(Is)> negateImpl(const array<T, sizeof...(Is)>& a,
                                                               std::index_sequence<Is...>)
{
    return { -a[Is]... };
}

template<class T, std::size_t... Is>
__host__ __device__ constexpr bool eqImpl(const T* a, const T* b, std::index_sequence<Is...>)
{
    return ((a[Is] == b[Is]) && ...);
}

template<class T, std::size_t... Is>
__host__ __device__ constexpr T dotImpl(const T* a, const T* b, std::index_sequence<Is...>)
{
    return ((a[Is] * b[Is]) + ...);
}

template<class T, std::size_t... Is>
__host__ __device__ constexpr array<T, sizeof...(Is)> absImpl(const T* a, std::index_sequence<Is...>)
{
    return { std::abs(a[Is])... };
}

template<class T, std::size_t... Is>
__host__ __device__ constexpr array<T, sizeof...(Is)> maxImpl(const T* a, const T* b, std::index_sequence<Is...>)
{
    return { (a[Is] > b[Is] ? a[Is] : b[Is])... };
}

template<class T, std::size_t... Is>
__host__ __device__ constexpr array<T, sizeof...(Is)> minImpl(const T* a, const T* b, std::index_sequence<Is...>)
{
    return { (a[Is] < b[Is] ? a[Is] : b[Is])... };
}

template<int N, int I = 0>
struct LexicographicalCompare
{
    template<class T, class F>
    __host__ __device__ constexpr static bool loop(const T* lhs, const T* rhs, F compare)
    {
        if (compare(lhs[I], rhs[I])) return true;
        if (compare(rhs[I], lhs[I])) return false;
        return LexicographicalCompare<N, I + 1>::loop(lhs, rhs, compare);
    }
};

template<int N>
struct LexicographicalCompare<N, N>
{
    template<class T, class F>
    __host__ __device__ constexpr static bool loop(const T*, const T*, F)
    {
        return false;
    }
};

} // namespace detail

template<class T, std::size_t N>
__host__ __device__ constexpr bool operator==(const array<T, N>& a, const array<T, N>& b)
{
    return detail::eqImpl(a.data(), b.data(), std::make_index_sequence<N>{});
}

template<class T, std::size_t N>
__host__ __device__ constexpr bool operator!=(const array<T, N>& a, const array<T, N>& b)
{
    return !(a == b);
}

template<class T, std::size_t N>
__host__ __device__ constexpr bool operator<(const array<T, N>& a, const array<T, N>& b)
{
    return detail::LexicographicalCompare<N>::loop(a.data(), b.data(), [](T x, T y) { return x < y; });
}

template<class T, std::size_t N>
__host__ __device__ constexpr bool operator>(const array<T, N>& a, const array<T, N>& b)
{
    return detail::LexicographicalCompare<N>::loop(a.data(), b.data(), [](T x, T y) { return x > y; });
}

template<class T, std::size_t N>
__host__ __device__ constexpr T dot(const array<T, N>& a, const array<T, N>& b)
{
    return detail::dotImpl(a.data(), b.data(), std::make_index_sequence<N>{});
}

template<class T, std::size_t N>
__host__ __device__ constexpr T norm2(const array<T, N>& a)
{
    return dot(a, a);
}

template<class T, std::size_t N>
__host__ __device__ constexpr array<T, N> abs(const array<T, N>& a)
{
    return detail::absImpl(a.data(), std::make_index_sequence<N>{});
}

template<class T, std::size_t N>
__host__ __device__ constexpr array<T, N> min(const array<T, N>& a, const array<T, N>& b)
{
    return detail::minImpl(a.data(), b.data(), std::make_index_sequence<N>{});
}

template<class T, std::size_t N>
__host__ __device__ constexpr array<T, N> max(const array<T, N>& a, const array<T, N>& b)
{
    return detail::maxImpl(a.data(), b.data(), std::make_index_sequence<N>{});
}

template<class T, std::size_t N>
__host__ __device__ constexpr T min(const array<T, N>& a)
{
    T ret = a[0];
    for (std::size_t i = 1; i < N; ++i)
    {
        ret = ret < a[i] ? ret : a[i];
    }
    return ret;
}

template<class T, std::size_t N>
__host__ __device__ constexpr T max(const array<T, N>& a)
{
    T ret = a[0];
    for (std::size_t i = 1; i < N; ++i)
    {
        ret = ret > a[i] ? ret : a[i];
    }
    return ret;
}

template<class T>
__host__ __device__ constexpr array<T, 3> cross(const array<T, 3>& a, const array<T, 3>& b)
{
    return { a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0] };
}

template<class T>
constexpr __host__ __device__ array<T, 3> makeVec3(array<T, 4> v)
{
    return { v[0], v[1], v[2] };
}

} // namespace util

// Structured binding support.
namespace std
{
    template<size_t N, class T, size_t N2>
    struct tuple_element<N, util::array<T, N2>>
    {
        using type = T;
    };

    template<class T, size_t N>
    struct tuple_size<util::array<T, N>>
    {
        static const size_t value = N;
    };
} // namespace std


#endif //ARRAY_H
