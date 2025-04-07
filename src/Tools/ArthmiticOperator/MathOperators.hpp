//
// Created by iqraa on 27-2-25.
//

#ifndef ARTHMITICOPERATORS_H
#define ARTHMITICOPERATORS_H


#include <cuda_runtime.h>
#include <cmath>

// Addition
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3& operator+=(float3& lhs, const float3& rhs) {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

__host__ __device__ inline float3 operator+(const float3& a, const float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

// Subtraction
__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

// Multiplication by scalar
__host__ __device__ inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

// Division by scalar
__host__ __device__ inline float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

// cross production operator
__host__ __device__ inline float3 operator^(const float3& a, const float3 b) {
    return make_float3
    (
        a.y*b.z - a.z*b.y,  // x-component
        a.z*b.x - a.x*b.z,  // y-component
        a.x*b.y - a.y*b.x   // z-component
    );
}


// Compound assignment operators
__host__ __device__ inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__host__ __device__ inline float3& operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

__host__ __device__ inline float3& operator/=(float3& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}

// Unary minus operator
__host__ __device__ inline float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

// dot product
__host__ __device__ inline float operator&(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float magSquared(const float3& a) {
    return a&a;
}

// loginc operators
// Less than operator for float3

__host__ __device__ inline bool operator<(const float3& a, const float3& b) {
    return magSquared(a) < magSquared(b);
}

// Greater than operator for float3
__host__ __device__ inline bool operator>(const float3& a, const float3& b) {
    return magSquared(a) > magSquared(b);
}

// Less than or equal to operator for float3
__host__ __device__ inline bool operator<=(const float3& a, const float3& b) {
    return magSquared(a) <= magSquared(b);
}

// Greater than or equal to operator for float3
__host__ __device__ inline bool operator>=(const float3& a, const float3& b) {
    return magSquared(a) >= magSquared(b);
}

__host__ __device__ inline float mag(const float3& a) {
    return sqrtf(magSquared(a));
}

__host__ __device__ inline float3 normalize(const float3& vec) {
    float magnit = mag(vec);
    if (magnit == 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return {vec.x / magnit, vec.y / magnit, vec.z / magnit};
}
#endif //ARTHMITICOPERATORS_H
