//
// Created by iqraa on 27-2-25.
//

#ifndef ARTHMITICOPERATORS_H
#define ARTHMITICOPERATORS_H


#include <cuda_runtime.h>

// Addition
__host__ __device__ inline float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

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
__host__ __device__ inline float2 operator-(const float2& a, const float2& b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

// Multiplication by scalar
__host__ __device__ inline float2 operator*(const float2& a, float b) {
    return make_float2(a.x * b, a.y * b);
}

__host__ __device__ inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

// Division by scalar
__host__ __device__ inline float2 operator/(const float2& a, float b) {
    return make_float2(a.x / b, a.y / b);
}

__host__ __device__ inline float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}


// Compound assignment operators
__host__ __device__ inline float2& operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__host__ __device__ inline float2& operator-=(float2& a, const float2& b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

__host__ __device__ inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__host__ __device__ inline float2& operator*=(float2& a, float b) {
    a.x *= b;
    a.y *= b;
    return a;
}

__host__ __device__ inline float3& operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

__host__ __device__ inline float2& operator/=(float2& a, float b) {
    a.x /= b;
    a.y /= b;
    return a;
}

__host__ __device__ inline float3& operator/=(float3& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}

// Unary minus operator
__host__ __device__ inline float2 operator-(const float2& a) {
    return make_float2(-a.x, -a.y);
}

__host__ __device__ inline float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}



#endif //ARTHMITICOPERATORS_H
