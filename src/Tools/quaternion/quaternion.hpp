//
// Created by iqraa on 15-2-25.
//

#ifndef QUATERNION_H
#define QUATERNION_H

#include <cmath>
#include <cuda_runtime.h>


// Add this struct for 3x3 matrix operations
struct Matrix3x3 {
    float m[3][3];

    Matrix3x3() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    // Set a specific element
    void set(int i, int j, float value) {
        m[i][j] = value;
    }

    // Get element
    float get(int i, int j) const {
        return m[i][j];
    }

    // Matrix determinant
    float determinant() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
};


class Quaternion {
public:
    float w, x, y, z;

    // Constructors
    __host__ __device__ Quaternion() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}

    __host__ __device__ Quaternion(float w_, float x_, float y_, float z_)
        : w(w_), x(x_), y(y_), z(z_) {}

    // Copy constructor
    __host__ __device__ Quaternion(const Quaternion& q)
        : w(q.w), x(q.x), y(q.y), z(q.z) {}

    // Identity Quaternion
    __host__ __device__ static Quaternion identity()
    {
        return {1.0f, 0.0f, 0.0f, 0.0f};
    }

    // Create from axis angle
    __host__ __device__ static Quaternion fromAxisAngle(float3 axis, float angle)
    {
        float halfAngle = angle * 0.5f;
        float s = sinf(halfAngle);
        float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

        if (length > 0.0f)
        {
            s /= length;
        }

        return {cosf(halfAngle),
                         axis.x * s,
                         axis.y * s,
                         axis.z * s};
    }

    // Basic operations
    __host__ __device__ Quaternion& operator=(const Quaternion& q)
    {
        w = q.w; x = q.x; y = q.y; z = q.z;
        return *this;
    }

    __host__ __device__ Quaternion operator*(const Quaternion& q) const
    {
        return {
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
        };
    }

    __host__ __device__ float3 rotateVector(const float3& v) const {
        // Convert vector to Quaternion
        Quaternion p(0.0f, v.x, v.y, v.z);

        // Perform rotation: q * p * q^(-1)
        Quaternion qInv = conjugate();
        Quaternion rotated = *this * p * qInv;

        return make_float3(rotated.x, rotated.y, rotated.z);
    }

    // Utility functions
    __host__ __device__ float norm() const
    {
        return sqrtf(w * w + x * x + y * y + z * z);
    }

    __host__ __device__ void normalize()
    {
        float n = norm();
        if (n > 0.0f) {
            float invN = 1.0f / n;
            w *= invN;
            x *= invN;
            y *= invN;
            z *= invN;
        }
    }

    __host__ __device__ Quaternion normalized() const
    {
        Quaternion q = *this;
        q.normalize();
        return q;
    }

    __host__ __device__ Quaternion conjugate() const
    {
        return {w, -x, -y, -z};
    }

    __host__ __device__ Quaternion inverse() const
    {
        float n = norm();
        n = n * n;
        if (n > 0.0f) {
            float invN = 1.0f / n;
            return Quaternion(w * invN, -x * invN, -y * invN, -z * invN);
        }
        return *this;
    }

    // Conversion to/from Euler angles (in radians)
    __host__ __device__ static Quaternion fromEuler(float3 euler)
    {
        float cx = cosf(euler.x * 0.5f);
        float cy = cosf(euler.y * 0.5f);
        float cz = cosf(euler.z * 0.5f);
        float sx = sinf(euler.x * 0.5f);
        float sy = sinf(euler.y * 0.5f);
        float sz = sinf(euler.z * 0.5f);

        return Quaternion(
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz
        );
    }

    __host__ __device__ float3 toEuler() const
    {
        float3 euler;

        // Roll (x-axis rotation)
        float sinr_cosp = 2.0f * (w * x + y * z);
        float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
        euler.x = atan2f(sinr_cosp, cosr_cosp);

        // Pitch (y-axis rotation)
        float sinp = 2.0f * (w * y - z * x);
        euler.y = fabsf(sinp) >= 1.0f ?
            copysignf(M_PI / 2.0f, sinp) : asinf(sinp);

        // Yaw (z-axis rotation)
        float siny_cosp = 2.0f * (w * z + x * y);
        float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
        euler.z = atan2f(siny_cosp, cosy_cosp);

        return euler;
    }

    // Spherical linear interpolation
    __host__ __device__ static Quaternion slerp(const Quaternion& q1,
                                               const Quaternion& q2,
                                               float t) {
        float cosHalfTheta = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;

        if (fabsf(cosHalfTheta) >= 1.0f)
        {
            return q1;
        }

        float halfTheta = acosf(cosHalfTheta);
        float sinHalfTheta = sqrtf(1.0f - cosHalfTheta * cosHalfTheta);

        if (fabsf(sinHalfTheta) < 0.001f)
        {
            return Quaternion(
                q1.w * 0.5f + q2.w * 0.5f,
                q1.x * 0.5f + q2.x * 0.5f,
                q1.y * 0.5f + q2.y * 0.5f,
                q1.z * 0.5f + q2.z * 0.5f
            );
        }

        float ratioA = sinf((1.0f - t) * halfTheta) / sinHalfTheta;
        float ratioB = sinf(t * halfTheta) / sinHalfTheta;

        return Quaternion(
            q1.w * ratioA + q2.w * ratioB,
            q1.x * ratioA + q2.x * ratioB,
            q1.y * ratioA + q2.y * ratioB,
            q1.z * ratioA + q2.z * ratioB
        );
    }

    // For printing
    friend std::ostream& operator<<(std::ostream& os, const Quaternion& q) {
        os << q.w << " + " << q.x << "i + " << q.y << "j + " << q.z << "k";
        return os;
    }
};


#endif //QUATERNION_H
