#include "Spherical.hpp"

#include <cmath>
#include <Tools/ArthmiticOperator/MathOperators.hpp>


Spherical::Spherical(const Material& material, const Sphere& shape):
Material(material), Sphere(shape) {
    boundingBox.min = position - Sphere::getMin();
    boundingBox.max = position + Sphere::getMax();
}


Spherical::Spherical(const Material& material, Sphere& shape):
Material(material), Sphere(shape)
{
    boundingBox.min = position - Sphere::getMin();
    boundingBox.max = position + Sphere::getMax();
}


float3 Spherical::getAxisDirection() const
{
    // The default axis direction is typically along the z-axis (0,0,1)
    float3 defaultAxis = make_float3(0.0f, 0.0f, 1.0f);

    // Convert quaternion to rotation matrix
    // Using quaternion components (w,x,y,z)
    float w = orientation.w;
    float x = orientation.x;
    float y = orientation.y;
    float z = orientation.z;

    // Apply quaternion rotation to default axis
    // This is an optimized version of the quaternion rotation formula
    // for rotating a vector, specifically optimized for (0,0,1)
    float3 rotatedAxis;
    rotatedAxis.x = 2.0f * (x*z + w*y);
    rotatedAxis.y = 2.0f * (y*z - w*x);
    rotatedAxis.z = 1.0f - 2.0f * (x*x + y*y);

    // Normalize the result to ensure we have a unit vector
    float length = sqrtf(rotatedAxis.x * rotatedAxis.x +
                        rotatedAxis.y * rotatedAxis.y +
                        rotatedAxis.z * rotatedAxis.z);

    if (length > 0.0f)
    {
        rotatedAxis.x /= length;
        rotatedAxis.y /= length;
        rotatedAxis.z /= length;
    }

    return rotatedAxis;
}


