#include "Polyhedral.hpp"

#include <cmath>
#include <Tools/ArthmiticOperator/MathOperators.hpp>


// Polyhedral::Polyhedral(const Material& material, const Polytope& polytope):
// Polytope(polytope), Material(material) {
//     boundingBox.min = position - Polytope::getMin();
//     boundingBox.max = position + Polytope::getMax();
// }


// Polyhedral::Polyhedral(Material& material, Polytope& polytope):
// Polytope(polytope), Material(material)
// {
//     boundingBox.min = position + Polytope::getMin();
//     boundingBox.max = position + Polytope::getMax();
// }


// float3 Polyhedral::getAxisDirection() const
// {
//     // The default axis direction is typically along the z-axis (0,0,1)
//     float3 defaultAxis = make_float3(0.0f, 0.0f, 1.0f);
//
//     // Convert quaternion to rotation matrix
//     // Using quaternion components (w,x,y,z)
//     float w = getOrientation().w;
//     float x = getOrientation().x;
//     float y = getOrientation().y;
//     float z = getOrientation().z;
//
//     // Apply quaternion rotation to default axis
//     // This is an optimized version of the quaternion rotation formula
//     // for rotating a vector, specifically optimized for (0,0,1)
//     float3 rotatedAxis;
//     rotatedAxis.x = 2.0f * (x*z + w*y);
//     rotatedAxis.y = 2.0f * (y*z - w*x);
//     rotatedAxis.z = 1.0f - 2.0f * (x*x + y*y);
//
//     // Normalize the result to ensure we have a unit vector
//     float length = sqrtf(rotatedAxis.x * rotatedAxis.x +
//                         rotatedAxis.y * rotatedAxis.y +
//                         rotatedAxis.z * rotatedAxis.z);
//
//     if (length > 0.0f)
//     {
//         rotatedAxis.x /= length;
//         rotatedAxis.y /= length;
//         rotatedAxis.z /= length;
//     }
//
//     return rotatedAxis;
// }


