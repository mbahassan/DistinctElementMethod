//
// Created by iqraa on 13-3-25.
//

#ifndef POLYTOPE_H
#define POLYTOPE_H
#include <cfloat>
#include <string>
#include <vector_types.h>
#include <Particle/Shape/Shape.hpp>
#include <Tools/ArthmiticOperator/MathOperators.hpp>


#include "Tools/objreader/ObjReader.h"
#include "Tools/quaternion/quaternion.hpp"


class Polytope : public Shape, public ObjReader
{
public:

    Polytope()
    {
        setShapeType(POLYHEDRAL);
    }

    /// Constructor from File
    explicit Polytope(const std::string &file) : ObjReader(file)
    {
        setShapeType(POLYHEDRAL);
        // computeVolumeAndCoM(vertices, faces, volume, centerOfMass);
    }

    /// Move Constructors
    Polytope(const Polytope& polytope) {
        setShapeType(POLYHEDRAL);
        // computeVolumeAndCoM(vertices, faces, volume, centerOfMass);
    }

    Polytope(Polytope& polytope) {
        setShapeType(POLYHEDRAL);
        // computeVolumeAndCoM(vertices, faces, volume, centerOfMass);
    }

    ~Polytope() override = default;

    float3 supportMapping(const float3 &direction) const
    {
        float maxDistance = getVertices()[0] & direction;
        int maxIdx = 0;
        for (int i = 1; i < getVerticesCount(); i += 1) {
            float distance = getVertex(i) & direction;
            if (distance > maxDistance) {
                maxDistance = distance;
                maxIdx = i;
            }
        }

        return getVertex(maxIdx);
    }

    __host__ __device__
    float getVolume() override { return getObjVolume(); }

    __host__ __device__
    float3 getMin() override { return getObjMin(); }

    __host__ __device__
    float3 getMax() override { return getObjMax(); }

    __host__ __device__
    float3 getCenterOfMass() const { return getObjCOM(); }

    // Method to get the orientation
    __host__ __device__
    Quaternion getOrientation() const {
        return getObjOrientation();
    }


};


#endif //POLYTOPE_H
