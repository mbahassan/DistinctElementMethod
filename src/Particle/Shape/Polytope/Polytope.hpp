#ifndef POLYTOPE_H
#define POLYTOPE_H

#include <iostream>
#include <vector>
#include <vector_types.h>
#include <Particle/Shape/Shape.hpp>
#include <Tools/ArthmiticOperator/MathOperators.hpp>


#include "Tools/StlReader/StlReader.h"
#include "Tools/quaternion/quaternion.hpp"

class Polytope : public Shape {
public:
    std::vector<float3> vertices;
    std::vector<int3> triangles;
    std::vector<float3> normals;
    std::vector<unsigned int> solids;



    __host__ __device__
    Polytope()
    {
        setShapeType(POLYHEDRAL);
    }

    ///  Copy constructor
    Polytope(const Polytope& polytope)
     : Shape(polytope) {
        vertices = polytope.vertices;
        triangles = polytope.triangles;
        normals = polytope.normals;
        solids = polytope.solids;
        setShapeType(POLYHEDRAL);
    }

    /// Constructor from File
    explicit Polytope(const std::string &file)
    {
        try {
            stl_reader::ReadStlFile(file.c_str(),
                raw_vertices,
                raw_normals,
                raw_triangles,
                solids);

            // Convert raw vertices to float3
            for (size_t i = 0; i < raw_vertices.size(); i += 3)
            {
                float3 p = {raw_vertices[i], raw_vertices[i+1], raw_vertices[i+2]};
                vertices.push_back(p);
            }

            // Convert raw triangles to int3
            for (size_t i = 0; i < raw_triangles.size(); i += 3)
            {
                int3 p = {raw_triangles[i], raw_triangles[i+1], raw_triangles[i+2]};
                triangles.push_back(p);
            }

            // Convert raw normals to float3
            for (size_t i = 0; i < raw_normals.size(); i += 3) {
                normals.push_back(make_float3(
                    raw_normals[i],
                    raw_normals[i+1],
                    raw_normals[i+2]
                ));
            }

        }
        catch (std::exception& e) {
            std::cout << e.what() << std::endl;
        }

        setShapeType(POLYHEDRAL);

    }

    ~Polytope() override = default;

    float3 supportMapping(const float3 &direction) const
    {
        float maxDistance = vertices[0] & direction;
        int maxIdx = 0;
        for (int i = 1; i < vertices.size(); i++)
        {
            float distance = vertices[i] & direction;
            if (distance > maxDistance) {
                maxDistance = distance;
                maxIdx = i;
            }
        }

        return vertices[maxIdx];
    }

    __host__ __device__
    float getVolume() override { return volume_; }

    __host__ __device__
    float3 getMin() override { return min_; }

    __host__ __device__
    float3 getMax() override { return max_; }

    // Method to get the orientation
    __host__ __device__
    Quaternion getOrientation() const {
        return orientation_;
    }


private:

    float volume_ = 0.0f;
    Quaternion orientation_;
    float3 min_ {0.0f, 0.0f, 0.0f};
    float3 max_ {0.0f, 0.0f, 0.0f};

    std::vector<float> raw_vertices;
    std::vector<int> raw_triangles;
    std::vector<float> raw_normals;
};

#endif //POLYTOPE_H