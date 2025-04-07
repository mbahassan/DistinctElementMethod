#ifndef POLYTOPE_H
#define POLYTOPE_H

#include <iostream>
#include <vector>
#include <vector_types.h>
#include <Particle/Shape/Shape.hpp>
#include <Tools/ArthmiticOperator/MathOperators.hpp>


#include "Tools/StlReader/StlReader.h"
#include "Tools/quaternion/quaternion.hpp"

class Polytope : public Shape
{
public:
    std::vector<float3> vertices;
    std::vector<int3> triangles;
    std::vector<float3> normals;
    std::vector<unsigned int> solids;

    float volume_ = 0.0f;
    float3 min_ {0.0f, 0.0f, 0.0f};
    float3 max_ {0.0f, 0.0f, 0.0f};

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

        calculateVolume();
        calculateMinMax();
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
                position += p;
                vertices.push_back(p);
            }
            position /= static_cast<float>(raw_vertices.size());

            // translate it to center 0,0,0,
            for (auto& v :vertices)
            {
                v -= position;
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
            calculateVolume();
            calculateMinMax();
        }
        catch (std::exception& e) {
            std::cout << e.what() << std::endl;
        }

        setShapeType(POLYHEDRAL);

    }

    ~Polytope() override = default;


    __host__ __device__
    float getVolume() override { return volume_; }

    __host__ __device__
    float3 getMin() override { return min_; }

    __host__ __device__
    float3 getMax() override { return max_; }



private:
    std::vector<float> raw_vertices;
    std::vector<int> raw_triangles;
    std::vector<float> raw_normals;
    float3 position {0.f,0.0f,0.f};

    void calculateVolume() {
        volume_ = 0.0f;
        const float epsilon = 1e-6f;

        // First pass: check if the object is approximately 2D
        bool is_2d = true;

        // Determine which axis might be constant (if any)
        float min_x = std::numeric_limits<float>::max(), max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max(), max_y = std::numeric_limits<float>::lowest();
        float min_z = std::numeric_limits<float>::max(), max_z = std::numeric_limits<float>::lowest();

        for (const auto& v : vertices) {
            min_x = std::min(min_x, v.x); max_x = std::max(max_x, v.x);
            min_y = std::min(min_y, v.y); max_y = std::max(max_y, v.y);
            min_z = std::min(min_z, v.z); max_z = std::max(max_z, v.z);
        }

        float x_range = max_x - min_x;
        float y_range = max_y - min_y;
        float z_range = max_z - min_z;

        // If any dimension is very small compared to others, the object is likely 2D
        if (z_range < epsilon * std::max(x_range, y_range) ||
            y_range < epsilon * std::max(x_range, z_range) ||
            x_range < epsilon * std::max(y_range, z_range)) {
            is_2d = true;
            } else {
                is_2d = false;
            }

        // If it's 2D, calculate area
        if (is_2d)
        {
            for (const auto& tri : triangles)
            {
                const float3& v0 = vertices[tri.x];
                const float3& v1 = vertices[tri.y];
                const float3& v2 = vertices[tri.z];

                // Calculate normal vector via cross product
                float nx = (v1.y - v0.y) * (v2.z - v0.z) - (v1.z - v0.z) * (v2.y - v0.y);
                float ny = (v1.z - v0.z) * (v2.x - v0.x) - (v1.x - v0.x) * (v2.z - v0.z);
                float nz = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);

                // Find magnitude of normal vector
                float normal_length = std::sqrt(nx*nx + ny*ny + nz*nz);

                // Area of triangle is half the length of the cross product
                volume_ += 0.5f * normal_length;
            }
        } else
        {
            for (const auto& tri : triangles)
            {
                const float3& v0 = vertices[tri.x];
                const float3& v1 = vertices[tri.y];
                const float3& v2 = vertices[tri.z];

                volume_ += (v0.x * (v1.y * v2.z - v1.z * v2.y) +
                           v0.y * (v1.z * v2.x - v1.x * v2.z) +
                           v0.z * (v1.x * v2.y - v1.y * v2.x)) / 6.0f;
            }
            volume_ = std::fabs(volume_);
        }
    }

    void calculateMinMax() {
        min_ = float3(std::numeric_limits<float>::max());
        max_ = float3(std::numeric_limits<float>::lowest());

        for (const auto& v : vertices)
        {
            min_.x = std::min(min_.x, v.x);
            min_.y = std::min(min_.y, v.y);
            min_.z = std::min(min_.z, v.z);

            max_.x = std::max(max_.x, v.x);
            max_.y = std::max(max_.y, v.y);
            max_.z = std::max(max_.z, v.z);
        }
    }
};

#endif //POLYTOPE_H