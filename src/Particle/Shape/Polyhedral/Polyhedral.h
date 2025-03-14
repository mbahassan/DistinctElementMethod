//
// Created by iqraa on 13-3-25.
//

#ifndef POLYHEDRAL_H
#define POLYHEDRAL_H
#include <iostream>
#include <string>
#include <vector>
#include <vector_types.h>
#include <Particle/Shape/Shape.hpp>
#include <Tools/ArthmiticOperator/MathOperators.hpp>
#include <Tools/tinyobjloader/objloader.h>

class Polyhedral : public Shape {
public:
    Polyhedral() = default;

    Polyhedral(std::string& file)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        // Load the OBJ file
        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, file.c_str());

        if (!warn.empty()) std::cout << "WARN: " << warn << std::endl;
        if (!err.empty()) std::cerr << "ERR: " << err << std::endl;

        if (!ret) {
            std::cerr << "Failed to load .obj file." << std::endl;
            return;
        }

        // Fill Vertices
        std::vector<float3> vertices;
        for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
            float3 v;
            v.x = attrib.vertices[i + 0];
            v.y = attrib.vertices[i + 1];
            v.z = attrib.vertices[i + 2];
            vertices.push_back(v);
        }

    }

    ~Polyhedral();

    float3 supportMapping(const float3& direction) const
    {
        float maxDistance = vertices[0] & direction;
        int maxIdx = 0;
        for (int i = 1; i < vertices.size(); i += 3) {
            float distance = vertices[i] & direction;
            if (distance > maxDistance) {
                maxDistance = distance;
                maxIdx = i;
            }
        }
        return vertices[maxIdx];
    }

    // Compute volume and center of mass
    void computeVolumeAndCoM(const std::vector<float3>& vertices, const std::vector<Face>& faces,
                             double& totalVolume, float3& centerOfMass) {
        totalVolume = 0.0;
        centerOfMass = {0.0, 0.0, 0.0};

        for (const auto& face : faces) {
            // Convert face to triangle(s)
            for (size_t i = 1; i + 1 < face.indices.size(); ++i) {
                float3 v0 = vertices[face.indices[0]];
                float3 v1 = vertices[face.indices[i]];
                float3 v2 = vertices[face.indices[i + 1]];

                // Tetrahedron: (0, v0, v1, v2)
                double vol = (v0&(v1^v2)) / 6.0;

                float3 centroid = (v0 + v1 + v2) * (1.0 / 4.0); // + origin implicitly (0,0,0)

                centerOfMass = centerOfMass + centroid * vol;
                totalVolume += vol;
            }
        }

        if (std::abs(totalVolume) > 1e-10)
            centerOfMass = centerOfMass * (1.0 / totalVolume);
        else
            std::cerr << "Warning: Total volume is near zero. Possibly open or invalid mesh." << std::endl;

        totalVolume = std::abs(totalVolume); // Use absolute volume
    }

private:
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<int3> faces;
};


#endif //POLYHEDRAL_H
