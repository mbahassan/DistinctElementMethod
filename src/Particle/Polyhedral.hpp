#ifndef POLYHEDRAL_PARTICLE_H
#define POLYHEDRAL_PARTICLE_H

#include <cuda_runtime_api.h>

#include "Material/Material.hpp"
#include "Shape/Polytope/Polytope.hpp"
#include "Tools/quaternion/quaternion.hpp"
#include "Tools/AABB/AABB.hpp"

class Polyhedral : public Material {
public:
    __host__ __device__
    Polyhedral() = default;

    /// Move Constructors
    Polyhedral(const Material& material, const Polytope& polytope)
        : Material(material)
    {
        numVertices = polytope.vertices.size();
        numNormals = polytope.normals.size() ;
        numTriangles  = polytope.triangles.size() ;
        numSolids  = polytope.solids.size();

        vertices  = new float3[numVertices]();
        triangles = new int3[numTriangles]();
        normals = new float3[numNormals]();
        solids  = new unsigned int[numSolids]();

        std::ranges::copy(polytope.vertices, vertices);
        std::ranges::copy(polytope.triangles, triangles);
        std::ranges::copy(polytope.normals, normals);
        std::ranges::copy(polytope.solids, solids);
    }

    Polyhedral(Material& material, Polytope& polytope)
        : Material(material)
    {
        numVertices = polytope.vertices.size();
        numNormals = polytope.normals.size();
        numTriangles  = polytope.triangles.size();
        numSolids  = polytope.solids.size();

        vertices  = new float3[numVertices];
        triangles = new int3[numTriangles];
        normals = new float3[numNormals];
        solids  = new unsigned int[numSolids];

        std::ranges::copy(polytope.vertices, vertices);
        std::ranges::copy(polytope.triangles, triangles);
        std::ranges::copy(polytope.normals, normals);
        std::ranges::copy(polytope.solids, solids);
    }


    /// Destructors
    ~Polyhedral() override
    {
        vertices = nullptr;
        triangles = nullptr;
        normals = nullptr;
        solids = nullptr;
    }

    int id = -1 ;

    Shape::ShapeType shapeType = Shape::POLYHEDRAL;

    float3 position{0.f, 0.f, 0.f};      // Position in 3D space

    BoundingBox<float3> boundingBox {};

    Quaternion orientation;

    float3* vertices = nullptr;
    float3* normals = nullptr;
    int3* triangles = nullptr;
    unsigned int* solids = nullptr;

    unsigned int numVertices = 0;
    unsigned int numNormals =0 ;
    unsigned int numSolids = 0;
    unsigned int numTriangles = 0;
};

#endif //POLYHEDRAL_PARTICLE_H