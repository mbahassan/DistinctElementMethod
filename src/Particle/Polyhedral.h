#ifndef POLYHEDRAL_PARTICLE_H
#define POLYHEDRAL_PARTICLE_H

#include <cuda_runtime_api.h>
#include <thrust/system/detail/sequential/copy.inl>

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
        copy(polytope);
    }

    Polyhedral(Material& material, Polytope& polytope)
        : Material(material)
    {
        copy(polytope);
    }


    /// Destructors
    ~Polyhedral() override
    {
        vertices = nullptr;
        triangles = nullptr;
        normals = nullptr;
        solids = nullptr;
    }


    float3 supportMapping(const float3 &direction) const
    {
        float maxDistance = vertices[0] & direction;
        int maxIdx = 0;
        for (int i = 1; i < numVertices; i++)
        {
            float distance = vertices[i] & direction;
            if (distance > maxDistance) {
                maxDistance = distance;
                maxIdx = i;
            }
        }

        return vertices[maxIdx];
    }

    void setId(const int id) {id_ = id;}

    int getId() const {return id_;}


    Shape::ShapeType shapeType = Shape::POLYHEDRAL;

    float volume = 0.0f;
    float mass = 0.0f;
    float3 position = {0.f, 0.f, 0.f};      // Position in 3D space
    float3 velocity = {0.f, 0.f, 0.f};      // Velocity in 3D space
    float3 force = {0.f, 0.f, 0.f};         // Force in 3D space
    float3 torque = {0.f,0.f,0.f};
    Quaternion orientation;


    float3* vertices = nullptr;
    float3* normals = nullptr;
    int3* triangles = nullptr;
    unsigned int* solids = nullptr;

    unsigned int numVertices = 0;
    unsigned int numNormals  = 0 ;
    unsigned int numSolids   = 0;
    unsigned int numTriangles = 0;


    BoundingBox<float3> boundingBox {};

private:
    int id_ = -1 ;

    /// Copy stl primitives
    void copy(const Polytope& polytope)
    {
        numVertices = polytope.vertices.size();
        numNormals = polytope.normals.size() ;
        numTriangles  = polytope.triangles.size() ;
        numSolids  = polytope.solids.size();

        volume = polytope.volume_;
        boundingBox = {polytope.min_ , polytope.max_};

        vertices  = new float3[numVertices]();
        triangles = new int3[numTriangles]();
        normals = new float3[numNormals]();
        solids  = new unsigned int[numSolids]();

        std::ranges::copy(polytope.vertices, vertices);
        std::ranges::copy(polytope.triangles, triangles);
        std::ranges::copy(polytope.normals, normals);
        std::ranges::copy(polytope.solids, solids);

        computeMass();
    }


    void computeMass() {
        mass = volume * getDensity();
    }
};

#endif //POLYHEDRAL_PARTICLE_H